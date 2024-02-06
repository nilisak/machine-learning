import argparse
from datetime import datetime
import os
import sys
import json


from torch.utils.data import Subset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets
import wandb
from sklearn.metrics import confusion_matrix
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
import random
from itertools import cycle
from torchvision import transforms


if "LOG_PATH" in os.environ:
    os.makedirs(os.path.dirname(os.environ["LOG_PATH"]), exist_ok=True)
    log = open(os.environ["LOG_PATH"], "a")
    sys.stdout = log
    sys.stderr = log


class CIFARNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def get_wandb_key():
    json_file = "../wandb_key.json"
    if os.path.isfile(json_file):
        with open(json_file, "r") as f:
            return json.load(f)
    elif "WANDB_KEY" in os.environ:
        return os.environ["WANDB_KEY"]


def save_checkpoint(state, is_best, checkpoint_dir=".", filename="best_checkpoint.pth"):
    best_path = os.path.join(checkpoint_dir, "best_checkpoint.pth")
    torch.save(state, best_path)


def load_best_model(checkpoint_dir=".", input_channels=10):
    best_model_path = os.path.join(checkpoint_dir, "best_checkpoint.pth")
    checkpoint = torch.load(best_model_path)
    model = CIFARNet(input_channels)
    model.load_state_dict(checkpoint["state_dict"])
    return model


def log_predictions(model, loader, device, num_samples_per_class=5):
    classes = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    model.eval()
    class_samples = {classname: 0 for classname in classes}  # Initialize sample count per class
    wandb_table = wandb.Table(
        columns=["Image", "Ground Truth", "Prediction"]
    )  # Initialize wandb.Table

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            for i in range(images.size(0)):
                label = labels[i].item()
                pred = preds[i].item()

                if class_samples[classes[label]] < num_samples_per_class:
                    # Convert image tensor to PIL Image
                    img_pil = transforms.ToPILImage()(images[i].cpu()).convert("RGB")
                    # Log image, true label name, and predicted label name
                    wandb_table.add_data(wandb.Image(img_pil), classes[label], classes[pred])
                    class_samples[classes[label]] += 1

                if all(value >= num_samples_per_class for value in class_samples.values()):
                    break  # Stop if we've logged enough samples for each class

    # Log the table to Weights & Biases
    wandb.log({"Validation Samples": wandb_table})


def compute_confusion_matrix(model, loader, device, class_names, epoch):
    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            _, true = torch.max(labels, 1)

            all_preds.extend(preds.cpu().numpy())
            all_true.extend(true.cpu().numpy())

    # Log the confusion matrix
    wandb.log(
        {
            f"confusion_matrix{epoch+1}": wandb.plot.confusion_matrix(
                preds=all_preds,
                y_true=all_true,
                class_names=class_names,
                title="Normalized Confusion Matrix",
            )
        }
    )
    cm = confusion_matrix(all_true, all_preds)

    # Normalize the confusion matrix over targets (rows)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    print(cm_normalized)


class ResampledDataset(Dataset):
    def __init__(self, dataset):
        class_0_samples = [sample for sample in dataset if sample[1][1] == 0]
        class_1_samples = [sample for sample in dataset if sample[1][1] == 1]

        # Determine minority and majority classes
        minority_class, majority_class = (
            (class_1_samples, class_0_samples)
            if len(class_1_samples) < len(class_0_samples)
            else (class_0_samples, class_1_samples)
        )

        # Calculate how many samples we need to replicate from the minority class
        extra_samples_needed = len(majority_class) - len(minority_class)

        # Shuffle the minority class samples to randomize the order
        random.shuffle(minority_class)

        # Create an iterator that cycles through the minority class
        minority_cycle = cycle(minority_class)

        # Use the cycle iterator to replicate the minority samples until we have enough
        replicated_samples = [next(minority_cycle) for _ in range(extra_samples_needed)]

        # Combine the original majority class with both the original and replicated minority samples
        self.resampled_data = majority_class + minority_class + replicated_samples

        random.shuffle(self.resampled_data)

    def __len__(self):
        return len(self.resampled_data)

    def __getitem__(self, idx):
        return self.resampled_data[idx]


def main(args):
    wandb.login(key=get_wandb_key())
    wandb.init(project="ms-in-dnns-cifar-net", config=args, name=args.run_name)

    torch.manual_seed(0xDEADBEEF)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]  # Normalize the tensors
    )

    # Load the CIFAR10 dataset with the defined transform
    train_dataset = datasets.CIFAR10(
        root="data",
        train=True,  # Specify to load the training data
        download=False,  # Assuming the dataset is already downloaded
        transform=transform,  # Apply the defined transformation
    )

    val_dataset = datasets.CIFAR10(
        root="data",
        train=False,  # Specify to load the training data
        download=False,  # Assuming the dataset is already downloaded
        transform=transform,  # Apply the defined transformation
    )
    if args.small:
        # Assuming `dataset` is your loaded dataset
        total_items = len(train_dataset)  # Total number of items in the dataset
        percentage = 0.01  # Percentage of the dataset you want to select
        subset_size = int(total_items * (percentage / 100))  # Calculate the size of the subset
        indices = torch.randperm(total_items).tolist()
        subset_indices = indices[:subset_size]
        train_dataset = Subset(train_dataset, subset_indices)

        total_items = len(val_dataset)  # Total number of items in the dataset
        percentage = 1  # Percentage of the dataset you want to select
        subset_size = int(total_items * (percentage / 100))  # Calculate the size of the subset
        indices = torch.randperm(total_items).tolist()
        subset_indices = indices[:subset_size]
        val_dataset = Subset(val_dataset, subset_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=2)

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    model = CIFARNet()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # best_val_loss = float("inf")
    best_acc = 0.0
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma_lr)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.cpu().detach().item()
        train_loss = total_loss / len(train_loader)

        model.eval()
        total_loss = 0.0
        true_pos = 0
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            total_loss += loss.cpu().item()
            preds = torch.argmax(outputs, dim=-1)
            true_pos += (preds == torch.argmax(labels, dim=-1)).cpu().sum()
        acc = true_pos / len(val_dataset)
        val_loss = total_loss / len(val_loader)
        print(
            f"Epoch [{epoch+1}/{args.epochs}]",
            f"Train Loss: {train_loss:.4f}",
            f"Val Loss: {val_loss:.4f}",
            f"Val Accuracy: {acc:.4f}",
        )

        is_best = acc > best_acc
        if is_best:
            best_acc = acc
            checkpoint_dir = (
                os.path.dirname(os.environ["LOG_PATH"]) if "LOG_PATH" in os.environ else "."
            )
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "acc": acc,
                },
                is_best,
                checkpoint_dir,
            )

        wandb.log(
            {"loss": {"train": train_loss, "val": val_loss, "acc": acc, "best acc": best_acc}},
            step=epoch + 1,
        )

        # scheduler.step()

    model.eval()
    true_pos = 0
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)

        preds = torch.argmax(outputs, dim=-1)
        true_pos += (preds == torch.argmax(labels, dim=-1)).cpu().sum()
    acc = true_pos / len(val_dataset)
    print(f"Accuracy at the end of training: {best_acc:.4f}")
    wandb.log({"final": {"val_acc": best_acc}})

    best_model = load_best_model(checkpoint_dir)
    best_model = best_model.to(device)
    log_predictions(
        best_model,
        val_loader,
        device,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--small", type=bool, default=True)
    parser.add_argument("--lr", type=float, default=1e-3)
    if "CREATION_TIMESTAMP" in os.environ:
        timestamp = os.environ["CREATION_TIMESTAMP"]
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parser.add_argument("--run-name", type=str, default=timestamp)
    args = parser.parse_args()
    main(args)

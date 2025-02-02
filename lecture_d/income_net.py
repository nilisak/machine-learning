import argparse
from datetime import datetime
import os
import sys
import json
import pathlib as pl


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import wandb
from sklearn.metrics import confusion_matrix
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
import random
from itertools import cycle

if "LOG_PATH" in os.environ:
    os.makedirs(os.path.dirname(os.environ["LOG_PATH"]), exist_ok=True)
    log = open(os.environ["LOG_PATH"], "a")
    sys.stdout = log
    sys.stderr = log


class IncomeNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class AdultDataset(Dataset):
    """Adult UCI dataset, download data from https://archive.ics.uci.edu/dataset/2/adult"""

    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)

        # one-hot encoding of categorical variables (including label)
        df = pd.get_dummies(df).astype("int32")

        data_columns = df.columns[:-2]
        labels_column = df.columns[-2:]
        self.data = torch.tensor(df[data_columns].values, dtype=torch.float32)
        self.labels = torch.tensor(df[labels_column].values, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def get_wandb_key():
    json_file = pl.Path("..", "wandb_key.json")
    if json_file.is_file():
        with open(json_file, "r") as f:
            return json.load(f)
    elif "WANDB_KEY" in os.environ:
        return os.environ["WANDB_KEY"]


def save_checkpoint(state, is_best, checkpoint_dir=".", filename="last_checkpoint.pth"):
    last_path = os.path.join(checkpoint_dir, filename)
    torch.save(state, last_path)
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_checkpoint.pth")
        torch.save(state, best_path)


def load_best_model(checkpoint_dir=".", input_size=1, input_channels=1):
    best_model_path = os.path.join(checkpoint_dir, "best_checkpoint.pth")
    checkpoint = torch.load(best_model_path)
    model = IncomeNet(input_size, input_channels)
    model.load_state_dict(checkpoint["state_dict"])
    return model


def log_predictions(model, loader, device, num_samples=100):
    model.eval()
    class_samples = {}  # A dictionary to store samples for each class
    predictions = []  # A list to store predictions

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            _, true_labels = torch.max(labels, 1)

            for i in range(inputs.size(0)):
                label = true_labels[i].item()
                if label not in class_samples:
                    class_samples[label] = 0
                if class_samples[label] < num_samples:
                    class_samples[label] += 1
                    predictions.append((inputs[i], true_labels[i], preds[i]))

    # Create a wandb.Table
    columns = ["Input Features", "True Label", "Predicted Label"]
    wandb_table = wandb.Table(columns=columns)

    for input_features, true_label, predicted_label in predictions:
        # Log to wandb
        wandb_table.add_data(
            input_features.cpu().numpy(), true_label.item(), predicted_label.item()
        )
        # Log to terminal
        print(f"True Label: {true_label.item()}, Predicted Label: {predicted_label.item()}")

    # Log the table to W&B
    wandb.log({"predictions": wandb_table})


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
    wandb.init(project="ms-in-dnns-income-net", config=args, name=args.run_name)

    torch.manual_seed(0xDEADBEEF)

    if "LOG_PATH" in os.environ:
        data_file = pl.PurePosixPath("/gcs", "msindnn_staging", "adult_data", "adult.data")
    else:
        data_file = pl.PurePath("..", "data", "adult_data", "adult.data")

    entire_dataset = AdultDataset(str(data_file))
    if args.resample:
        entire_dataset = ResampledDataset(entire_dataset)
    train_dataset, val_dataset = random_split(
        entire_dataset, [args.train_share, 1 - args.train_share]
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=2)

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    input_size = train_dataset[0][0].shape[0]
    num_classes = train_dataset[0][1].shape[0]
    model = IncomeNet(input_size, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([args.weight, 1 - args.weight], dtype=torch.float)
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # best_val_loss = float("inf")
    best_acc = 0.0
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma_lr)
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

        wandb.log({"loss": {"train": train_loss, "val": val_loss, "acc": acc}}, step=epoch + 1)
        if epoch == 9:
            compute_confusion_matrix(model, val_loader, device, class_names=["0", "1"], epoch=epoch)
        scheduler.step()

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

    checkpoint_dir = os.path.dirname(os.environ["LOG_PATH"]) if "LOG_PATH" in os.environ else "."
    best_model = load_best_model(checkpoint_dir, input_size, num_classes)
    best_model = best_model.to(device)
    log_predictions(best_model, val_loader, device, num_samples=100)
    compute_confusion_matrix(best_model, val_loader, device, class_names=["0", "1"], epoch=epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-share", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--gamma-lr", type=float, default=1)
    parser.add_argument("--step-size", type=int, default=1)
    parser.add_argument("--weight", type=float, default=0.5)
    parser.add_argument("--resample", type=bool, default=False)

    if "CREATION_TIMESTAMP" in os.environ:
        timestamp = os.environ["CREATION_TIMESTAMP"]
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parser.add_argument("--run-name", type=str, default=timestamp)
    args = parser.parse_args()
    main(args)

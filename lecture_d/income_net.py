import argparse
from datetime import datetime
import os
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import wandb

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
    json_file = "../wandb_key.json"
    if os.path.isfile(json_file):
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


def load_best_model(checkpoint_dir="."):
    best_model_path = os.path.join(checkpoint_dir, "best_checkpoint.pth")
    checkpoint = torch.load(best_model_path)
    model = IncomeNet(input_size, num_classes)  # Assuming you have these variables set
    model.load_state_dict(checkpoint["state_dict"])
    return model


def log_predictions(model, loader, device, num_samples=10):
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


def main(args):
    wandb.login(key=get_wandb_key())
    wandb.init(project="ms-in-dnns-income-net", config=args, name=args.run_name)

    torch.manual_seed(0xDEADBEEF)

    if "LOG_PATH" in os.environ:
        data_root = "/gcs/msindnn_staging/adult_data"
    else:
        data_root = "../data/adult_data"
    entire_dataset = AdultDataset(os.path.join(data_root, "adult.data"))
    train_dataset, val_dataset = random_split(
        entire_dataset, [args.train_share, 1 - args.train_share]
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=2)

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    model = IncomeNet(train_dataset[0][0].shape[0], train_dataset[0][1].shape[0])
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # best_val_loss = float("inf")
    best_acc = 0.0
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
    best_model = load_best_model(checkpoint_dir)
    best_model = best_model.to(device)
    log_predictions(best_model, val_loader, device, num_samples=10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-share", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    if "CREATION_TIMESTAMP" in os.environ:
        timestamp = os.environ["CREATION_TIMESTAMP"]
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parser.add_argument("--run-name", type=str, default=timestamp)
    args = parser.parse_args()
    main(args)

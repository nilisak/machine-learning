from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import lightning as L
import torch
from torch.utils.data import Subset


class CIFARDataModule(L.LightningDataModule):
    def __init__(self, data_root: str = "./data", batch_size: int = 32, small="Y", augmented="Y"):
        super().__init__()
        self.data_dir = data_root
        self.batch_size = batch_size
        self.small = small == "Y"
        self.augmented = augmented == "Y"

        self.train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                ),  # Normalize the dataset using the mean and std of CIFAR10
            ]
        )
        if self.augmented:
            self.train_transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(degrees=(-10, 10)),
                    transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0), ratio=(0.8, 1.2)),
                    transforms.ColorJitter(
                        brightness=(0.8, 1.2),
                        contrast=(0.8, 1.2),
                        saturation=(0.8, 1.2),
                        hue=(-0.1, 0.1),
                    ),
                    transforms.ToTensor(),  # Convert images to PyTorch tensors
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                ),  # Normalize the dataset using the mean and std of CIFAR10
            ]
        )

    def prepare_data(self):
        # Downloads the CIFAR10 dataset
        datasets.CIFAR10(root=self.data_dir, train=True, download=True)
        datasets.CIFAR10(root=self.data_dir, train=False, download=True)

    def setup(self, stage):

        # Load and split the dataset
        self.train_dataset = datasets.CIFAR10(
            root=self.data_dir, train=True, download=True, transform=self.train_transform
        )
        self.val_dataset = datasets.CIFAR10(
            root=self.data_dir, train=False, download=True, transform=self.transform
        )
        if self.small:
            # Assuming `dataset` is your loaded dataset
            total_items = len(self.train_dataset)  # Total number of items in the dataset
            percentage = 0.01  # Percentage of the dataset you want to select
            subset_size = int(total_items * (percentage / 100))  # Calculate the size of the subset
            indices = torch.randperm(total_items).tolist()
            subset_indices = indices[:subset_size]
            self.train_dataset = Subset(self.train_dataset, subset_indices)

            total_items = len(self.val_dataset)  # Total number of items in the dataset
            percentage = 1  # Percentage of the dataset you want to select
            subset_size = int(total_items * (percentage / 100))  # Calculate the size of the subset
            indices = torch.randperm(total_items).tolist()
            subset_indices = indices[:subset_size]
            self.val_dataset = Subset(self.val_dataset, subset_indices)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return self.val_dataloader()

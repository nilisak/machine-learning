import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L
import torchmetrics
import wandb
from torchvision import models
import os


class CIFAR10VGG(nn.Module):
    def __init__(self, num_classes, pretrained):

        super().__init__()

        if "LOG_PATH" in os.environ:

            cache_dir = "/gcs/isakbucket/"
        else:
            cache_dir = "."

        torch.hub.set_dir(cache_dir)
        # Load the pretrained VGG-16 model features
        original_model = models.vgg16_bn(pretrained=pretrained)
        self.features = original_model.features

        # Replace avgpool with an identity layer
        self.avgpool = nn.Identity()

        # Custom classifier for CIFAR-10
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class CIFARVGGModule(L.LightningModule):
    def __init__(self, lr=1e-3, num_classes=10, pretrained="Y"):
        super().__init__()
        self.save_hyperparameters()
        self.model = CIFAR10VGG(num_classes=num_classes, pretrained=pretrained == "Y")
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.num_classes = num_classes
        metrics = torchmetrics.MetricCollection(
            {
                "acc": torchmetrics.Accuracy(num_classes=num_classes, task="multiclass"),
                "precision": torchmetrics.Precision(
                    num_classes=num_classes, average="macro", task="multiclass"
                ),
                "recall": torchmetrics.Recall(
                    num_classes=num_classes, average="macro", task="multiclass"
                ),
            }
        )
        self.confmat = torchmetrics.ConfusionMatrix(num_classes=num_classes, task="multiclass")

        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.best_metrics = metrics.clone(prefix="best/")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)  # Convert outputs to predicted class indices
        self.train_metrics.update(preds, labels)  # Correct order: preds first, then labels
        self.log("train/loss", loss, on_epoch=True, on_step=False)
        self.log_dict(self.train_metrics, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=-1)

        acc = (preds == labels).sum() / inputs.shape[0]
        self.val_metrics.update(preds, labels)
        self.log("val/loss", loss, on_epoch=True, on_step=False)
        self.log("val/acc_manual", acc, on_epoch=True, on_step=False)
        self.log_dict(self.val_metrics, on_epoch=True, on_step=False)
        self.log("step", float(self.current_epoch + 1), on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=-1)

        self.best_metrics.update(preds, labels)
        self.log("best/loss", loss, on_epoch=True, on_step=False)
        self.log_dict(self.best_metrics, on_epoch=True, on_step=False)

        self.confmat.update(preds, labels)

    def on_test_epoch_end(self):
        conf_matrix = self.confmat.compute()
        class_names = [str(i) for i in range(self.num_classes)]

        # Initialize an empty list to store the data for each cell in the confusion matrix
        data = []
        for actual_class in range(len(class_names)):
            for predicted_class in range(len(class_names)):
                data.append(
                    [
                        class_names[actual_class],
                        class_names[predicted_class],
                        int(conf_matrix[actual_class, predicted_class]),
                    ]
                )

        # Log the data as a custom table in Wandb
        wandb.log(
            {
                "confusion_matrix_detailed": wandb.Table(
                    data=data, columns=["Actual", "Predicted", "Count"]
                )
            }
        )

        self.confmat.reset()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

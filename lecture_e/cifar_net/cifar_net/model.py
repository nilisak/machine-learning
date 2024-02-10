import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L
import torchmetrics
import wandb


class CIFARNet(nn.Module):
    def __init__(self, num_classes=10, batch_norm=False, dropout=False):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3) if dropout else nn.Identity(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3) if dropout else nn.Identity(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3) if dropout else nn.Identity(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3) if dropout else nn.Identity(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3) if dropout else nn.Identity(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3) if dropout else nn.Identity(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class CIFARNetModule(L.LightningModule):
    def __init__(self, lr=1e-3, batch_norm="N", dropout="N", num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.model = CIFARNet(
            num_classes=num_classes, batch_norm=batch_norm == "Y", dropout=dropout == "Y"
        )
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

        conf_data = conf_matrix.cpu().numpy()
        data = [[conf_data[i, j] for j in range(len(class_names))] for i in range(len(class_names))]

        wandb.log({"confusion_matrix": wandb.Table(data=data, columns=class_names)})

        self.confmat.reset()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

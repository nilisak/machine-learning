# interpolate_mnist.py
import torch
import matplotlib.pyplot as plt
from pytorch_lightning import seed_everything
from mnist_flow.data import MNISTDataModule
from mnist_flow.model import MNISTFlowModule
import wandb
from mnist_flow.utils import get_wandb_key
from lightning.pytorch.cli import LightningArgumentParser
from datetime import datetime
import os
from mnist_flow.data import KMNISTDataModule
import torchvision
from torchvision.utils import make_grid
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix


def logit_transform(x, constraint=0.9, reverse=False):
    """Transforms data from [0, 1] into unbounded space.

    Restricts data into [0.05, 0.95].
    Calculates logit(alpha+(1-alpha)*x).

    Args:
        x: input tensor.
        constraint: data constraint before logit.
        reverse: True if transform data back to [0, 1].
    Returns:
        transformed tensor and log-determinant of Jacobian from the transform.
        (if reverse=True, no log-determinant is returned.)
    """
    if reverse:
        x = 1.0 / (torch.exp(-x) + 1.0)  # [0.05, 0.95]
        x *= 2.0  # [0.1, 1.9]
        x -= 1.0  # [-0.9, 0.9]
        x /= constraint  # [-1, 1]
        x += 1.0  # [0, 2]
        x /= 2.0  # [0, 1]
        return x, 0
    else:
        [B, C, H, W] = list(x.size())

        # dequantization
        noise = torch.distributions.Uniform(0.0, 1.0).sample((B, C, H, W)).to(x.device)
        x = (x * 255.0 + noise) / 256.0

        # restrict data
        x *= 2.0  # [0, 2]
        x -= 1.0  # [-1, 1]
        x *= constraint  # [-0.9, 0.9]
        x += 1.0  # [0.1, 1.9]
        x /= 2.0  # [0.05, 0.95]

        # logit data
        logit_x = torch.log(x) - torch.log(1.0 - x)

        # log-determinant of Jacobian from the transform
        pre_logit_scale = torch.tensor(np.log(constraint) - np.log(1.0 - constraint))
        log_diag_J = F.softplus(logit_x) + F.softplus(-logit_x) - F.softplus(-pre_logit_scale)

        return logit_x, torch.sum(log_diag_J, dim=(1, 2, 3))


def compute_confusion_matrix(true_labels, predicted_labels):
    """
    Computes a confusion matrix using numpy for binary classification.

    Args:
    true_labels (numpy.ndarray): Ground truth binary labels.
    predicted_labels (numpy.ndarray): Predicted binary labels.

    Returns:
    numpy.ndarray: Confusion matrix ([TN, FP], [FN, TP])
    """
    # Convert boolean or non-integer arrays to integers
    true_labels = true_labels.astype(int)
    predicted_labels = predicted_labels.astype(int)

    K = len(np.unique(true_labels))  # Number of classes
    result = np.zeros((K, K))

    for i in range(len(true_labels)):
        result[true_labels[i]][predicted_labels[i]] += 1

    return result


def compute_nll(model, dataloader, device="cpu"):
    model.eval()  # Set the model to evaluation mode
    model.to(device)  # Ensure the model is on the correct device
    total_nll = []
    with torch.no_grad():  # Disable gradient computation
        for imgs, _ in dataloader:  # Assume the DataLoader yields (images, labels)
            imgs = imgs.to(device)  # Move images to the correct device
            input_image, _ = logit_transform(imgs, reverse=False)  # Apply logit transform directly
            imgs_z, ldj = model(input_image, reverse=False)  # Forward pass through the model
            nll = model.nll(imgs_z, ldj, mean=False)  # Compute NLL for each sample
            total_nll.extend(nll.detach().cpu().numpy())  # Accumulate NLL values
    return np.array(total_nll)  # Return NLL values as a NumPy array


def main(args):
    seed_everything(0xDEADBEEF)
    mnist_data_module = MNISTDataModule(**vars(args.data))
    kmnist_data_module = KMNISTDataModule(**vars(args.datak))

    mnist_data_module.setup(stage="")
    kmnist_data_module.setup(stage="")

    # Get a batch of MNIST and KMNIST samples
    mnist_samples = next(iter(mnist_data_module.train_dataloader()))
    kmnist_samples = next(iter(kmnist_data_module.train_dataloader()))

    # Create grids
    mnist_grid = make_grid(mnist_samples[0], nrow=8)  # mnist_samples[0] contains images
    kmnist_grid = make_grid(kmnist_samples[0], nrow=8)  # kmnist_samples[0] contains images

    # Function to convert a tensor to a plot
    def tensor_to_plot(tensor):
        plt.figure(figsize=(10, 5))
        plt.imshow(torchvision.transforms.functional.to_pil_image(tensor))
        plt.axis("off")
        plt.ioff()  # Turn interactive plotting off
        return plt

    # Log images to WandB
    wandb.login(key=get_wandb_key())
    wandb.init(project="mnist_anomaly")
    wandb.log({"MNIST Samples": [wandb.Image(tensor_to_plot(mnist_grid), caption="MNIST Samples")]})
    wandb.log(
        {"KMNIST Samples": [wandb.Image(tensor_to_plot(kmnist_grid), caption="KMNIST Samples")]}
    )

    model = MNISTFlowModule.load_from_checkpoint(args.ckpt_path)

    mnist_data = mnist_data_module.val_dataloader()
    kmnist_data = kmnist_data_module.val_dataloader()

    mnist_nll = compute_nll(model, mnist_data)
    kmnist_nll = compute_nll(model, kmnist_data)

    # Define threshold as a percentile of MNIST NLLs
    threshold = np.percentile(mnist_nll, 95)

    # Classify samples based on the threshold
    mnist_anomalies = mnist_nll > threshold
    kmnist_anomalies = kmnist_nll > threshold

    # True labels: MNIST (0), KMNIST (1)
    true_labels = np.concatenate([np.zeros(len(mnist_nll)), np.ones(len(kmnist_nll))])
    predicted_labels = np.concatenate([mnist_anomalies, kmnist_anomalies])

    # Log the confusion matrix to WandB
    wandb.sklearn.plot_confusion_matrix(true_labels, predicted_labels, ["MNIST", "KMNIST"])

    # Log ten false positives
    fp_indices = np.where((kmnist_nll < threshold))[0][:10]  # Get first 10 false positives
    fp_images = torch.stack([kmnist_data_module.val_dataset[i][0] for i in fp_indices])
    fp_grid = make_grid(fp_images, nrow=5)
    wandb.log({"False Positives": [wandb.Image(fp_grid, caption="False Positives")]})


if __name__ == "__main__":
    parser = LightningArgumentParser()

    parser.add_lightning_class_args(MNISTFlowModule, "model")
    parser.add_lightning_class_args(MNISTDataModule, "data")
    parser.add_lightning_class_args(MNISTDataModule, "datak")

    # Set default data path based on environment
    default_data_path = (
        "/gcs/msindnn_staging/mnist_data" if "LOG_PATH" in os.environ else "../../../data/mnist"
    )
    parser.set_defaults({"data.data_root": default_data_path})

    default_data_path_k = (
        "/gcs/msindnn_staging/kmnist_data" if "LOG_PATH" in os.environ else "../../../data/kmnist"
    )
    parser.set_defaults({"datak.data_root": default_data_path_k})

    # Include run-name and ckpt-path arguments
    timestamp = os.environ.get("CREATION_TIMESTAMP", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    parser.add_argument("--run-name", type=str, default=timestamp)
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to the checkpoint file.")

    args = parser.parse_args()
    main(args)

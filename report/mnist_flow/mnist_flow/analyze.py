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


def compute_jacobian(inputs, model):
    """
    Computes the Jacobian matrix of the model output with respect to the input.

    Args:
        inputs (torch.Tensor): Input tensor for which the Jacobian is computed.
        model (torch.nn.Module): Model for which the Jacobian is computed.

    Returns:
        torch.Tensor: The Jacobian matrix.
    """
    # Ensure model is in evaluation mode to disable dropout, batchnorm, etc.
    model.eval()

    # Ensure inputs require gradients
    inputs.requires_grad_(True)

    # Forward pass through the model
    outputs, _ = model(inputs, reverse=False)

    jacobian = []
    for i in range(outputs.size(1)):
        # Zero gradients from previous iterations
        model.zero_grad()

        # Compute gradient of outputs with respect to inputs
        output = outputs[:, i].sum()
        output.backward(retain_graph=True)

        # Copy the gradients (which represent one row of the Jacobian)
        jacobian.append(inputs.grad.data.clone())

        # Zero gradients specifically for inputs to compute next row of Jacobian
        inputs.grad.data.zero_()

    # Stack to form the Jacobian matrix [batch_size, output_dim, input_dim]
    return torch.stack(jacobian, dim=1)


def compute_svd(jacobian, sample_idx=0):
    """
    Compute the Singular Value Decomposition (SVD) of the Jacobian matrix for a given sample.

    Args:
        jacobian (torch.Tensor): The Jacobian matrix with shape [batch_size, output_dim, 1, height, width].
        sample_idx (int): Index of the sample in the batch for which to compute the SVD.

    Returns:
        U, S, V: The singular value decomposition of the Jacobian matrix for the specified sample.
    """
    # Select the Jacobian for the specified sample
    # Reshape to 2D matrix [output_dim, height*width]
    jacobian_sample = jacobian[sample_idx].squeeze().view(jacobian.size(1), -1)

    # Compute SVD
    U, S, V = torch.svd(jacobian_sample)

    return U, S, V


# Example usage
# Assuming `jacobian` is your computed Jacobian with shape [10, 8, 1, 28, 28]
# Compute SVD for the first sample in the batch


def main(args):
    # seed_everything(42)

    # Load the model (adjust path to your checkpoint)
    model = MNISTFlowModule.load_from_checkpoint(args.ckpt_path)

    # Prepare a data module and load a batch of data
    mnist_data_module = MNISTDataModule(batch_size=1, data_root="path_to_data")
    mnist_data_module.setup(stage="test")
    data_loader = mnist_data_module.test_dataloader()
    batch = next(iter(data_loader))
    inputs, _ = batch

    # Compute the logit transformation (if necessary)
    inputs_logit, _ = logit_transform(inputs, reverse=False)

    # Compute the Jacobian matrix for the batch of inputs
    jacobian = compute_jacobian(inputs_logit, model)

    U, S, V = compute_svd(jacobian, sample_idx=0)

    print(f"Jacobian shape: {jacobian.shape}")
    print("Singular values:", S)


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

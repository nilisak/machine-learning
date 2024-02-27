import torch

from mnist_flow.data import MNISTDataModule
from mnist_flow.model import MNISTFlowModule
from mpl_toolkits.axes_grid1 import make_axes_locatable
from lightning.pytorch.cli import LightningArgumentParser
from datetime import datetime
import os
import matplotlib.pyplot as plt
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


def visualize_principal_directions_with_mnist_image_and_average_in_grid(
    V, original_image, label, image_shape=(28, 28), num_vectors=8
):
    """
    Visualize the original MNIST image, the first few principal directions from the matrix V with colorbars,
    and the average of all principal directions, arranging 4 subplots per row. Color mapping is made more aggressive.

    Parameters:
    - V: The matrix containing right singular vectors (n x n), expected to be a PyTorch tensor.
    - original_image: The original MNIST image to be visualized.
    - label: The label for the image being visualized.
    - image_shape: The shape of the images (height, width).
    - num_vectors: The number of principal directions to visualize.
    """
    # Calculate aggressive color mapping range based on the standard deviation of V's elements
    std_factor = 1  # Adjust this factor to make color mapping more or less aggressive
    mean_val = V.mean().item()
    std_val = V.std().item()
    vmin, vmax = mean_val - (std_factor * std_val), mean_val + (std_factor * std_val)

    # Adjust the layout to have each subplot in its own row
    total_plots = num_vectors + 2
    nrows = (total_plots + 3) // 4
    ncols = 4

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    axes = axes.flatten()

    # Display the original MNIST image
    ax = axes[0]
    original_image_np = (
        original_image.cpu().numpy() if isinstance(original_image, torch.Tensor) else original_image
    )
    ax.imshow(original_image_np.reshape(image_shape), cmap="gray")
    ax.set_title("Original Image")
    ax.axis("off")

    for i in range(num_vectors):
        ax = axes[i + 1]
        principal_direction = V[:, i].cpu().numpy().reshape(image_shape)
        im = ax.imshow(principal_direction, cmap="coolwarm", vmin=vmin, vmax=vmax)
        ax.set_title(f"PD {i+1}")
        ax.axis("off")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    # Display the average of all columns in V in the last subplot
    V_average = V.mean(axis=1)
    V_average_np = V_average.cpu().numpy() if V.is_cuda else V_average.numpy()
    ax = axes[num_vectors + 1]
    ax.imshow(V_average_np.reshape(image_shape), cmap="coolwarm", vmin=vmin, vmax=vmax)
    ax.set_title("Average PD")
    ax.axis("off")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    # Hide any unused subplots
    for i in range(total_plots, nrows * ncols):
        axes[i].axis("off")

    plt.suptitle(f"Number: {label}")
    plt.tight_layout()
    plt.show()


# Example usage
# Assuming `jacobian` is your computed Jacobian with shape [10, 8, 1, 28, 28]
# Compute SVD for the first sample in the batch


def main(args):
    # seed_everything(42)

    # Load the model (adjust path to your checkpoint)
    desired_label = args.label  # Example: looking for images of the digit '3'

    # Load the model (adjust path to your checkpoint)
    model = MNISTFlowModule.load_from_checkpoint(args.ckpt_path)

    # Prepare a data module and load a batch of data
    mnist_data_module = MNISTDataModule(batch_size=1, data_root="path_to_data")
    mnist_data_module.setup(stage="test")
    data_loader = mnist_data_module.test_dataloader()

    for batch in data_loader:
        inputs, labels = batch
        if labels == desired_label and np.random.randint(1, 11) == 1:
            break  # Break the loop once we find an image with the desired label

    # Compute the logit transformation
    inputs_logit, _ = logit_transform(inputs, reverse=False)

    # Compute the Jacobian matrix for the batch of inputs
    jacobian = compute_jacobian(inputs_logit, model)

    U, S, V = compute_svd(jacobian, sample_idx=0)

    print(f"Jacobian shape: {jacobian.shape}")
    print("Singular values:", S)
    k = 8  # Example threshold

    # The first k columns of V span the estimated tangent space at the sample
    tangent_space_basis = V[:, :k]

    # print(f"Estimated Tangent Space Basis for the first sample:\n{tangent_space_basis}")

    singular_values = S.numpy()  # Assuming S is a PyTorch tensor
    normalized_singular_values = singular_values / np.sum(singular_values)

    # Compute cumulative sum
    cumulative_sum = np.cumsum(normalized_singular_values)

    # Determine the dimensionality based on a percentage threshold
    percentage_threshold = 0.99  # or any other value that suits your analysis
    dimensionality = np.where(cumulative_sum >= percentage_threshold)[0][0] + 1

    print(f"Estimated dimensionality of the data manifold: {dimensionality}")
    visualize_principal_directions_with_mnist_image_and_average_in_grid(V, inputs, desired_label)


if __name__ == "__main__":
    parser = LightningArgumentParser()

    parser.add_lightning_class_args(MNISTFlowModule, "model")
    parser.add_lightning_class_args(MNISTDataModule, "data")

    # Set default data path based on environment
    default_data_path = (
        "/gcs/msindnn_staging/mnist_data" if "LOG_PATH" in os.environ else "../../../data/mnist"
    )
    parser.set_defaults({"data.data_root": default_data_path})

    default_data_path_k = (
        "/gcs/msindnn_staging/kmnist_data" if "LOG_PATH" in os.environ else "../../../data/kmnist"
    )

    # Include run-name and ckpt-path arguments
    timestamp = os.environ.get("CREATION_TIMESTAMP", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    parser.add_argument("--run-name", type=str, default=timestamp)
    parser.add_argument("--label", type=int, default=0)
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to the checkpoint file.")

    args = parser.parse_args()
    main(args)

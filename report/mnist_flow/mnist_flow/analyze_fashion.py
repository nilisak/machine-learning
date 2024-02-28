import torch

from mnist_flow.data import FashionMNISTDataModule
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


def compute_jacobian_base_wrt_latent(mnist_sample, model, epsilon=1e-7):
    """
    Computes the Jacobian of the base space pixels with respect to the latent space pixels.

    Args:
        mnist_sample (torch.Tensor): An MNIST sample for which to compute the Jacobian.
                                     Shape should be (1, 28, 28) for a single sample.
        model (torch.nn.Module): Model used for encoding and decoding.
        epsilon (float): A small value used for numerical approximation of gradients.

    Returns:
        torch.Tensor: The Jacobian matrix.
    """
    model.eval()

    if mnist_sample.dim() == 2:
        mnist_sample = mnist_sample.unsqueeze(0)  # Add batch dimension

    # Encode the input sample to latent space
    latent_repr, _ = model(mnist_sample, reverse=False)

    # Prepare for Jacobian computation
    num_latent_dims = latent_repr.numel()  # Total number of latent dimensions
    num_base_pixels = (
        mnist_sample.numel()
    )  # Assuming the reconstructed image has the same size as the input
    jacobian = torch.zeros(num_base_pixels, num_latent_dims)

    # Iterate over each dimension in the latent space
    for i in range(num_latent_dims):
        # Perturb the latent dimension
        latent_perturbed = latent_repr.clone()
        latent_perturbed.view(-1)[i] += epsilon

        # Reverse transform the perturbed latent representation
        reconstructed_perturbed, _ = model(latent_perturbed, reverse=True)

        # Compute the numerical gradient with respect to the perturbed dimension
        diff = (reconstructed_perturbed - mnist_sample) / epsilon
        jacobian[:, i] = diff.view(-1)

    return jacobian


def plot_S(S_numpy):
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(S_numpy, marker="o", linestyle="-", color="b")
    plt.title("Singular Values")
    plt.xlabel("Index")
    plt.ylabel("Singular Value")
    plt.grid(True)
    plt.show()


def variance_threshold(S, threshold):
    """Compute the number of significant singular values based on variance threshold and plot normalized cumsums."""
    # Calculate the proportion of variance explained by each singular value
    proportions = (S**2) / np.sum(S**2)
    # Calculate the cumulative sum of these proportions
    cumulative_proportions = np.cumsum(proportions, axis=0)

    # Find the number of singular values needed to meet the variance threshold
    num_singular_values = np.sum(cumulative_proportions < threshold).item() + 1

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(cumulative_proportions, marker="o", linestyle="-", color="b")
    plt.axhline(y=threshold, color="r", linestyle="--")
    plt.title("Cumulative Proportions of Explained Variance")
    plt.xlabel("Index of Singular Value")
    plt.ylabel("Cumulative Proportion of Variance Explained")
    plt.grid(True)
    plt.show()

    return num_singular_values


def visualize_principal_directions_with_mnist_image_and_average_in_grid(
    V, original_image, label, image_shape=(28, 28), desired_num_vectors=8
):
    """
    Visualize the original MNIST image, the first few principal directions from the matrix V with colorbars,
    and the average of all principal directions, arranging 4 subplots per row. Color mapping is made more aggressive.

    Parameters:
    - V: The matrix containing right singular vectors (n x n), expected to be a PyTorch tensor.
    - original_image: The original MNIST image to be visualized.
    - label: The label for the image being visualized.
    - image_shape: The shape of the images (height, width).
    - desired_num_vectors: The number of principal directions to visualize (adjusted based on V's size).
    """
    # Adjust num_vectors based on V's size to avoid IndexError
    num_vectors = min(desired_num_vectors, V.shape[1])

    # Calculate aggressive color mapping range based on the standard deviation of V's elements
    std_factor = 1  # Adjust this factor to make color mapping more or less aggressive
    mean_val = V.mean().item()
    std_val = V.std().item()
    vmin, vmax = mean_val - (std_factor * std_val), mean_val + (std_factor * std_val)

    # Adjust the layout to have each subplot in its own row
    total_plots = num_vectors + 2  # Including the original image and the average
    nrows = (total_plots + 3) // 4
    ncols = 4 if total_plots > 1 else 1  # Use only 1 column if showing 2 plots

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    if total_plots > 1:
        axes = axes.flatten()

    # Display the original MNIST image
    ax = axes[0] if total_plots > 1 else plt.gca()
    original_image_np = (
        original_image.cpu().numpy().reshape(image_shape)
        if original_image.requires_grad
        else original_image.reshape(image_shape)
    )
    ax.imshow(original_image_np, cmap="gray")
    ax.set_title("Original Image")
    ax.axis("off")

    for i in range(num_vectors):
        ax = axes[i + 1]
        principal_direction = V[:, i].reshape(image_shape)
        im = ax.imshow(principal_direction, cmap="coolwarm", vmin=vmin, vmax=vmax)
        ax.set_title(f"PD {i+1}")
        ax.axis("off")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    if total_plots > 1:
        # Display the average of all columns in V in the last subplot
        V_average = V.mean(axis=1)
        V_average_np = V_average.reshape(image_shape)
        ax = axes[num_vectors + 1]
        ax.imshow(V_average_np, cmap="coolwarm", vmin=vmin, vmax=vmax)
        ax.set_title("Average PD")
        ax.axis("off")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    # Hide any unused subplots
    if total_plots > 1:
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
    mnist_data_module = FashionMNISTDataModule(batch_size=1, data_root="path_to_data")
    mnist_data_module.setup(stage="test")
    data_loader = mnist_data_module.test_dataloader()

    for batch in data_loader:
        inputs, labels = batch
        if labels == desired_label and np.random.randint(1, 11) == 1:
            break  # Break the loop once we find an image with the desired label

    # Compute the logit transformation
    inputs_logit, _ = logit_transform(inputs, reverse=False)

    # Compute the Jacobian matrix for the batch of inputs
    jacobian = compute_jacobian_base_wrt_latent(inputs_logit, model)

    U, S, V = torch.svd(jacobian)

    S = S.detach().numpy()
    V = V.detach().numpy()

    print(f"Jacobian shape: {jacobian.shape}")

    threshold = 0.9  # Example threshold
    num_significant_singular_values = variance_threshold(S, threshold)
    print(f"Number of significant singular values: {num_significant_singular_values}")

    plot_S(S[:50])

    visualize_principal_directions_with_mnist_image_and_average_in_grid(
        V, inputs, desired_label, desired_num_vectors=8
    )


if __name__ == "__main__":
    parser = LightningArgumentParser()

    parser.add_lightning_class_args(MNISTFlowModule, "model")
    parser.add_lightning_class_args(FashionMNISTDataModule, "data")

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

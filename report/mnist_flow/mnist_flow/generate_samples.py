import torch

from mnist_flow.data import MNISTDataModule, KMNISTDataModule, FashionMNISTDataModule
from mnist_flow.model import MNISTFlowModule
from mpl_toolkits.axes_grid1 import make_axes_locatable
from lightning.pytorch.cli import LightningArgumentParser
from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

# Plotting the reconstructed images
import matplotlib.pyplot as plt
import numpy as np


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


def main(args):
    if args.kmnist:
        model = MNISTFlowModule.load_from_checkpoint("../kmnist_flow_weights_trained.ckpt")
        data_module = KMNISTDataModule(batch_size=25, data_root="data")
        data_name = "MNIST"
    elif args.fmnist:
        model = MNISTFlowModule.load_from_checkpoint("../fashion_flow_weights_trained.ckpt")
        data_module = FashionMNISTDataModule(batch_size=25, data_root="data")
        data_name = "KMNIST"
    else:
        model = MNISTFlowModule.load_from_checkpoint("../mnist_flow_weights_trained.ckpt")
        data_module = MNISTDataModule(batch_size=25, data_root="data")
        data_name = "FMNIST"

    if args.true_images:
        data_module.setup(stage="test")
        data_loader = data_module.test_dataloader()

        images = []

        # Iterate over the dataloader and collect images until we have 25
        for image_batch, label_batch in data_loader:
            for img, lbl in zip(image_batch, label_batch):
                images.append(img)
                if len(images) == 25:  # Check if we've collected 25 images
                    break
            if len(images) == 25:  # Exit the outer loop if we have enough images
                break

        # Plotting the images in a 5x5 grid
        fig, axes = plt.subplots(5, 5, figsize=(10, 10))
        for i, img in enumerate(images):
            row = i // 5  # Determine row index
            col = i % 5  # Determine column index
            img = img.squeeze().detach().cpu().numpy()  # Prepare image for plotting
            axes[row, col].imshow(img, cmap="gray")
            axes[row, col].axis("off")
        plt.suptitle(f"Real images of {data_name}")
        plt.tight_layout()
        plt.show()
    if args.generated_images:
        # You may need to set the model to evaluation mode
        model.eval()
        # Generate samples from the latent space
        n_samples = 25
        imgs_z = model.z_dist.sample(sample_shape=(n_samples, 1, 28, 28)).to(model.device)

        # Reconstruct images from the latent representations
        # Make sure to move the latent samples to the same device as the model
        reconstructed_images, _ = model(imgs_z, reverse=True)

        # Apply the inverse logit transform to get the images back to [0, 1] range
        reconstructed_images, _ = logit_transform(reconstructed_images, reverse=True)

        fig, axes = plt.subplots(5, 5, figsize=(10, 10))  # Adjust for 5x5 grid

        for i, img in enumerate(reconstructed_images):
            row = i // 5  # Determine row index
            col = i % 5  # Determine column index
            img = img.squeeze().detach().cpu().numpy()  # Prepare image for plotting
            axes[row, col].imshow(img, cmap="gray")
            axes[row, col].axis("off")
        plt.suptitle(f"generated images from normalized flow for {data_name}")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = LightningArgumentParser()

    # Include run-name and ckpt-path arguments
    timestamp = os.environ.get("CREATION_TIMESTAMP", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    parser.add_argument("--run-name", type=str, default=timestamp)
    parser.add_argument("--kmnist", action="store_true", default=False)
    parser.add_argument("--fmnist", action="store_true", default=False)
    parser.add_argument("--mnist", action="store_true", default=True)
    parser.add_argument("--generated_images", action="store_true", default=True)
    parser.add_argument("--true_images", action="store_true", default=True)
    args = parser.parse_args()
    main(args)

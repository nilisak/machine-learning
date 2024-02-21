# interpolate_mnist.py
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from pytorch_lightning import seed_everything
from argparse import ArgumentParser
from mnist_flow.data import MNISTDataModule
from mnist_flow.model import MNISTFlowModule
import wandb
from mnist_flow.utils import get_wandb_key, args_to_flat_dict
from lightning.pytorch.cli import LightningArgumentParser
from datetime import datetime
import os
import torch.nn.functional as F
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


def select_two_images_from_different_classes(data_loader):
    images, labels = next(iter(data_loader))
    # Index of the first image of two different classes
    class_0_index = (labels == 0).nonzero(as_tuple=True)[0][0].item()
    class_1_index = (labels == 1).nonzero(as_tuple=True)[0][0].item()

    # Extract the images
    image_0 = images[class_0_index]
    image_1 = images[class_1_index]

    return image_0, image_1


def interpolate_pixel_space(image1, image2, steps=10):
    image1 = image1.unsqueeze(0)  # Add batch dimension
    image2 = image2.unsqueeze(0)  # Add batch dimension
    weights = torch.linspace(0, 1, steps=steps).view(steps, 1, 1, 1)
    interpolated_images = (1 - weights) * image1 + weights * image2
    return interpolated_images


def interpolate_base_space(model, image1, image2, steps=10):
    # Combine images into a batch for forward processing
    images = torch.stack([image1, image2], dim=0)  # Shape: [2, C, H, W]
    images_latent, _ = model(images, reverse=False)  # Encode both images to latent space

    # Generate interpolation weights and interpolate in latent space
    weights = torch.linspace(0, 1, steps=steps, device=image1.device).view(steps, 1, 1, 1)
    interpolated_latents = (1 - weights) * images_latent[0:1] + weights * images_latent[
        1:2
    ]  # Shape: [steps, C, H, W]

    # Decode interpolated latents back to images
    print(interpolated_latents.shape)
    interpolated_images = [model(latent, reverse=True)[0] for latent in interpolated_latents]
    interpolated_images = torch.cat(
        interpolated_images, dim=0
    )  # Concatenate list of tensors into a single tensor

    return interpolated_images


def main(args):
    seed_everything(0xDEADBEEF)
    dm = MNISTDataModule(**vars(args.data))
    dm.setup(stage="test")
    model = MNISTFlowModule.load_from_checkpoint(args.ckpt_path)
    model.eval()
    model.freeze()

    val_dataloader = dm.val_dataloader()
    image_0, image_1 = select_two_images_from_different_classes(val_dataloader)

    interpolated_pixel = interpolate_pixel_space(image_0, image_1, steps=10)
    interpolated_base = interpolate_base_space(model, image_0, image_1, steps=10)

    wandb.login(key=get_wandb_key())
    wandb.init(project="mnist_interpolation")
    wandb.log(
        {
            "Pixel Space Interpolation": [wandb.Image(img) for img in interpolated_pixel],
            "Base Space Interpolation": [wandb.Image(img) for img in interpolated_base],
        }
    )


if __name__ == "__main__":
    parser = LightningArgumentParser()

    parser.add_lightning_class_args(MNISTFlowModule, "model")
    parser.add_lightning_class_args(MNISTDataModule, "data")

    # Set default data path based on environment
    default_data_path = (
        "/gcs/msindnn_staging/mnist_data" if "LOG_PATH" in os.environ else "../../../data/mnist"
    )
    parser.set_defaults({"data.data_root": default_data_path})

    # Include run-name and ckpt-path arguments
    timestamp = os.environ.get("CREATION_TIMESTAMP", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    parser.add_argument("--run-name", type=str, default=timestamp)
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to the checkpoint file.")

    args = parser.parse_args()
    main(args)

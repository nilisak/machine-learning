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


data_module = MNISTDataModule(batch_size=1, data_root="data")

model = MNISTFlowModule.load_from_checkpoint("../mnist_flow_adj_weights_trained.ckpt")

# You may need to set the model to evaluation mode
model.eval()

# Generate samples from the latent space
n_samples = 10
imgs_z = model.z_dist.sample(sample_shape=(n_samples, 1, 28, 28)).to(model.device)

# Reconstruct images from the latent representations
# Make sure to move the latent samples to the same device as the model
reconstructed_images, _ = model(imgs_z, reverse=True)

# Apply the inverse logit transform to get the images back to [0, 1] range
reconstructed_images, _ = logit_transform(reconstructed_images, reverse=True)


fig, axes = plt.subplots(1, n_samples, figsize=(20, 2))
for i, img in enumerate(reconstructed_images):
    img = img.squeeze().detach().cpu().numpy()  # Squeeze to remove channel dim, and move to cpu
    axes[i].imshow(img, cmap="gray")
    axes[i].axis("off")
plt.show()

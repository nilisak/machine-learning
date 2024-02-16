import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import wandb
import json
import os
import sys
from pytorch_lightning import Trainer


def get_wandb_key():
    json_file = "../wandb_key.json"
    if os.path.isfile(json_file):
        with open(json_file, "r") as f:
            return json.load(f)
    elif "WANDB_KEY" in os.environ:
        return os.environ["WANDB_KEY"]


wandb.login(key=get_wandb_key())
wandb.init(project="project_test", name="project")


# Define the Generator
class Generator(nn.Module):
    def __init__(self, z_dim=100, img_channels=3, img_size=32):
        super().__init__()
        self.img_channels = img_channels
        self.img_size = img_size
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, img_channels * img_size * img_size),
            nn.Tanh(),
        )

    def forward(self, z):
        img = self.gen(z)
        img = img.view(img.size(0), self.img_channels, self.img_size, self.img_size)
        return img


# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self, img_channels=3, img_size=32):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_channels * img_size * img_size, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.disc(img_flat)
        return validity


# Define the GAN model using PyTorch Lightning
class GAN(pl.LightningModule):
    def __init__(self, channels=3, width=32, height=32, z_dim=100, num_samples=16):
        super().__init__()
        self.save_hyperparameters()
        self.generator = Generator(z_dim=z_dim, img_channels=channels, img_size=width)
        self.discriminator = Discriminator(img_channels=channels, img_size=height)
        self.automatic_optimization = False  # Disable automatic optimization
        self.fixed_noise = torch.randn(num_samples, z_dim)

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx):
        imgs, _ = batch
        z = torch.randn(imgs.shape[0], self.hparams.z_dim, device=self.device)

        # Train Generator
        generated_imgs = self(z)
        g_loss = self.adversarial_loss(
            self.discriminator(generated_imgs), torch.ones(imgs.size(0), 1, device=self.device)
        )
        self.manual_backward(g_loss)
        self.optimizers()[0].step()
        self.optimizers()[0].zero_grad()

        # Train Discriminator
        real_loss = self.adversarial_loss(
            self.discriminator(imgs), torch.ones(imgs.size(0), 1, device=self.device)
        )
        fake_loss = self.adversarial_loss(
            self.discriminator(generated_imgs.detach()),
            torch.zeros(imgs.size(0), 1, device=self.device),
        )
        d_loss = (real_loss + fake_loss) / 2
        self.manual_backward(d_loss)
        self.optimizers()[1].step()
        self.optimizers()[1].zero_grad()

        self.log("g_loss", g_loss, on_epoch=True, prog_bar=True)
        self.log("d_loss", d_loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        lr = 0.0002
        b1 = 0.5
        b2 = 0.999
        opt_gen = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))

        scheduler_gen = torch.optim.lr_scheduler.StepLR(opt_gen, step_size=1, gamma=0.995)
        scheduler_disc = torch.optim.lr_scheduler.StepLR(opt_disc, step_size=1, gamma=0.995)
        return [opt_gen, opt_disc], [scheduler_gen, scheduler_disc]

    def compute_jacobian(self, z):
        # Assumes z is a noise vector for a single image, i.e., z.shape == (1, z_dim)
        self.generator.zero_grad()
        generated_image = self.generator(z)  # Generate a single image
        batch_size, img_channels, img_size, _ = generated_image.shape
        num_output_pixels = img_channels * img_size * img_size

        jacobian = torch.zeros(num_output_pixels, z.size(1), device=self.device)
        for j in range(num_output_pixels):
            # Zero gradients
            if z.grad is not None:
                z.grad.data.zero_()
            generated_image.view(-1)[j].backward(retain_graph=True, inputs=z)
            jacobian[j, :] = z.grad.data

        return jacobian

    def on_epoch_end(self):
        # Generate images at the end of each epoch using the fixed noise vector
        self.generator.eval()  # Set the generator to eval mode
        with torch.no_grad():
            generated_imgs = self.generator(self.fixed_noise).to("cpu")

        # Log generated images to wandb with the current epoch number
        images_to_log = [
            wandb.Image(
                ((img.numpy().transpose(1, 2, 0) + 1) / 2), caption=f"Epoch {self.current_epoch}"
            )
            for img in generated_imgs
        ]
        wandb.log({"generated_images": images_to_log}, commit=True)

        self.generator.train()  # Set the generator back to train


# Load CIFAR-10 Dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
)
train_dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
# train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Train the GAN
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

if "LOG_PATH" in os.environ:
    os.makedirs(os.path.dirname(os.environ["LOG_PATH"]), exist_ok=True)
    log = open(os.environ["LOG_PATH"], "a")
    sys.stdout = log
    sys.stderr = log

model = GAN()
trainer = Trainer(max_epochs=50)
trainer.fit(model, train_loader)

z = torch.randn(16, 100).type_as(
    next(model.parameters())
)  # Assuming the model and data are on the same device
model.eval()
with torch.no_grad():
    generated_imgs = model(z)

# Convert generated images for logging
images_to_log = [
    wandb.Image(((img.cpu().numpy().transpose(1, 2, 0) + 1) / 2)) for img in generated_imgs
]
wandb.log({"generated_images": images_to_log})

z_single = torch.randn(1, 100, device=device, requires_grad=True)

# Compute the Jacobian for the single generated image
jacobian_single = model.compute_jacobian(z_single)
print(jacobian_single)

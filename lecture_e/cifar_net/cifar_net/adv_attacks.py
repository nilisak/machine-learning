import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse
import wandb
from cifar_net.data import CIFARDataModule
from cifar_net.model2 import CIFARVGGModule  # Adjust according to your package structure
from cifar_net.utils import get_wandb_key  # Adjust according to your package structure
import os
import datetime


def adversarial_attack(
    model, source_image, target_class, device, lr=1e-3, max_iter=100, prob_threshold=0.99
):
    # Ensure source_image has a batch dimension and is on the correct device
    source_image = (
        source_image.to(device).unsqueeze(0) if source_image.dim() == 3 else source_image.to(device)
    )
    target_class_tensor = torch.tensor([target_class], device=device)

    # Clone source_image for optimization and ensure it requires gradient
    source_image_opt = source_image.clone().requires_grad_(True)

    optimizer = optim.Adam([source_image_opt], lr=lr)
    for _ in range(max_iter):
        optimizer.zero_grad()
        output = model(source_image_opt)
        loss = F.cross_entropy(output, target_class_tensor)
        loss.backward()
        optimizer.step()

        probs = F.softmax(output, dim=1)
        if probs[0, target_class] >= prob_threshold:
            break

    difference = torch.abs(source_image_opt - source_image)
    return source_image_opt.detach(), difference.detach(), probs[0, target_class].item()


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CIFARVGGModule.load_from_checkpoint(args.ckpt_path).to(device).eval()

    data_module = CIFARDataModule(data_root=args.data_root)
    data_module.setup(stage="test")
    val_dataloader = data_module.val_dataloader()

    # Initialize lists to store source images and their labels
    source_images = []
    source_labels = []

    # Iterate over the validation dataset to extract images from the specified source class
    for images, labels in val_dataloader:
        # Find indices of images that belong to the source class
        indices = torch.where(labels == args.source_class)[0]
        # Select images and labels for those indices
        selected_images = images[indices].to(device)
        selected_labels = labels[indices]

        # Append selected images and labels to the lists
        source_images.append(selected_images)
        source_labels.append(selected_labels)

        # Break the loop if we have collected enough samples
        if len(source_images) >= args.n_samples:
            break

    # Concatenate lists of tensors into single tensors
    source_images = torch.cat(source_images)[: args.n_samples]
    source_labels = torch.cat(source_labels)[: args.n_samples]

    wandb.login(key=get_wandb_key())
    wandb.init(project="adversarial-attacks", name=args.run_name)

    classes = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    table_data = []
    for idx, (source_image, source_label) in enumerate(zip(source_images, source_labels)):
        for target_class in range(10):  # CIFAR-10 has 10 classes
            if target_class == source_label.item():
                continue  # Skip if target class is the same as source class

            adversary_image, difference, final_prob = adversarial_attack(
                model,
                source_image,
                target_class,
                device,
                args.lr,
                args.max_iter,
                args.prob_threshold,
            )
            table_data.append(
                [
                    classes[args.source_class],
                    wandb.Image(source_image.squeeze()),
                    classes[target_class],
                    wandb.Image(adversary_image.squeeze()),
                    wandb.Image(difference.squeeze()),
                    final_prob,
                ]
            )

    columns = [
        "Source Class",
        "Source Image",
        "Target Class",
        "Adversary Image",
        "Difference Image",
        "Final Target Probability",
    ]
    wandb_table = wandb.Table(data=table_data, columns=columns)
    wandb.log({"adversarial_examples": wandb_table})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adversarial Attacks on CIFAR-10 Model")
    parser.add_argument("--data-root", type=str, default="../data")
    if "CREATION_TIMESTAMP" in os.environ:
        timestamp = os.environ["CREATION_TIMESTAMP"]
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parser.add_argument("--run-name", type=str, default=timestamp)
    if "LOG_PATH" in os.environ:
        path = "/gcs/isakbucket/custom-training-python-package/cifar_package/2024-02-11_133127/ms-in-dnns-cifar-net-lightning/j88owbkn/checkpoints/last.ckpt"
    else:
        path = "./ms-in-dnns-cifar-net-lightning/lelyjxwu/checkpoints/last.ckpt"
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=path,
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for optimizer")
    parser.add_argument("--max-iter", type=int, default=100, help="Max iteration for optimization")
    parser.add_argument(
        "--prob-threshold",
        type=float,
        default=0.5,
        help="Stop threshold for target class probability",
    )

    parser.add_argument(
        "--n-samples", type=int, default=5, help="Number of samples to start the attack from"
    )
    parser.add_argument(
        "--source-class", type=int, default=1, help="Index of the source image to attack"
    )
    args = parser.parse_args()
    main(args)

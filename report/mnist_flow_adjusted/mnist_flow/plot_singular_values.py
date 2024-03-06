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


def plot_singular_values_by_label(
    singular_values_matrix, num_values_to_plot=784, log_x=True, log_y=True, data_set=""
):
    num_labels = singular_values_matrix.shape[0]  # Number of labels

    plt.figure(figsize=(12, 6))
    all_labels_mean_singular_values = []

    # Loop through each label
    for label in range(num_labels):
        # Extract all singular values for the current label across all images
        singular_values_for_label = singular_values_matrix[label, :, :]

        # Calculate the mean across images for each singular value
        mean_singular_values = np.mean(singular_values_for_label, axis=1)

        # If we have too many singular values, plot only the first few
        if mean_singular_values.size > num_values_to_plot:
            mean_singular_values = mean_singular_values[:num_values_to_plot]

        # Plot the mean singular value of the current label as a line
        plt.plot(
            range(1, len(mean_singular_values) + 1), mean_singular_values, label=f"Label {label}"
        )
        all_labels_mean_singular_values.append(mean_singular_values)

    # Calculate the average mean singular values across all labels
    overall_mean = np.mean(all_labels_mean_singular_values, axis=0)

    # Plot the dashed line representing the average over all labels
    plt.plot(range(1, len(overall_mean) + 1), overall_mean, "k--", label="Average over all labels")

    plt.title(r"$\lambda$ for each label of {}".format(data_set), fontsize=16)
    plt.xlabel(r"$\lambda$ Index", fontsize=14)
    plt.ylabel(r"$\lambda$", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    if log_x:
        plt.xscale("log")
    if log_y:
        plt.yscale("log")  # Use log scale for y-axis
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.legend()  # Add legend to differentiate between labels
    plt.tight_layout()  # Adjust layout to not cut off labels
    plt.show()

    plt.figure(figsize=(12, 6))
    all_labels_mean_singular_values = []

    # Loop through each label
    for label in range(num_labels):
        # Extract all singular values for the current label across all images
        singular_values_for_label = singular_values_matrix[label, :, :]

        # Calculate the mean across images for each singular value
        mean_singular_values = np.mean(singular_values_for_label, axis=1)

        # If we have too many singular values, plot only the first few
        if mean_singular_values.size > num_values_to_plot:
            mean_singular_values = mean_singular_values[:num_values_to_plot]

        # Plot the mean singular value of the current label as a line
        plt.plot(
            range(1, len(mean_singular_values) + 1),
            mean_singular_values / max(mean_singular_values),
            label=f"Label {label}",
        )
        all_labels_mean_singular_values.append(mean_singular_values / max(mean_singular_values))

    # Calculate the average mean singular values across all labels
    overall_mean = np.mean(all_labels_mean_singular_values, axis=0)

    # Plot the dashed line representing the average over all labels
    plt.plot(range(1, len(overall_mean) + 1), overall_mean, "k--", label="Average over all labels")

    plt.title(
        r"$\frac{{\lambda}}{{\lambda_{{max}}}}$ for each label of {}".format(data_set), fontsize=16
    )
    plt.xlabel(r"$\lambda$ Index", fontsize=14)
    plt.ylabel(r"$\frac{\lambda}{\lambda_{max}}$", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    if log_x:
        plt.xscale("log")
    if log_y:
        plt.yscale("log")  # Use log scale for y-axis
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.legend()  # Add legend to differentiate between labels
    plt.tight_layout()  # Adjust layout to not cut off labels
    plt.show()


def plot_mean(l1, log_x=True, log_y=True):

    # Plot each array
    plt.figure(figsize=(12, 6))  # Set the figure size for better visibility
    plt.plot(range(1, len(l1) + 1), l1, "k--", label="MNIST")

    # Add some plot configurations
    plt.title(r"Average $\lambda$", fontsize=16)
    plt.xlabel(r"$\lambda$ Index", fontsize=14)
    plt.ylabel(r"$\lambda$", fontsize=14)
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.legend()  # Add legend to differentiate between labels
    plt.tight_layout()  # Adjust layout to not cut off labels

    if log_x:
        plt.xscale("log")
    if log_y:
        plt.yscale("log")  # Use log scale for y-axis

    # Show the plot
    plt.show()

    plt.figure(figsize=(12, 6))  # Set the figure size for better visibility
    plt.plot(range(1, len(l1) + 1), l1 / max(l1), "k--", label="MNIST")

    # Add some plot configurations
    plt.title(r"Average $\frac{\lambda}{\lambda_{max}}$", fontsize=16)
    plt.xlabel(r"$\lambda$ Index", fontsize=14)
    plt.ylabel(r"$\frac{\lambda}{\lambda_{max}}$", fontsize=14)
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.legend()  # Add legend to differentiate between labels
    plt.tight_layout()  # Adjust layout to not cut off labels

    if log_x:
        plt.xscale("log")
    if log_y:
        plt.yscale("log")  # Use log scale for y-axis

    # Show the plot
    plt.show()


def main():
    filenames = [
        "singular_values_done/mnist_arrays.npy",
        "singular_values_done/kmnist_arrays.npy",
        "singular_values_done/fmnist_arrays.npy",
    ]

    mnist_S = np.load(filenames[0])

    plot_singular_values_by_label(mnist_S, log_x=False, log_y=True, data_set="MNIST")

    average_M = np.mean(mnist_S, axis=(0, 2))

    plot_mean(average_M, log_x=True, log_y=True)


if __name__ == "__main__":
    main()

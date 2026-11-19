"""Joint MDS visualization of spectral bases and their class centroids."""

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
from sklearn.manifold import MDS


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.LLUM.utils import latest_checkpoint, load_config


def next_result_directory(root, experiment_name, run_name):
    """Create the next evaluation directory for the selected checkpoint run."""
    root.mkdir(parents=True, exist_ok=True)
    index = 1
    while (root / f"{experiment_name}-eval_{index}-{run_name}").exists():
        index += 1
    directory = root / f"{experiment_name}-eval_{index}-{run_name}"
    directory.mkdir()
    return directory


def save_plots(embedding, basis_labels, classes, directory):
    """Save the joint basis-and-centroid view and the centroid-only view."""
    palette_name = "tab10" if len(classes) <= 10 else "tab20"
    palette = plt.get_cmap(palette_name, len(classes))
    colors = [palette(index) for index in range(len(classes))]
    basis_coordinates = embedding[: len(basis_labels)]
    centroid_coordinates = embedding[len(basis_labels) :]

    figure, axis = plt.subplots(figsize=(11, 8))
    for class_index, class_name in enumerate(classes):
        selected = basis_labels == class_index
        axis.scatter(
            basis_coordinates[selected, 0],
            basis_coordinates[selected, 1],
            color=colors[class_index],
            s=58,
            alpha=0.8,
            label=class_name,
        )
        axis.scatter(
            *centroid_coordinates[class_index],
            color=colors[class_index],
            edgecolor="black",
            linewidth=0.9,
            marker="*",
            s=260,
            zorder=3,
        )
    centroid_key = Line2D(
        [],
        [],
        color="white",
        marker="*",
        markerfacecolor="gray",
        markeredgecolor="black",
        markersize=14,
        label="Class centroid",
    )
    handles, labels = axis.get_legend_handles_labels()
    axis.legend(handles + [centroid_key], labels + ["Class centroid"], ncol=2)
    axis.set_title("MDS of spectral basis vectors and class centroids")
    axis.set_xlabel("MDS dimension 1")
    axis.set_ylabel("MDS dimension 2")
    axis.grid(alpha=0.25, linestyle="--")
    figure.tight_layout()
    figure.savefig(directory / "bases_and_centroids_mds.png", dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 8))
    for class_index, class_name in enumerate(classes):
        axis.scatter(
            *centroid_coordinates[class_index],
            color=colors[class_index],
            edgecolor="black",
            linewidth=0.8,
            s=150,
            label=class_name,
            zorder=3,
        )
    axis.set_title("MDS of class centroids")
    axis.set_xlabel("MDS dimension 1")
    axis.set_ylabel("MDS dimension 2")
    axis.grid(alpha=0.25, linestyle="--")
    axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(directory / "centroids_mds.png", dpi=200)
    plt.close(figure)


def pairwise_mmd(bases):
    """Compute biased RBF-MMD distances between every pair of class basis sets."""
    vectors = bases.reshape(-1, bases.shape[-1])
    all_squared_distances = (
        (vectors[:, None] - vectors[None, :]) ** 2
    ).sum(axis=-1)
    unique_distances = all_squared_distances[
        np.triu_indices(len(vectors), k=1)
    ]
    positive_distances = unique_distances[unique_distances > 0]
    bandwidth_squared = (
        float(np.median(positive_distances)) if len(positive_distances) else 1.0
    )
    bandwidth_squared = max(bandwidth_squared, np.finfo(float).eps)

    def kernel(left, right):
        """Compute the squared Euclidean distances used by the RBF kernel."""
        squared_distances = ((left[:, None] - right[None, :]) ** 2).sum(axis=-1)
        return np.exp(-squared_distances / (2 * bandwidth_squared))

    num_classes = len(bases)
    within_kernels = [kernel(class_bases, class_bases).mean() for class_bases in bases]
    mmd_distances = np.zeros((num_classes, num_classes), dtype=np.float64)
    for left_index, left in enumerate(bases):
        for right_index in range(left_index + 1, num_classes):
            right = bases[right_index]
            mmd_squared = (
                within_kernels[left_index]
                + within_kernels[right_index]
                - 2 * kernel(left, right).mean()
            )
            mmd_distances[left_index, right_index] = mmd_distances[
                right_index, left_index
            ] = np.sqrt(max(float(mmd_squared), 0.0))
    return mmd_distances, bandwidth_squared


def save_mmd_results(distances, bandwidth_squared, classes, directory):
    """Save the pairwise MMD matrix as JSON, CSV, and a labeled heatmap."""
    report = {
        "kernel": "RBF",
        "estimator": "biased",
        "distance": "sqrt(MMD^2)",
        "bandwidth_squared": bandwidth_squared,
        "bandwidth_method": "median positive squared pairwise distance",
        "classes": classes,
        "pairwise_distances": distances.tolist(),
    }
    (directory / "bases_mmd.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    with (directory / "bases_mmd.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["class", *classes])
        writer.writerows(
            [class_name, *row] for class_name, row in zip(classes, distances)
        )

    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(distances, cmap="viridis")
    axis.set_xticks(np.arange(len(classes)), classes, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(classes)), classes)
    axis.set_title("Pairwise RBF-MMD between class basis sets")
    figure.colorbar(image, ax=axis, label="MMD distance")
    for row in range(len(classes)):
        for column in range(len(classes)):
            axis.text(
                column,
                row,
                f"{distances[row, column]:.3f}",
                ha="center",
                va="center",
                color="white",
            )
    figure.tight_layout()
    figure.savefig(directory / "bases_mmd.png", dpi=200)
    plt.close(figure)


def main():
    """Load the best checkpoint, save the MDS plots, and compare class bases."""
    config = load_config()
    experiment_name = config["experiment"]["name"]
    checkpoint_prefix = f"{experiment_name}_ddpm"
    checkpoint_path = latest_checkpoint(
        config["paths"]["ddpm_checkpoints"], checkpoint_prefix, "best"
    )
    if checkpoint_path is None:
        raise FileNotFoundError("No best DDPM checkpoint found")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    classes = checkpoint["classes"]
    bases = checkpoint["unmixing_model"]["latent_bases"].detach().float().cpu().numpy()
    band_mean = np.asarray(checkpoint["band_stats"]["mean"])
    band_std = np.asarray(checkpoint["band_stats"]["standard_deviation"])
    bases = bases * band_std[None, None, :] + band_mean[None, None, :]
    centroids = bases.mean(axis=1)

    basis_vectors = bases.reshape(-1, bases.shape[-1])
    basis_labels = np.repeat(np.arange(len(classes)), bases.shape[1])
    all_vectors = np.concatenate((basis_vectors, centroids), axis=0)
    # MDS has no out-of-sample transform, so fit vectors and centroids together.
    embedding = MDS(
        n_components=2,
        random_state=config["experiment"]["seed"],
        n_init=4,
        max_iter=500,
        dissimilarity="euclidean",
    ).fit_transform(all_vectors)

    run_name = checkpoint_path.stem.removeprefix(
        f"{checkpoint_prefix}-"
    ).removesuffix("-best")
    result_directory = next_result_directory(
        Path(config["paths"]["results"]), experiment_name, run_name
    )
    plots_directory = result_directory / "plots"
    plots_directory.mkdir()
    save_plots(
        embedding,
        basis_labels,
        classes,
        plots_directory,
    )
    mmd_distances, bandwidth_squared = pairwise_mmd(bases)
    save_mmd_results(mmd_distances, bandwidth_squared, classes, plots_directory)
    print(f"MDS plots saved to {plots_directory}")


if __name__ == "__main__":
    main()

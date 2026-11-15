"""Evaluate class-conditioned generation with bases from other classes."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import EuroSATMSDataset
from experiments.latent_unmixing_per_class.evaluation import (
    rmc_euclidean_distance,
    rmc_sam_degrees,
    spectra,
)
from experiments.latent_unmixing_per_class.model import (
    ClassSpecificLatentUnmixingModel,
    create_ddpm_model,
    create_sampling_scheduler,
    generate_ILR,
)
from experiments.latent_unmixing_per_class.utils import (
    from_ILR,
    latest_checkpoint,
    load_config,
    save_generated_pair,
    set_seed,
)


def parse_args():
    """Read optional run settings while taking defaults from the config."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    parser.add_argument(
        "--images-per-pair", "--images-per-class", dest="images_per_pair", type=int
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--inference-steps", type=int)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def next_result_directory(root, run_name, experiment_name):
    """Create a uniquely numbered evaluation directory for this checkpoint."""
    root.mkdir(parents=True, exist_ok=True)
    index = 1
    while (root / f"{experiment_name}-eval_{index}-{run_name}").exists():
        index += 1
    result = root / f"{experiment_name}-eval_{index}-{run_name}"
    result.mkdir()
    return result


@torch.inference_mode()
def generate_samples(
    ddpm, unmixing, checkpoint, config, settings, root, device, generator
):
    """Generate each target class with a fixed bank from every other class."""
    scheduler = create_sampling_scheduler(config)
    base_counts = {name: Counter() for name in checkpoint["classes"]}
    progress = tqdm(
        total=settings["images_per_pair"]
        * len(checkpoint["classes"])
        * (len(checkpoint["classes"]) - 1),
        desc="Ablation target/base",
        dynamic_ncols=True,
    )

    for target_index, class_name in enumerate(checkpoint["classes"]):
        ms_directory = root / class_name / "MS"
        rgb_directory = root / class_name / "RGB"
        ms_directory.mkdir(parents=True)
        rgb_directory.mkdir()
        for base_index, base_name in enumerate(checkpoint["classes"]):
            if base_index == target_index:
                continue
            generated = 0
            while generated < settings["images_per_pair"]:
                batch_size = min(
                    settings["batch_size"],
                    settings["images_per_pair"] - generated,
                )
                target_labels = torch.full(
                    (batch_size,), target_index, dtype=torch.long, device=device
                )
                base_labels = torch.full(
                    (batch_size,), base_index, dtype=torch.long, device=device
                )
                # Each sample gets its own Gaussian starting latent.
                latents = generate_ILR(
                    ddpm,
                    scheduler,
                    target_labels,
                    config,
                    settings["inference_steps"],
                    generator,
                )
                abundances = from_ILR(latents, target_labels, checkpoint["ilr_stats"])
                images = unmixing.decode(abundances, base_labels)

                for image in images:
                    generated += 1
                    base_counts[class_name][base_name] += 1
                    stem = f"{class_name}_from_{base_name}_{generated:04d}"
                    save_generated_pair(
                        image,
                        ms_directory / f"{stem}.tif",
                        rgb_directory / f"{stem}.png",
                        checkpoint["band_stats"],
                        config,
                    )
                progress.update(batch_size)

    progress.close()
    return base_counts


def real_test_paths(dataset, checkpoint, classes, images_per_pair):
    """Load test-set TIFF paths grouped by class."""
    paths_by_class = {name: [] for name in classes}
    for index in checkpoint["test_indices"]:
        path, label = dataset.samples[index]
        paths_by_class[dataset.classes[label]].append(path)
    for name, paths in paths_by_class.items():
        if len(paths) < images_per_pair:
            raise ValueError(
                f"Not enough real test images for {name}: found {len(paths)}"
            )
    return {name: paths[:images_per_pair] for name, paths in paths_by_class.items()}


def confusion_matrices(samples_directory, classes, real):
    """Compare every generated target/base pair with real samples of its target."""
    generated_paths = {
        (target, base): sorted(
            (samples_directory / target / "MS").glob(f"{target}_from_{base}_*.tif")
        )
        for target in classes
        for base in classes
        if target != base
    }
    generated_rmc = {
        pair: spectra(paths, 1.0) for pair, paths in generated_paths.items()
    }
    euclidean_distances = np.zeros((len(classes), len(classes)))
    sam_distances = np.zeros_like(euclidean_distances)
    for row, target in enumerate(classes):
        for column, base in enumerate(classes):
            if target == base:
                continue
            generated_curves = generated_rmc[(target, base)]
            euclidean_distances[row, column] = rmc_euclidean_distance(
                real[target], generated_curves
            )
            sam_distances[row, column] = rmc_sam_degrees(
                real[target], generated_curves
            )
    return euclidean_distances, sam_distances


def summarize_rmc_metrics(values):
    """Compute aggregate per-class RMC metrics without producing plots."""
    metrics = {}
    for class_name, samples in values.items():
        real, generated = samples["real"], samples["generated"]
        real_mean, generated_mean = real.mean(axis=0), generated.mean(axis=0)
        difference = generated_mean - real_mean
        metrics[class_name] = {
            "rmc_rmse": float(np.sqrt(np.mean(difference**2))),
            "rmc_mae": float(np.mean(np.abs(difference))),
            "rmc_euclidean_distance": rmc_euclidean_distance(real, generated),
            "rmc_sam_degrees": rmc_sam_degrees(real, generated),
        }
    return metrics


def save_distance_matrix(matrix, classes, path, title, colorbar_label):
    """Save a target-by-base heatmap with zero on the diagonal."""
    figure, axis = plt.subplots(
        figsize=(max(8, len(classes) * 0.8), max(7, len(classes) * 0.7))
    )
    image = axis.imshow(matrix, cmap="viridis")
    axis.set_xticks(np.arange(len(classes)), classes, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(classes)), classes)
    axis.set_xlabel("Base class")
    axis.set_ylabel("Target class")
    axis.set_title(title)
    for row in range(len(classes)):
        for column in range(len(classes)):
            if np.isnan(matrix[row, column]):
                continue
            axis.text(
                column,
                row,
                f"{matrix[row, column]:.3f}",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
            )
    figure.colorbar(image, ax=axis, label=colorbar_label)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    """Run the ablation, save generated pairs, and compare class RMC curves."""
    args = parse_args()
    config = load_config(args.config)
    settings = {
        "images_per_pair": (
            config["evaluation"]["images_per_class"]
            if args.images_per_pair is None
            else args.images_per_pair
        ),
        "batch_size": (
            config["evaluation"]["batch_size"]
            if args.batch_size is None
            else args.batch_size
        ),
        "inference_steps": (
            config["evaluation"]["inference_steps"]
            if args.inference_steps is None
            else args.inference_steps
        ),
        "seed": config["experiment"]["seed"] if args.seed is None else args.seed,
    }
    if min(
        settings["images_per_pair"], settings["batch_size"], settings["inference_steps"]
    ) <= 0:
        raise ValueError(
            "Images per pair, batch size, and inference steps must be positive"
        )
    set_seed(settings["seed"])

    prefix = "latent_unmixing_per_class_ddpm"
    checkpoint_path = latest_checkpoint(
        config["paths"]["ddpm_checkpoints"], prefix, "best"
    )
    if checkpoint_path is None:
        raise FileNotFoundError("No best checkpoint found for the class-specific DDPM")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if len(checkpoint["classes"]) < 2:
        raise ValueError("The ablation requires at least two classes")
    dataset = EuroSATMSDataset(config["data"]["root"])
    classes = checkpoint["classes"]
    real_paths = real_test_paths(
        dataset,
        checkpoint,
        classes,
        settings["images_per_pair"],
    )
    real = {
        name: spectra(paths, config["data"]["normalization_max"])
        for name, paths in real_paths.items()
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unmixing = ClassSpecificLatentUnmixingModel(
        config, checkpoint["unmixing_model"]["latent_bases"]
    ).to(device)
    unmixing.load_state_dict(checkpoint["unmixing_model"])
    unmixing.eval()
    ddpm = create_ddpm_model(config).to(device)
    ddpm.load_state_dict(checkpoint["model"])
    ddpm.eval()

    run_name = checkpoint_path.stem.removeprefix(f"{prefix}-").removesuffix("-best")
    result_directory = next_result_directory(
        Path(config["paths"]["results"]), run_name, config["experiment"]["name"]
    )
    samples_directory = result_directory / "samples"
    samples_directory.mkdir()
    generator = torch.Generator(device=device).manual_seed(settings["seed"])
    base_counts = generate_samples(
        ddpm,
        unmixing,
        checkpoint,
        config,
        settings,
        samples_directory,
        device,
        generator,
    )

    values = {
        name: {
            "real": real[name],
            "generated": spectra(
                sorted((samples_directory / name / "MS").glob("*.tif")), 1.0
            ),
        }
        for name in classes
    }
    metrics = summarize_rmc_metrics(values)
    euclidean_distances, sam_distances = confusion_matrices(
        samples_directory, classes, real
    )
    plots_directory = result_directory / "plots"
    plots_directory.mkdir(parents=True, exist_ok=True)
    save_distance_matrix(
        euclidean_distances,
        classes,
        plots_directory / "target_base_rmc_euclidean_distance.png",
        "Target/base distance — RMC vectors",
        "Mean Euclidean distance from target RMC",
    )
    save_distance_matrix(
        sam_distances,
        classes,
        plots_directory / "target_base_rmc_sam.png",
        "Target/base distance — mean RMC curves",
        "SAM to target class (degrees)",
    )
    summary = {
        "checkpoint": str(checkpoint_path),
        "settings": settings,
        "base_class_counts": {
            name: dict(counts) for name, counts in base_counts.items()
        },
        "per_class_metrics": metrics,
        "target_base_rmc_euclidean_matrix": {
            "target_classes": classes,
            "base_classes": classes,
            "metric": "mean Euclidean distance of generated RMC vectors from real target mean RMC",
            "values": [
                [None if np.isnan(value) else float(value) for value in row]
                for row in euclidean_distances
            ],
        },
        "target_base_rmc_sam_matrix": {
            "target_classes": classes,
            "base_classes": classes,
            "metric": "spectral angle mapper in degrees",
            "values": [
                [None if np.isnan(value) else float(value) for value in row]
                for row in sam_distances
            ],
        },
        "macro_metrics": {
            key: float(np.mean([entry[key] for entry in metrics.values()]))
            for key in next(iter(metrics.values()))
        },
    }
    with (result_directory / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(json.dumps(summary["macro_metrics"], indent=2))
    print(f"Ablation results saved to {result_directory}")


if __name__ == "__main__":
    main()

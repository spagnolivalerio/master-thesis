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
from experiments.LLUM.evaluation import (
    rmc_sam_degrees,
    spectra,
)
from experiments.LLUM.model import (
    ClassSpecificLatentUnmixingModel,
    create_ddpm_model,
    create_sampling_scheduler,
    generate_ILR,
)
from experiments.LLUM.utils import (
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
    """Generate each conditioning class with every class-specific basis bank."""
    scheduler = create_sampling_scheduler(config)
    base_counts = {name: Counter() for name in checkpoint["classes"]}
    progress = tqdm(
        total=settings["images_per_pair"]
        * len(checkpoint["classes"])
        * len(checkpoint["classes"]),
        desc="Ablation decoder/condition",
        dynamic_ncols=True,
    )

    for base_index, base_name in enumerate(checkpoint["classes"]):
        ms_directory = root / base_name / "MS"
        rgb_directory = root / base_name / "RGB"
        ms_directory.mkdir(parents=True)
        rgb_directory.mkdir()
        for condition_index, condition_name in enumerate(checkpoint["classes"]):
            generated = 0
            while generated < settings["images_per_pair"]:
                batch_size = min(
                    settings["batch_size"],
                    settings["images_per_pair"] - generated,
                )
                condition_labels = torch.full(
                    (batch_size,), condition_index, dtype=torch.long, device=device
                )
                base_labels = torch.full(
                    (batch_size,), base_index, dtype=torch.long, device=device
                )
                # Each sample gets its own Gaussian starting latent.
                latents = generate_ILR(
                    ddpm,
                    scheduler,
                    condition_labels,
                    config,
                    settings["inference_steps"],
                    generator,
                )
                abundances = from_ILR(
                    latents, condition_labels, checkpoint["ilr_stats"]
                )
                images = unmixing.decode(abundances, base_labels)

                for image in images:
                    generated += 1
                    base_counts[base_name][condition_name] += 1
                    stem = (
                        f"{base_name}_basis_conditioned_"
                        f"{condition_name}_{generated:04d}"
                    )
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
    """Compare every conditioning/decoder pair with its conditioning class RMC."""
    generated_paths = {
        (base, condition): sorted(
            (samples_directory / base / "MS").glob(
                f"{base}_basis_conditioned_{condition}_*.tif"
            )
        )
        for base in classes
        for condition in classes
    }
    generated_rmc = {
        pair: spectra(paths, 1.0) for pair, paths in generated_paths.items()
    }
    sid_distances = np.zeros((len(classes), len(classes)))
    sam_distances = np.zeros_like(sid_distances)
    rmse_distances = np.zeros_like(sid_distances)
    for row, condition in enumerate(classes):
        for column, base in enumerate(classes):
            generated_curves = generated_rmc[(base, condition)]
            real_mean, generated_mean = normalized_mean_rmcs(
                real[condition], generated_curves
            )
            sid_distances[row, column] = rmc_sid(real_mean, generated_mean)
            sam_distances[row, column] = rmc_sam_degrees(
                real[condition], generated_curves
            )
            raw_real_mean = real[condition].mean(axis=0)
            raw_generated_mean = generated_curves.mean(axis=0)
            rmse_distances[row, column] = np.sqrt(
                np.mean((raw_real_mean - raw_generated_mean) ** 2)
            )
    return sid_distances, sam_distances, rmse_distances


def normalized_mean_rmcs(real, generated):
    """Return the mean RMCs normalized to spectral distributions."""
    real_mean = np.maximum(real.mean(axis=0), 1e-12)
    generated_mean = np.maximum(generated.mean(axis=0), 1e-12)
    return real_mean / real_mean.sum(), generated_mean / generated_mean.sum()


def rmc_sid(real_distribution, generated_distribution):
    """Return symmetric SID between normalized mean real and generated RMCs."""
    return float(
        np.sum(real_distribution * np.log(real_distribution / generated_distribution))
        + np.sum(
            generated_distribution * np.log(generated_distribution / real_distribution)
        )
    )


def summarize_rmc_metrics(values):
    """Compute aggregate per-class RMC metrics without producing plots."""
    metrics = {}
    for class_name, samples in values.items():
        real, generated = samples["real"], samples["generated"]
        real_mean, generated_mean = normalized_mean_rmcs(real, generated)
        difference = generated_mean - real_mean
        metrics[class_name] = {
            "rmc_rmse": float(np.sqrt(np.mean(difference**2))),
            "rmc_mae": float(np.mean(np.abs(difference))),
            "rmc_sid": rmc_sid(real_mean, generated_mean),
            "rmc_sam_degrees": rmc_sam_degrees(real, generated),
        }
    return metrics


def save_distance_matrix(matrix, classes, path, title):
    """Save a conditioning-class-by-decoder-basis heatmap."""
    figure, axis = plt.subplots(
        figsize=(max(8, len(classes) * 0.8), max(7, len(classes) * 0.7))
    )
    image = axis.imshow(matrix, cmap="viridis")
    figure.colorbar(image, ax=axis)
    axis.set_xticks(np.arange(len(classes)), classes, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(classes)), classes)
    axis.set_xlabel("Decoder basis class")
    axis.set_ylabel("Abundance conditioning class")
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

    prefix = "LLUM_ddpm"
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
    sid_distances, sam_distances, rmse_distances = confusion_matrices(
        samples_directory, classes, real
    )
    plots_directory = result_directory / "plots"
    plots_directory.mkdir(parents=True, exist_ok=True)
    save_distance_matrix(
        sid_distances,
        classes,
        plots_directory / "condition_decoder_rmc_sid.png",
        "SID",
    )
    save_distance_matrix(
        sam_distances,
        classes,
        plots_directory / "condition_decoder_rmc_sam.png",
        "SAM",
    )
    save_distance_matrix(
        rmse_distances,
        classes,
        plots_directory / "condition_decoder_rmc_rmse.png",
        "RMSE",
    )
    summary = {
        "checkpoint": str(checkpoint_path),
        "settings": settings,
        "decoder_condition_counts": {
            name: dict(counts) for name, counts in base_counts.items()
        },
        "per_class_metrics": metrics,
        "condition_decoder_rmc_sid_matrix": {
            "abundance_conditioning_classes_j": classes,
            "decoder_basis_classes_i": classes,
            "metric": "Spectral Information Divergence between the normalized mean generated RMC conditioned on class j and decoded with basis i, and the normalized real mean RMC of class j",
            "values": [
                [None if np.isnan(value) else float(value) for value in row]
                for row in sid_distances
            ],
        },
        "condition_decoder_rmc_sam_matrix": {
            "abundance_conditioning_classes_j": classes,
            "decoder_basis_classes_i": classes,
            "metric": "spectral angle in degrees between the mean generated RMC conditioned on class j and decoded with basis i, and the real mean RMC of class j",
            "values": [
                [None if np.isnan(value) else float(value) for value in row]
                for row in sam_distances
            ],
        },
        "condition_decoder_rmc_rmse_matrix": {
            "abundance_conditioning_classes_j": classes,
            "decoder_basis_classes_i": classes,
            "metric": "RMSE in reflectance units between the mean generated RMC conditioned on class j and decoded with basis i, and the real mean RMC of class j",
            "values": [
                [None if np.isnan(value) else float(value) for value in row]
                for row in rmse_distances
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

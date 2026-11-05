"""Generazione ed evaluation RMC con decodifica class-specific."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import EuroSATMSDataset
from experiments.latent_unmixing_per_class_large.model import (
    ClassSpecificLatentUnmixingModel,
    create_ddpm_model,
    create_sampling_scheduler,
    generate_ILR,
)
from experiments.latent_unmixing_per_class_large.utils import (
    from_ILR,
    latest_checkpoint,
    load_config,
    save_generated_pair,
    set_seed,
)


def parse_args(config):
    parser = argparse.ArgumentParser(description="Evaluation per-class unmixing")
    parser.add_argument(
        "--images-per-class",
        type=int,
        default=config["evaluation"]["images_per_class"],
    )
    parser.add_argument(
        "--batch-size", type=int, default=config["evaluation"]["batch_size"]
    )
    parser.add_argument(
        "--inference-steps",
        type=int,
        default=config["evaluation"]["inference_steps"],
    )
    parser.add_argument("--path", type=Path, help="Directory samples/ esistente")
    return parser.parse_args()


def next_result_directory(root, run_name):
    root.mkdir(parents=True, exist_ok=True)
    index = 1
    while (root / f"latent_unmixing_per_class_large-eval_{index}-{run_name}").exists():
        index += 1
    path = root / f"latent_unmixing_per_class_large-eval_{index}-{run_name}"
    path.mkdir()
    return path


@torch.inference_mode()
def generate_samples(ddpm, unmixing, checkpoint, config, args, root, device):
    scheduler = create_sampling_scheduler(config)
    selected_classes = list(enumerate(checkpoint["classes"]))
    class_name = getattr(args, "class_name", None)
    if class_name is not None:
        selected_classes = [(checkpoint["classes"].index(class_name), class_name)]
    suffix = getattr(args, "suffix", "_eval")
    progress = tqdm(
        total=args.images_per_class * len(selected_classes),
        desc="Generazione per classe",
        dynamic_ncols=True,
    )
    for class_index, class_name in selected_classes:
        ms_directory = root / class_name / "MS"
        rgb_directory = root / class_name / "RGB"
        ms_directory.mkdir(parents=True)
        rgb_directory.mkdir()
        generator = torch.Generator(device=device).manual_seed(
            config["experiment"]["seed"] + 10_000 + class_index
        )
        generated = 0
        while generated < args.images_per_class:
            batch_size = min(args.batch_size, args.images_per_class - generated)
            labels = torch.full(
                (batch_size,), class_index, dtype=torch.long, device=device
            )
            latents = generate_ILR(
                ddpm,
                scheduler,
                labels,
                config,
                args.inference_steps,
                generator,
            )
            abundances = from_ILR(latents, labels, checkpoint["ilr_stats"])
            images = unmixing.decode(abundances, labels)
            for image in images:
                generated += 1
                stem = f"{class_name}_{generated:04d}{suffix}"
                save_generated_pair(
                    image,
                    ms_directory / f"{stem}.tif",
                    rgb_directory / f"{stem}.png",
                    checkpoint["band_stats"],
                    config,
                )
            progress.update(batch_size)
    progress.close()


def spectra(paths, scale):
    values = []
    for path in paths:
        image = tifffile.imread(path).astype(np.float64)
        values.append((image / scale).mean((0, 1)))
    if not values:
        raise ValueError("Nessuna immagine per il calcolo RMC")
    return np.stack(values)


def draw(axis, real, generated, class_name, config):
    wavelengths = np.asarray(config["data"]["wavelengths_nm"])
    order = np.argsort(wavelengths)
    wavelengths = wavelengths[order]
    real_mean, generated_mean = real.mean(0)[order], generated.mean(0)[order]
    real_std, generated_std = real.std(0)[order], generated.std(0)[order]
    axis.fill_between(
        wavelengths, real_mean - real_std, real_mean + real_std, color="black", alpha=0.1
    )
    axis.fill_between(
        wavelengths,
        generated_mean - generated_std,
        generated_mean + generated_std,
        color="#E15759",
        alpha=0.15,
    )
    axis.plot(wavelengths, real_mean, "o-", color="black", label="Real")
    axis.plot(wavelengths, generated_mean, "o-", color="#E15759", label="Generated")
    axis.set_title(class_name)
    axis.grid(alpha=0.25, linestyle="--")
    return {
        "rmc_rmse": float(np.sqrt(np.mean((real_mean - generated_mean) ** 2))),
        "rmc_mae": float(np.mean(np.abs(real_mean - generated_mean))),
    }


def save_plots(values, config, directory):
    directory.mkdir(parents=True, exist_ok=True)
    metrics = {}
    maximum = max(
        float(np.max(samples.mean(0) + samples.std(0)))
        for class_values in values.values()
        for samples in (class_values["real"], class_values["generated"])
    )
    y_limits = (0, maximum + max(maximum * 0.10, 0.01))
    figure, axes = plt.subplots(2, 5, figsize=(24, 9), sharex=True, sharey=True)
    axes.flat[0].set_ylim(*y_limits)
    for axis, (class_name, class_values) in zip(axes.flat, values.items()):
        metrics[class_name] = draw(
            axis,
            class_values["real"],
            class_values["generated"],
            class_name,
            config,
        )
    axes.flat[0].legend()
    figure.supxlabel("Wavelength (nm)")
    figure.supylabel("Reflectance")
    figure.suptitle("Real and generated RMCs with class-specific bases")
    figure.tight_layout()
    figure.savefig(directory / "rmc_all_classes.png", dpi=180)
    plt.close(figure)
    return metrics


def main():
    config = load_config()
    args = parse_args(config)
    set_seed(config["experiment"]["seed"])
    prefix = "latent_unmixing_per_class_large_ddpm"
    checkpoint_path = latest_checkpoint(
        config["paths"]["ddpm_checkpoints"], prefix, "best"
    )
    if checkpoint_path is None:
        raise FileNotFoundError("Nessun checkpoint best del DDPM per-class")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if args.path is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        unmixing = ClassSpecificLatentUnmixingModel(
            config, checkpoint["unmixing_model"]["latent_bases"]
        ).to(device)
        unmixing.load_state_dict(checkpoint["unmixing_model"])
        unmixing.eval()
        ddpm = create_ddpm_model(config).to(device)
        ddpm.load_state_dict(checkpoint["model"])
        ddpm.eval()
        result_directory = next_result_directory(
            Path(config["paths"]["results"]),
            checkpoint_path.stem.removeprefix(f"{prefix}-").removesuffix("-best"),
        )
        samples_directory = result_directory / "samples"
        samples_directory.mkdir()
        generate_samples(
            ddpm, unmixing, checkpoint, config, args, samples_directory, device
        )
    else:
        samples_directory = args.path.resolve()
        result_directory = samples_directory.parent

    dataset = EuroSATMSDataset(config["data"]["root"])
    real_paths = {name: [] for name in dataset.classes}
    for index in checkpoint["test_indices"]:
        path, label = dataset.samples[index]
        real_paths[dataset.classes[label]].append(path)
    values = {}
    for class_name in dataset.classes:
        generated = sorted((samples_directory / class_name / "MS").glob("*.tif"))
        if min(len(real_paths[class_name]), len(generated)) < args.images_per_class:
            raise ValueError(f"Campioni insufficienti per {class_name}")
        values[class_name] = {
            "real": spectra(
                real_paths[class_name][: args.images_per_class],
                config["data"]["normalization_max"],
            ),
            "generated": spectra(generated[: args.images_per_class], 1.0),
        }
    metrics = save_plots(values, config, result_directory / "plots")
    names = ("rmc_rmse", "rmc_mae")
    macro = {
        name: float(np.mean([entry[name] for entry in metrics.values()]))
        for name in names
    }
    print(json.dumps(macro, indent=2))
    return result_directory


if __name__ == "__main__":
    main()


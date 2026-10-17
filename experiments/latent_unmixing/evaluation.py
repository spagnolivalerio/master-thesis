"""Sampling ed evaluation spettrale del DDPM nello spazio ILR."""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import tifffile
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import EuroSATMSDataset  # noqa: E402
from experiments.latent_unmixing.model import (  # noqa: E402
    LatentUnmixingModel,
    create_ddpm_model,
    create_sampling_scheduler,
    generate_ILR,
)
from experiments.latent_unmixing.utils import (  # noqa: E402
    from_ILR,
    latest_checkpoint,
    load_config,
    save_generated_pair,
    set_seed,
)


def parse_args(config):
    parser = argparse.ArgumentParser(description="Evaluation latent unmixing")
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
    parser.add_argument(
        "--path", type=Path, help="Directory samples/ gia generata"
    )
    return parser.parse_args()


def next_evaluation_directory(root, run_name):
    root.mkdir(parents=True, exist_ok=True)
    index = 1
    while (root / f"latent_unmixing-eval_{index}-{run_name}").exists():
        index += 1
    path = root / f"latent_unmixing-eval_{index}-{run_name}"
    path.mkdir()
    return path


@torch.inference_mode()
def generate_samples(ddpm, encoder, checkpoint, config, args, directory, device):
    scheduler = create_sampling_scheduler(config)
    progress = tqdm(
        total=args.images_per_class * len(checkpoint["classes"]),
        desc="Generazione evaluation",
        dynamic_ncols=True,
    )
    for label, class_name in enumerate(checkpoint["classes"]):
        ms_directory = directory / class_name / "MS"
        rgb_directory = directory / class_name / "RGB"
        ms_directory.mkdir(parents=True)
        rgb_directory.mkdir()
        generator = torch.Generator(device=device).manual_seed(
            config["experiment"]["seed"] + 10_000 + label
        )
        generated = 0
        while generated < args.images_per_class:
            batch_size = min(args.batch_size, args.images_per_class - generated)
            labels = torch.full(
                (batch_size,), label, device=device, dtype=torch.long
            )
            standardized_ilr = generate_ILR(
                ddpm,
                scheduler,
                labels,
                config,
                args.inference_steps,
                generator,
            )
            abundances = from_ILR(standardized_ilr, checkpoint["ilr_stats"])
            images = encoder.decode(abundances)
            for image in images:
                generated += 1
                stem = f"{class_name}_{generated:04d}_eval"
                save_generated_pair(
                    image,
                    ms_directory / f"{stem}.tif",
                    rgb_directory / f"{stem}.png",
                    checkpoint["band_stats"],
                    config,
                )
            progress.update(batch_size)
    progress.close()


def read_spectra(paths, scale):
    spectra = []
    for path in paths:
        image = tifffile.imread(path).astype(np.float64)
        if image.ndim != 3:
            raise ValueError(f"TIFF non valido {path}: {image.shape}")
        spectra.append((image / scale).mean((0, 1)))
    if not spectra:
        raise ValueError("Nessuno spettro disponibile")
    return np.stack(spectra)


def draw_rmc(axis, real, generated, class_name, config):
    wavelengths = np.asarray(config["data"]["wavelengths_nm"])
    order = np.argsort(wavelengths)
    wavelengths = wavelengths[order]
    real_mean, generated_mean = real.mean(0)[order], generated.mean(0)[order]
    real_std, generated_std = real.std(0)[order], generated.std(0)[order]
    rmse = float(np.sqrt(np.mean((real_mean - generated_mean) ** 2)))
    mae = float(np.mean(np.abs(real_mean - generated_mean)))
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
    axis.plot(
        wavelengths, generated_mean, "o-", color="#E15759", label="Generated"
    )
    axis.set_title(class_name)
    axis.grid(alpha=0.25, linestyle="--")
    return {"rmc_rmse": rmse, "rmc_mae": mae}


def save_plots(spectra, config, directory):
    directory.mkdir(parents=True, exist_ok=True)
    metrics = {}
    maximum = max(
        float(np.max(samples.mean(0) + samples.std(0)))
        for values in spectra.values()
        for samples in (values["real"], values["generated"])
    )
    y_limits = (0, maximum + max(maximum * 0.10, 0.01))
    figure, axes = plt.subplots(2, 5, figsize=(24, 9), sharex=True, sharey=True)
    axes.flat[0].set_ylim(*y_limits)
    for axis, (class_name, values) in zip(axes.flat, spectra.items()):
        metrics[class_name] = draw_rmc(
            axis, values["real"], values["generated"], class_name, config
        )
        class_figure, class_axis = plt.subplots(figsize=(10, 6))
        draw_rmc(
            class_axis, values["real"], values["generated"], class_name, config
        )
        class_axis.set_ylim(*y_limits)
        class_axis.set_xlabel("Wavelength (nm)")
        class_axis.set_ylabel("Reflectance")
        class_axis.legend()
        class_figure.tight_layout()
        class_figure.savefig(directory / f"rmc_{class_name}.png", dpi=180)
        plt.close(class_figure)
    axes.flat[0].legend()
    figure.supxlabel("Wavelength (nm)")
    figure.supylabel("Reflectance")
    figure.suptitle("Real and generated RMCs by class")
    figure.tight_layout()
    figure.savefig(directory / "rmc_all_classes.png", dpi=180)
    plt.close(figure)
    return metrics


def main():
    config = load_config()
    args = parse_args(config)
    if args.images_per_class <= 0 or args.batch_size <= 0 or args.inference_steps <= 0:
        raise ValueError("I parametri numerici devono essere positivi")
    set_seed(config["experiment"]["seed"])
    checkpoint_path = latest_checkpoint(
        config["paths"]["ddpm_checkpoints"], "latent_unmixing_ddpm", "best"
    )
    if checkpoint_path is None:
        raise FileNotFoundError("Nessun checkpoint best del DDPM ILR")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if args.path is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = LatentUnmixingModel(
            config, checkpoint["encoder_model"]["latent_basis"]
        ).to(device)
        encoder.load_state_dict(checkpoint["encoder_model"])
        encoder.eval()
        ddpm = create_ddpm_model(config).to(device)
        ddpm.load_state_dict(checkpoint["model"])
        ddpm.eval()
        run_name = checkpoint_path.stem.removesuffix("-best")
        result_directory = next_evaluation_directory(
            Path(config["paths"]["results"]), run_name
        )
        samples_directory = result_directory / "samples"
        samples_directory.mkdir()
        generate_samples(
            ddpm, encoder, checkpoint, config, args, samples_directory, device
        )
    else:
        samples_directory = args.path.resolve()
        if not samples_directory.is_dir():
            raise FileNotFoundError(f"Directory non trovata: {samples_directory}")
        result_directory = samples_directory.parent

    dataset = EuroSATMSDataset(config["data"]["root"])
    if dataset.classes != checkpoint["classes"]:
        raise ValueError("Classi del dataset diverse dal checkpoint")
    real_paths = {name: [] for name in dataset.classes}
    for index in checkpoint["test_indices"]:
        path, label = dataset.samples[index]
        real_paths[dataset.classes[label]].append(path)

    spectra = {}
    for class_name in dataset.classes:
        generated_paths = sorted(
            (samples_directory / class_name / "MS").glob("*.tif")
        )
        if len(real_paths[class_name]) < args.images_per_class:
            raise ValueError(f"Test set insufficiente per {class_name}")
        if len(generated_paths) < args.images_per_class:
            raise ValueError(f"Campioni generati insufficienti per {class_name}")
        spectra[class_name] = {
            "real": read_spectra(
                real_paths[class_name][: args.images_per_class],
                config["data"]["normalization_max"],
            ),
            "generated": read_spectra(
                generated_paths[: args.images_per_class], 1.0
            ),
        }

    metrics = save_plots(spectra, config, result_directory / "plots")
    names = ("rmc_rmse", "rmc_mae")
    macro = {
        name: float(np.mean([values[name] for values in metrics.values()]))
        for name in names
    }
    summary = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": checkpoint["epoch"],
        "images_per_class": args.images_per_class,
        "inference_steps": args.inference_steps,
        "per_class": metrics,
        "macro_average": macro,
    }
    with (result_directory / "evaluation.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    with (result_directory / "evaluation.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["class", *names])
        for class_name, values in metrics.items():
            writer.writerow([class_name, *[values[name] for name in names]])
        writer.writerow(["macro_average", *[macro[name] for name in names]])
    print(json.dumps(macro, indent=2))
    print(f"Evaluation salvata in {result_directory}")
    return result_directory


if __name__ == "__main__":
    main()


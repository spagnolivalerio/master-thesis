"""Valutazione RMC della baseline, senza dipendenze da ``common``."""

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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import EuroSATMSDataset  
from experiments.baseline.sample import main as sample  
from experiments.baseline.utils import latest_checkpoint, load_config  




def parse_args():
    parser = argparse.ArgumentParser(description="Valuta i campioni della baseline")
    parser.add_argument(
        "--path",
        type=Path,
        help="Directory samples/ esistente; se omessa esegue prima il sampling",
    )
    return parser.parse_args()


def read_image_spectrum(path, scale):
    image = tifffile.imread(path).astype(np.float64)
    if image.ndim != 3:
        raise ValueError(f"TIFF non valido {path}: {image.shape}")
    return (image / scale).mean(axis=(0, 1))


def generated_paths(samples_directory, class_name):
    directory = samples_directory / class_name / "MS"
    return sorted((*directory.glob("*.tif"), *directory.glob("*.tiff")))


def real_paths_by_class(dataset, test_indices):
    paths = {name: [] for name in dataset.classes}
    for index in test_indices:
        path, label = dataset.samples[index]
        paths[dataset.classes[label]].append(path)
    return paths


def draw_comparison(axis, real, generated, class_name, config):
    order = np.argsort(config["data"]["wavelengths_nm"])
    wavelengths = np.asarray(config["data"]["wavelengths_nm"])[order]
    real_mean = real.mean(axis=0)[order]
    generated_mean = generated.mean(axis=0)[order]
    real_std = real.std(axis=0)[order]
    generated_std = generated.std(axis=0)[order]
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
    return float(np.sqrt(np.mean((real_mean - generated_mean) ** 2)))


def save_histogram_difference(real, generated, wavelengths, class_name, directory):
    order = np.argsort(wavelengths)
    real, generated = real[:, order], generated[:, order]
    wavelengths = np.asarray(wavelengths)[order]
    minimum = min(float(real.min()), float(generated.min()))
    maximum = max(float(real.max()), float(generated.max()))
    if minimum == maximum:
        maximum = minimum + 1e-6
    edges = np.linspace(minimum, maximum, 31)
    differences = []
    distances = []
    for band in range(real.shape[1]):
        real_hist = np.histogram(real[:, band], bins=edges)[0] / len(real)
        generated_hist = np.histogram(generated[:, band], bins=edges)[0] / len(generated)
        differences.append(generated_hist - real_hist)
        distances.append(0.5 * np.abs(generated_hist - real_hist).sum())
    differences = np.asarray(differences)
    limit = max(float(np.abs(differences).max()), 1e-6)
    figure, axis = plt.subplots(figsize=(12, 6))
    image = axis.imshow(
        differences,
        aspect="auto",
        extent=(edges[0], edges[-1], len(wavelengths) - 0.5, -0.5),
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
    )
    axis.set_yticks(np.arange(len(wavelengths)), [f"{value:g}" for value in wavelengths])
    axis.set_xlabel("Mean reflectance")
    axis.set_ylabel("Wavelength (nm)")
    axis.set_title(f"RMC histogram difference (fake − real) — {class_name}")
    figure.colorbar(image, ax=axis, label="Probability difference")
    figure.tight_layout()
    figure.savefig(directory / f"rmc_histogram_difference_{class_name}.png", dpi=180)
    plt.close(figure)
    return float(np.mean(distances))


def save_plots(spectra, config, plots_directory):
    plots_directory.mkdir(exist_ok=True)
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
        rmse = draw_comparison(
            axis, values["real"], values["generated"], class_name, config
        )
        metrics[class_name] = {"rmc_rmse": rmse}
        metrics[class_name]["rmc_histogram_tv"] = save_histogram_difference(
            values["real"],
            values["generated"],
            config["data"]["wavelengths_nm"],
            class_name,
            plots_directory,
        )
        class_figure, class_axis = plt.subplots(figsize=(10, 6))
        draw_comparison(
            class_axis, values["real"], values["generated"], class_name, config
        )
        class_axis.set_ylim(*y_limits)
        class_axis.set_xlabel("Wavelength (nm)")
        class_axis.set_ylabel("Reflectance")
        class_axis.legend()
        class_figure.tight_layout()
        class_figure.savefig(plots_directory / f"rmc_{class_name}.png", dpi=180)
        plt.close(class_figure)
    axes.flat[0].legend()
    figure.supxlabel("Wavelength (nm)")
    figure.supylabel("Reflectance")
    figure.suptitle("Real and generated RMCs by class")
    figure.tight_layout()
    figure.savefig(plots_directory / "rmc_all_classes.png", dpi=180)
    plt.close(figure)
    return metrics


def main():
    args = parse_args()
    config = load_config()
    checkpoint_path = latest_checkpoint(
        Path(config["paths"]["checkpoints"]), suffix="best"
    )
    if checkpoint_path is None:
        raise FileNotFoundError("Nessun checkpoint best della baseline")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if args.path is None:
        root = Path(config["paths"]["results"])
        root.mkdir(parents=True, exist_ok=True)
        run_name = checkpoint_path.stem.removeprefix("baseline-").removesuffix("-best")
        index = 1
        while (root / f"baseline-eval_{index}-{run_name}").exists():
            index += 1
        result_directory = root / f"baseline-eval_{index}-{run_name}"
        result_directory.mkdir()
        sample(result_directory=result_directory, suffix="_eval")
        samples_directory = result_directory / "samples"
    else:
        samples_directory = args.path.resolve()
        if not samples_directory.is_dir():
            raise FileNotFoundError(f"Directory non trovata: {samples_directory}")
        result_directory = samples_directory.parent

    dataset = EuroSATMSDataset(config["data"]["root"])
    if dataset.classes != checkpoint["classes"]:
        raise ValueError("Le classi del dataset non corrispondono al checkpoint")
    real_paths = real_paths_by_class(dataset, checkpoint["test_indices"])
    spectra = {}
    for class_name in dataset.classes:
        generated = generated_paths(samples_directory, class_name)
        if not generated:
            raise ValueError(f"Nessun campione generato per {class_name}")
        count = min(len(generated), len(real_paths[class_name]))
        spectra[class_name] = {
            "real": np.stack(
                [
                    read_image_spectrum(path, config["data"]["normalization_max"])
                    for path in real_paths[class_name][:count]
                ]
            ),
            "generated": np.stack(
                [read_image_spectrum(path, 1.0) for path in generated[:count]]
            ),
        }

    metrics = save_plots(spectra, config, result_directory / "plots")
    macro = {
        key: float(np.mean([values[key] for values in metrics.values()]))
        for key in ("rmc_rmse", "rmc_histogram_tv")
    }
    print(json.dumps(macro, indent=2))
    print(f"Valutazione salvata in {result_directory}")
    return result_directory


if __name__ == "__main__":
    main()


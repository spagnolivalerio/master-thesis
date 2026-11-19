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
from experiments.DDPM.sample import main as sample  
from experiments.DDPM.utils import latest_checkpoint, load_config  




def parse_args():
    """Parse the optional sample directory used for evaluation."""
    parser = argparse.ArgumentParser(description="Valuta i campioni della DDPM")
    parser.add_argument(
        "--path",
        type=Path,
        help="Directory samples/ esistente; se omessa esegue prima il sampling",
    )
    return parser.parse_args()


def read_image_spectrum(path, scale):
    """Load an image and average its reflectance across spatial pixels."""
    image = tifffile.imread(path).astype(np.float64)
    if image.ndim != 3:
        raise ValueError(f"TIFF non valido {path}: {image.shape}")
    return (image / scale).mean(axis=(0, 1))


def generated_paths(samples_directory, class_name):
    """List generated multispectral TIFF files for one class."""
    directory = samples_directory / class_name / "MS"
    return sorted((*directory.glob("*.tif"), *directory.glob("*.tiff")))


def real_paths_by_class(dataset, test_indices):
    """Group real image paths by class for the selected dataset indices."""
    paths = {name: [] for name in dataset.classes}
    for index in test_indices:
        path, label = dataset.samples[index]
        paths[dataset.classes[label]].append(path)
    return paths


def draw_comparison(axis, real, generated, class_name, config):
    """Plot real and generated mean RMCs and return their RMSE."""
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


def rmc_euclidean_distance(real, generated):
    """Compute Euclidean distance between the mean real and generated RMCs."""
    difference = generated.mean(axis=0) - real.mean(axis=0)
    return float(np.linalg.norm(difference))


def rmc_sam_degrees(real, generated):
    """Compute the spectral angle between mean RMCs in degrees."""
    real_mean, generated_mean = real.mean(axis=0), generated.mean(axis=0)
    cosine = np.dot(real_mean, generated_mean) / max(
        np.linalg.norm(real_mean) * np.linalg.norm(generated_mean), 1e-12
    )
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def save_plots(spectra, config, plots_directory):
    """Save class-level RMC plots and return their metrics."""
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
        metrics[class_name] = {
            "rmc_rmse": rmse,
            "rmc_euclidean_distance": rmc_euclidean_distance(
                values["real"], values["generated"]
            ),
            "rmc_sam_degrees": rmc_sam_degrees(
                values["real"], values["generated"]
            ),
        }
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
    """Run DDPM sampling if needed, evaluate RMCs, and save results."""
    args = parse_args()
    config = load_config()
    checkpoint_path = latest_checkpoint(
        Path(config["paths"]["checkpoints"]), suffix="best"
    )
    if checkpoint_path is None:
        raise FileNotFoundError("Nessun checkpoint best della DDPM")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if args.path is None:
        root = Path(config["paths"]["results"])
        root.mkdir(parents=True, exist_ok=True)
        run_name = checkpoint_path.stem.removeprefix("DDPM-").removesuffix("-best")
        index = 1
        while (root / f"DDPM-eval_{index}-{run_name}").exists():
            index += 1
        result_directory = root / f"DDPM-eval_{index}-{run_name}"
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
        for key in ("rmc_rmse", "rmc_euclidean_distance", "rmc_sam_degrees")
    }
    print(json.dumps(macro, indent=2))
    print(f"Valutazione salvata in {result_directory}")
    return result_directory


if __name__ == "__main__":
    main()


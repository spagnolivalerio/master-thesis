"""Confronta le RMC annotate reali e sintetiche, con media ±1 SD tra immagini."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.latent_unmixing_per_class.utils import load_config


def read_region_spectrum(image_path, mask_path, scale, mask_class):
    """Restituisce la firma media dei soli pixel della classe annotata."""
    with Image.open(mask_path) as mask:
        classes = json.loads(mask.info["Classes"])
        class_id = next(item["id"] for item in classes if item["name"] == mask_class)
        selected = np.asarray(mask) == class_id
    if not selected.any():
        raise ValueError(f"Nessun pixel {mask_class!r} in {mask_path}")
    image = tifffile.imread(image_path).astype(np.float64) / scale
    return image[selected].mean(axis=0)


def collect_spectra(image_directory, mask_directory, scale, mask_class):
    """Una firma per immagine: le regioni grandi non ricevono un peso maggiore."""
    spectra = []
    for image_path in sorted(image_directory.glob("*.tif")):
        mask_path = mask_directory / f"{image_path.stem}_annotation.png"
        spectra.append(read_region_spectrum(image_path, mask_path, scale, mask_class))
    if len(spectra) < 2:
        raise ValueError(f"Servono almeno due immagini per stimare la varianza: {image_directory}")
    return np.stack(spectra)


def plot_rmc(real, fake, config, directory):
    """Sovrappone le medie e le fasce ±1 SD delle firme delle singole immagini."""
    wavelengths = np.asarray(config["data"]["wavelengths_nm"])
    order = np.argsort(wavelengths)
    wavelengths = wavelengths[order]
    figure, axis = plt.subplots(figsize=(12, 7))
    maximum = 0.0
    for label, spectra, color in (("Real", real, "black"), ("Fake", fake, "#E15759")):
        mean = spectra.mean(axis=0)[order]
        # La SD è la radice della varianza campionaria; ha le unità della riflettanza.
        std = spectra.std(axis=0, ddof=1)[order]
        axis.plot(wavelengths, mean, "o-", color=color, label=f"{label} (n={len(spectra)})")
        axis.fill_between(wavelengths, mean - std, mean + std, color=color, alpha=0.15)
        maximum = max(maximum, float(np.max(mean + std)))
    axis.set_ylim(0, maximum + max(maximum * 0.10, 0.01))
    axis.set_xlabel("Wavelength (nm)")
    axis.set_ylabel("Reflectance")
    mask_class = config["annotation_evaluation"]["mask_class"]
    axis.set_title(f"Real and fake RMCs — annotated {mask_class}")
    axis.grid(alpha=0.25, linestyle="--")
    axis.legend()
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        figure.savefig(directory / f"annotated_rmc_comparison.{extension}", dpi=180)
    plt.close(figure)


def main():
    config = load_config()
    root = Path(config["paths"]["annotation_images"])
    mask_class = config["annotation_evaluation"]["mask_class"]
    # EuroSAT reale usa valori scalati; i TIFF sintetici sono già riflettanze.
    real = collect_spectra(
        root / "real", root / "real_mask",
        config["data"]["normalization_max"], mask_class,
    )
    fake = collect_spectra(root / "fake", root / "fake_mask", 1.0, mask_class)
    directory = Path(config["paths"]["annotation_plots"])
    plot_rmc(real, fake, config, directory)
    print(f"Real: {len(real)} immagini; fake: {len(fake)} immagini")
    print(f"Grafici salvati in {directory}")


if __name__ == "__main__":
    main()

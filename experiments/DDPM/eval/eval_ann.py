"""Confronta RMC reali e sintetiche su pixel annotati e non, con media ±1 SD."""

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

from experiments.DDPM.utils import load_config


def read_region_spectrum(image_path, mask_path, scale, mask_class, unannotated=False):
    """Return the mean spectrum of annotated pixels or unassigned pixels (ID 0)."""
    with Image.open(mask_path) as mask:
        if unannotated:
            class_id = 0
        else:
            classes = json.loads(mask.info["Classes"])
            class_id = next(item["id"] for item in classes if item["name"] == mask_class)
        selected = np.asarray(mask) == class_id
    if not selected.any():
        if unannotated:
            return None
        raise ValueError(f"Nessun pixel {mask_class!r} in {mask_path}")
    image = tifffile.imread(image_path).astype(np.float64) / scale
    return image[selected].mean(axis=0)


def collect_spectra(image_directory, mask_directory, scale, mask_class, unannotated=False):
    """Collect one spectrum per image so larger regions receive no extra weight."""
    spectra = []
    for image_path in sorted(image_directory.glob("*.tif")):
        mask_path = mask_directory / f"{image_path.stem}_annotation.png"
        spectrum = read_region_spectrum(image_path, mask_path, scale, mask_class, unannotated)
        if spectrum is not None:
            spectra.append(spectrum)
    if len(spectra) < 2:
        region = "non annotati" if unannotated else f"annotati {mask_class!r}"
        raise ValueError(
            f"Servono almeno due immagini con pixel {region} per stimare la varianza: {image_directory}"
        )
    return np.stack(spectra)


def rmc_euclidean_distance(real, fake):
    """Compute the Euclidean distance between the mean real and generated RMCs."""
    difference = fake.mean(axis=0) - real.mean(axis=0)
    return float(np.linalg.norm(difference))


def rmc_sam_degrees(real, fake):
    """Compute the spectral angle between mean RMCs in degrees."""
    real_mean, fake_mean = real.mean(axis=0), fake.mean(axis=0)
    cosine = np.dot(real_mean, fake_mean) / max(
        np.linalg.norm(real_mean) * np.linalg.norm(fake_mean), 1e-12
    )
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def plot_rmc(real, fake, config, directory, unannotated=False):
    """Plot the mean RMCs and ±1 SD ranges across images."""
    wavelengths = np.asarray(config["data"]["wavelengths_nm"])
    order = np.argsort(wavelengths)
    wavelengths = wavelengths[order]
    figure, axis = plt.subplots(figsize=(12, 7))
    for label, spectra, color in (("Real", real, "black"), ("Generated", fake, "#E15759")):
        mean = spectra.mean(axis=0)[order]
        # La SD è la radice della varianza campionaria; ha le unità della riflettanza.
        std = spectra.std(axis=0, ddof=1)[order]
        axis.plot(wavelengths, mean, "o-", color=color, label=f"{label} (n={len(spectra)})")
        axis.fill_between(wavelengths, mean - std, mean + std, color=color, alpha=0.15)
    axis.set_ylim(*config["annotation_evaluation"]["y_limits"])
    axis.set_xlabel("Wavelength (nm)")
    axis.set_ylabel("Reflectance")
    mask_class = config["annotation_evaluation"]["mask_class"]
    region = "unannotated pixels" if unannotated else f"annotated {mask_class}"
    axis.set_title(f"Real and fake RMCs — {region}")
    axis.grid(alpha=0.25, linestyle="--")
    axis.legend()
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    directory.mkdir(parents=True, exist_ok=True)
    prefix = "unannotated" if unannotated else "annotated"
    for extension in ("png", "pdf"):
        figure.savefig(directory / f"{prefix}_rmc_comparison.{extension}", dpi=180)
    plt.close(figure)

    return {
        "rmc_euclidean_distance": rmc_euclidean_distance(real, fake),
        "rmc_sam_degrees": rmc_sam_degrees(real, fake),
    }


def plot_rmc_std_parallel(annotated, unannotated, config, directory):
    """Save parallel coordinates of RMC standard deviations for both regions."""
    order = np.argsort(config["data"]["wavelengths_nm"])
    band_names = config["data"]["band_names"]
    labels = [band_names[index] for index in order]
    positions = np.arange(len(labels))
    groups = [
        ("Annotated pixels", annotated[0], annotated[1]),
        ("Unannotated pixels", unannotated[0], unannotated[1]),
    ]

    profiles = [
        (real.std(axis=0, ddof=1)[order], fake.std(axis=0, ddof=1)[order])
        for _, real, fake in groups
    ]
    maximum = max(float(values.max()) for pair in profiles for values in pair)
    y_max = max(maximum * 1.05, 1e-6)
    colors = {"Real": "#1F77B4", "Generated": "#FF7F0E"}
    figure, axes = plt.subplots(2, 1, figsize=(14, 9), sharey=True)

    for axis, (title, _, _), (real_std, fake_std) in zip(axes, groups, profiles):
        axis.vlines(positions, 0, y_max, color="#555555", linewidth=0.9, zorder=0)
        axis.plot(positions, real_std, color=colors["Real"], linewidth=2.3, alpha=1.0)
        axis.plot(positions, fake_std, color=colors["Generated"], linewidth=2.3, alpha=1.0)
        axis.set_xticks(positions, labels)
        axis.xaxis.set_ticks_position("top")
        axis.tick_params(
            axis="x", top=True, labeltop=True, bottom=False, labelbottom=False, pad=7
        )
        axis.set_xlim(-0.4, positions[-1] + 0.55)
        axis.set_ylim(0, y_max)
        axis.set_title(title, pad=28)
        axis.grid(False)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["bottom"].set_visible(False)

        real_offset, fake_offset = 0, 0
        if abs(real_std[-1] - fake_std[-1]) < y_max * 0.03:
            real_offset, fake_offset = 5, -5
        axis.annotate(
            "Real", (positions[-1], real_std[-1]), xytext=(7, real_offset),
            textcoords="offset points", color=colors["Real"], va="center",
        )
        axis.annotate(
            "Generated", (positions[-1], fake_std[-1]), xytext=(7, fake_offset),
            textcoords="offset points", color=colors["Generated"], va="center",
        )

    figure.text(
        0.025, 0.5, "Reflectance standard deviation", rotation=90, va="center"
    )
    figure.subplots_adjust(
        left=0.08, right=0.95, top=0.96, bottom=0.04, hspace=0.48
    )
    directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(directory / "rmc_std_parallel.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    """Compare real and generated spectra for annotated and unannotated pixels."""
    config = load_config()
    root = Path(config["paths"]["annotation_images"])
    mask_class = config["annotation_evaluation"]["mask_class"]
    directory = Path(config["paths"]["annotation_plots"])
    parallel_groups = {}
    for unannotated in (False, True):
        # EuroSAT reale usa valori scalati; i TIFF sintetici sono già riflettanze.
        real = collect_spectra(
            root / "real" / "tiff", root / "real" / "mask",
            config["data"]["normalization_max"], mask_class, unannotated,
        )
        fake = collect_spectra(
            root / "fake" / "tiff", root / "fake" / "mask", 1.0, mask_class, unannotated,
        )
        metrics = plot_rmc(real, fake, config, directory, unannotated)
        parallel_groups[unannotated] = (real, fake)
        region = "Non annotati" if unannotated else f"Annotati {mask_class}"
        print(
            f"{region} — real: {len(real)} immagini; generated: {len(fake)} immagini; "
            f"distanza euclidea RMC: {metrics['rmc_euclidean_distance']:.6f}; "
            f"SAM: {metrics['rmc_sam_degrees']:.6f}°"
        )
    plot_rmc_std_parallel(
        parallel_groups[False], parallel_groups[True], config, directory
    )
    print(f"Grafici salvati in {directory}")


if __name__ == "__main__":
    main()

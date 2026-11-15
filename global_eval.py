import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATHS = {
    "Real": PROJECT_ROOT / "experiments/baseline/config.json",
    "Latent Unmixing": PROJECT_ROOT / "experiments/latent_unmixing_per_class/config.json",
    "Baseline": PROJECT_ROOT / "experiments/baseline/config.json",
    "Stable Diffusion": PROJECT_ROOT / "kaggle/config.json",
}
COLORS = {
    "Real": "#111111",
    "Latent Unmixing": "#1F77B4",
    "Baseline": "#2CA02C",
    "Stable Diffusion": "#FF7F0E",
}
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs/results/global_eval"


def load_config(path):
    """Load an experiment config and resolve paths from the project root."""
    config = json.loads(path.read_text(encoding="utf-8"))
    for key, value in config["paths"].items():
        config["paths"][key] = PROJECT_ROOT / value
    return config


def read_region_spectrum(image_path, mask_path, scale, mask_class, unannotated):
    """Return the mean spectrum of annotated pixels or unassigned pixels."""
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


def collect_spectra(region_directory, scale, mask_class, unannotated):
    """Collect one mean spectrum per image for a selected pixel region."""
    spectra = []
    for image_path in sorted((region_directory / "tiff").glob("*.tif")):
        mask_path = region_directory / "mask" / f"{image_path.stem}_annotation.png"
        spectrum = read_region_spectrum(
            image_path, mask_path, scale, mask_class, unannotated
        )
        if spectrum is not None:
            spectra.append(spectrum)
    if len(spectra) < 2:
        region = "non annotati" if unannotated else f"annotati {mask_class!r}"
        raise ValueError(
            f"Servono almeno due spettri per i pixel {region}: {region_directory}"
        )
    return np.stack(spectra)


def collect_all_spectra(configs, mask_class):
    """Load shared real references and generated spectra from all approaches."""
    real_config = configs["Real"]
    real_directory = real_config["paths"]["annotation_images"] / "real"
    groups = {}
    for unannotated in (False, True):
        spectra = {
            "Real": collect_spectra(
                real_directory,
                real_config["data"]["normalization_max"],
                mask_class,
                unannotated,
            )
        }
        for approach in ("Latent Unmixing", "Baseline", "Stable Diffusion"):
            config = configs[approach]
            generated_directory = config["paths"]["annotation_images"] / "fake"
            spectra[approach] = collect_spectra(
                generated_directory,
                1.0,
                mask_class,
                unannotated,
            )
        groups[unannotated] = spectra
    return groups


def plot_rmc_comparison(groups, wavelengths, y_limits, output_path):
    """Plot mean RMC curves for annotated and unannotated pixels."""
    order = np.argsort(wavelengths)
    wavelengths = np.asarray(wavelengths)[order]
    figure, axes = plt.subplots(1, 2, figsize=(17, 6), sharey=True)
    for axis, unannotated in zip(axes, (False, True)):
        for approach, spectra in groups[unannotated].items():
            mean = spectra.mean(axis=0)[order]
            color = COLORS[approach]
            axis.plot(wavelengths, mean, color=color, linewidth=2, label=approach)
        axis.set_title("Unannotated pixels" if unannotated else "Annotated pixels")
        axis.set_xlabel("Wavelength (nm)")
        axis.set_ylim(*y_limits)
        axis.grid(alpha=0.2, linestyle="--")
    axes[0].set_ylabel("Reflectance")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_rmc_std_parallel(groups, band_names, wavelengths, output_path):
    """Plot per-band RMC standard deviations in parallel coordinates."""
    order = np.argsort(wavelengths)
    labels = [band_names[index] for index in order]
    positions = np.arange(len(labels))
    stds = {
        unannotated: {
            approach: spectra.std(axis=0, ddof=1)[order]
            for approach, spectra in groups[unannotated].items()
        }
        for unannotated in (False, True)
    }
    maximum = max(float(values.max()) for group in stds.values() for values in group.values())
    y_max = max(maximum * 1.05, 1e-6)
    figure, axes = plt.subplots(1, 2, figsize=(21, 6), sharey=True)

    for axis, unannotated in zip(axes, (False, True)):
        axis.vlines(positions, 0, y_max, color="#555555", linewidth=0.8, zorder=0)
        for approach, values in stds[unannotated].items():
            axis.plot(
                positions,
                values,
                color=COLORS[approach],
                linewidth=2,
                label=approach,
                zorder=2,
            )
        axis.set_xticks(positions, labels)
        axis.xaxis.set_ticks_position("top")
        axis.tick_params(
            axis="x", top=True, labeltop=True, bottom=False, labelbottom=False, pad=7
        )
        axis.set_xlim(-0.4, positions[-1] + 0.4)
        axis.set_ylim(0, y_max)
        axis.set_title("Unannotated pixels" if unannotated else "Annotated pixels", pad=28)
        axis.grid(False)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["bottom"].set_visible(False)

    axes[0].set_ylabel("Standard deviation of reflectance")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    figure.subplots_adjust(left=0.07, right=0.99, top=0.82, bottom=0.06, wspace=0.12)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    """Load all spectra and save the global comparison plots."""
    configs = {name: load_config(path) for name, path in CONFIG_PATHS.items()}
    baseline_config = configs["Baseline"]
    mask_class = baseline_config["annotation_evaluation"]["mask_class"]
    groups = collect_all_spectra(configs, mask_class)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    plot_rmc_std_parallel(
        groups,
        baseline_config["data"]["band_names"],
        baseline_config["data"]["wavelengths_nm"],
        OUTPUT_DIRECTORY / "global_rmc_std_parallel.png",
    )
    plot_rmc_comparison(
        groups,
        baseline_config["data"]["wavelengths_nm"],
        baseline_config["annotation_evaluation"]["y_limits"],
        OUTPUT_DIRECTORY / "global_rmc_comparison.png",
    )

    for unannotated, name in ((False, "Annotated pixels"), (True, "Unannotated pixels")):
        counts = ", ".join(f"{method}: {len(values)}" for method, values in groups[unannotated].items())
        print(f"{name} — {counts}")
    print(f"Plots saved in {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()

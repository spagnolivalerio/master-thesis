"""Generate RGB samples from DDPM and LLUM and compare them with EuroSAT RGB.

Run from any directory with: python fid_rgb_comparison.py
Requires clean-fid (pip install clean-fid).
"""

import csv
import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import EuroSATMSDataset
from experiments.DDPM.diffusion import create_sampling_scheduler as create_ddpm_scheduler
from experiments.DDPM.diffusion import generate as generate_ddpm
from experiments.DDPM.model import create_model as create_ddpm_model
from experiments.DDPM.utils import latest_checkpoint as latest_ddpm_checkpoint
from experiments.DDPM.utils import load_config as load_ddpm_config
from experiments.DDPM.utils import save_generated_pair as save_ddpm_pair
from experiments.DDPM.utils import set_seed as set_ddpm_seed
from experiments.LLUM.model import (
    ClassSpecificLatentUnmixingModel,
    create_ddpm_model as create_llum_model,
    create_sampling_scheduler as create_llum_scheduler,
    generate_ILR,
)
from experiments.LLUM.utils import (
    from_ILR,
    latest_checkpoint as latest_llum_checkpoint,
    load_config as load_llum_config,
    save_generated_pair as save_llum_pair,
    set_seed as set_llum_seed,
)


def next_output_directory():
    root = PROJECT_ROOT / "outputs/results/FID_RGB"
    root.mkdir(parents=True, exist_ok=True)
    index = 1
    while (root / f"FID_RGB-comparison_{index}").exists():
        index += 1
    path = root / f"FID_RGB-comparison_{index}"
    path.mkdir()
    return path


def class_counts(dataset):
    return {
        name: sum(label == index for label in dataset.targets)
        for index, name in enumerate(dataset.classes)
    }


@torch.inference_mode()
def generate_ddpm_samples(config, dataset, destination, device):
    checkpoint_path = latest_ddpm_checkpoint(
        Path(config["paths"]["checkpoints"]), suffix="best"
    )
    if checkpoint_path is None:
        raise FileNotFoundError("Nessun checkpoint best della DDPM")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = create_ddpm_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    scheduler = create_ddpm_scheduler(config)
    counts = class_counts(dataset)

    for label, class_name in enumerate(checkpoint["classes"]):
        ms_output = destination / class_name / "MS"
        rgb_output = destination / class_name / "RGB"
        ms_output.mkdir(parents=True)
        rgb_output.mkdir()
        generator = torch.Generator(device=device).manual_seed(
            config["experiment"]["seed"] + label
        )
        for start in tqdm(
            range(0, counts[class_name], config["sampling"]["batch_size"]),
            desc=f"DDPM {class_name}",
        ):
            batch_size = min(config["sampling"]["batch_size"], counts[class_name] - start)
            labels = torch.full((batch_size,), label, device=device, dtype=torch.long)
            images = generate_ddpm(
                model, scheduler, labels, config,
                config["sampling"]["inference_steps"], generator,
            )
            for offset, image in enumerate(images, start=1):
                index = start + offset
                save_ddpm_pair(
                    image,
                    ms_output / f"{class_name}_{index:04d}.tif",
                    rgb_output / f"{class_name}_{index:04d}.png",
                    checkpoint["band_stats"],
                    config,
                )


@torch.inference_mode()
def generate_llum_samples(config, dataset, destination, device):
    prefix = "LLUM_ddpm"
    checkpoint_path = latest_llum_checkpoint(
        config["paths"]["ddpm_checkpoints"], prefix, "best"
    )
    if checkpoint_path is None:
        raise FileNotFoundError("Nessun checkpoint best di LLUM")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    unmixing = ClassSpecificLatentUnmixingModel(
        config, checkpoint["unmixing_model"]["latent_bases"]
    ).to(device)
    unmixing.load_state_dict(checkpoint["unmixing_model"])
    unmixing.eval()
    model = create_llum_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    scheduler = create_llum_scheduler(config)
    counts = class_counts(dataset)

    for label, class_name in enumerate(checkpoint["classes"]):
        ms_output = destination / class_name / "MS"
        rgb_output = destination / class_name / "RGB"
        ms_output.mkdir(parents=True)
        rgb_output.mkdir()
        generator = torch.Generator(device=device).manual_seed(
            config["experiment"]["seed"] + 10_000 + label
        )
        for start in tqdm(
            range(0, counts[class_name], config["evaluation"]["batch_size"]),
            desc=f"LLUM {class_name}",
        ):
            batch_size = min(
                config["evaluation"]["batch_size"], counts[class_name] - start
            )
            labels = torch.full((batch_size,), label, device=device, dtype=torch.long)
            latents = generate_ILR(
                model, scheduler, labels, config,
                config["evaluation"]["inference_steps"], generator,
            )
            images = unmixing.decode(from_ILR(latents, labels, checkpoint["ilr_stats"]), labels)
            for offset, image in enumerate(images, start=1):
                index = start + offset
                save_llum_pair(
                    image,
                    ms_output / f"{class_name}_{index:04d}.tif",
                    rgb_output / f"{class_name}_{index:04d}.png",
                    checkpoint["band_stats"],
                    config,
                )


def calculate_fid(real_root, generated_root, results_file, settings, device):
    try:
        from cleanfid import fid
    except ImportError as error:
        raise ImportError("Installa clean-fid con: pip install clean-fid") from error

    rows = []
    classes = sorted(path.name for path in generated_root.iterdir() if path.is_dir())
    for class_name in classes:
        score = fid.compute_fid(
            str(real_root / class_name),
            str(generated_root / class_name / "RGB"),
            mode="clean",
            batch_size=settings["batch_size"],
            num_workers=settings["num_workers"],
            device=device,
        )
        rows.append({"class": class_name, "fid": float(score)})
        print(f"{class_name}: FID clean = {score:.6f}", flush=True)

    with results_file.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["class", "fid"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def calculate_overall_fid(run_directory):
    """Compare all classes together, reusing only the saved RGB images."""
    try:
        from cleanfid import fid
    except ImportError as error:
        raise ImportError("Installa clean-fid con: pip install clean-fid") from error

    config = load_ddpm_config()
    settings = config["fid_evaluation"]
    real_root = PROJECT_ROOT / "data/EuroSAT_RGB"
    samples_root = run_directory / "samples"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []

    with tempfile.TemporaryDirectory(prefix="fid_rgb_aggregate_") as temporary:
        temporary = Path(temporary)
        real_flat = temporary / "real"
        real_flat.mkdir()
        for class_directory in sorted(path for path in real_root.iterdir() if path.is_dir()):
            for image in sorted(class_directory.iterdir()):
                if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS:
                    (real_flat / f"{class_directory.name}_{image.name}").symlink_to(image.resolve())

        for model_name in ("DDPM", "LLUM"):
            generated_flat = temporary / model_name
            generated_flat.mkdir()
            for class_directory in sorted(
                path for path in (samples_root / model_name).iterdir() if path.is_dir()
            ):
                rgb_directory = class_directory / "RGB"
                for image in sorted(rgb_directory.iterdir()):
                    if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS:
                        (generated_flat / f"{class_directory.name}_{image.name}").symlink_to(
                            image.resolve()
                        )

            real_count = sum(1 for path in real_flat.iterdir() if path.is_file())
            generated_count = sum(1 for path in generated_flat.iterdir() if path.is_file())
            if real_count != generated_count:
                raise ValueError(
                    f"Conteggi diversi per la FID complessiva: reali={real_count}, "
                    f"{model_name}={generated_count}"
                )
            score = fid.compute_fid(
                str(real_flat),
                str(generated_flat),
                mode="clean",
                batch_size=settings["batch_size"],
                num_workers=settings["num_workers"],
                device=device,
            )
            rows.append({"model": model_name, "fid": float(score), "images": real_count})
            print(f"FID complessiva {model_name} vs REAL: {score:.6f}", flush=True)

    with (run_directory / "fid_overall.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=["model", "fid", "images"])
        writer.writeheader()
        writer.writerows(rows)

    results_path = run_directory / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    results["overall"] = {row["model"]: row for row in rows}
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate samples and compute clean-FID")
    parser.add_argument(
        "--overall-existing",
        type=Path,
        help="Compute overall FID from an existing FID_RGB-comparison directory",
    )
    args = parser.parse_args()
    if args.overall_existing:
        run_directory = args.overall_existing
        if not run_directory.is_absolute():
            run_directory = PROJECT_ROOT / run_directory
        calculate_overall_fid(run_directory)
        return

    ddpm_config = load_ddpm_config()
    llum_config = load_llum_config()
    dataset = EuroSATMSDataset(ddpm_config["data"]["root"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = next_output_directory()
    real_root = PROJECT_ROOT / "data/EuroSAT_RGB"
    counts = class_counts(dataset)
    real_counts = {
        name: sum(
            path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
            for path in (real_root / name).iterdir()
            if path.is_file()
        )
        for name in dataset.classes
    }
    if counts != real_counts:
        raise ValueError(
            "EuroSAT_MS ed EuroSAT_RGB devono avere lo stesso numero di immagini "
            f"per classe per un confronto bilanciato: MS={counts}, RGB={real_counts}"
        )
    (output / "image_counts.json").write_text(
        json.dumps(counts, indent=2), encoding="utf-8"
    )

    print(f"Generazione su {device}; conteggi per classe: {counts}", flush=True)
    ddpm_generated = output / "samples/DDPM"
    llum_generated = output / "samples/LLUM"
    ddpm_generated.mkdir(parents=True)
    llum_generated.mkdir(parents=True)
    set_ddpm_seed(ddpm_config["experiment"]["seed"])
    generate_ddpm_samples(ddpm_config, dataset, ddpm_generated, device)
    set_llum_seed(llum_config["experiment"]["seed"])
    generate_llum_samples(llum_config, dataset, llum_generated, device)

    if not real_root.is_dir():
        raise FileNotFoundError(f"Dataset RGB non trovato: {real_root}")
    results = {}
    for name, generated in (("DDPM", ddpm_generated), ("LLUM", llum_generated)):
        print(f"Calcolo FID clean per {name}", flush=True)
        results[name] = calculate_fid(
            real_root,
            generated,
            output / f"fid_{name.lower()}.csv",
            ddpm_config["fid_evaluation"],
            str(device),
        )
    (output / "results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(f"Risultati e immagini salvati in {output}", flush=True)


if __name__ == "__main__":
    main()

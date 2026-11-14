"""Evaluate class-conditioned generation with bases from other classes."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import EuroSATMSDataset
from experiments.latent_unmixing_per_class.evaluation import save_plots, spectra
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
    parser.add_argument("--images-per-class", type=int)
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
    """Generate each target class using a randomly chosen non-target basis bank."""
    scheduler = create_sampling_scheduler(config)
    base_counts = {name: Counter() for name in checkpoint["classes"]}
    progress = tqdm(
        total=settings["images_per_class"] * len(checkpoint["classes"]),
        desc="Ablation per classe",
        dynamic_ncols=True,
    )

    for target_index, class_name in enumerate(checkpoint["classes"]):
        ms_directory = root / class_name / "MS"
        rgb_directory = root / class_name / "RGB"
        ms_directory.mkdir(parents=True)
        rgb_directory.mkdir()
        generated = 0
        while generated < settings["images_per_class"]:
            batch_size = min(
                settings["batch_size"], settings["images_per_class"] - generated
            )
            target_labels = torch.full(
                (batch_size,), target_index, dtype=torch.long, device=device
            )
            # Draw uniformly from all classes except the requested target class.
            base_labels = torch.randint(
                len(checkpoint["classes"]) - 1,
                (batch_size,),
                generator=generator,
                device=device,
            )
            base_labels += base_labels >= target_index
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

            for image, base_index in zip(images, base_labels.tolist()):
                generated += 1
                base_counts[class_name][checkpoint["classes"][base_index]] += 1
                stem = f"{class_name}_{generated:04d}"
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


def real_test_spectra(dataset, checkpoint, classes, images_per_class, scale):
    """Load test-set TIFF paths grouped by class and compute their spectra."""
    paths_by_class = {name: [] for name in classes}
    for index in checkpoint["test_indices"]:
        path, label = dataset.samples[index]
        paths_by_class[dataset.classes[label]].append(path)
    for name, paths in paths_by_class.items():
        if len(paths) < images_per_class:
            raise ValueError(
                f"Not enough real test images for {name}: found {len(paths)}"
            )
    return {
        name: spectra(paths_by_class[name][:images_per_class], scale)
        for name in classes
    }


def main():
    """Run the ablation, save generated pairs, and compare class RMC curves."""
    args = parse_args()
    config = load_config(args.config)
    settings = {
        "images_per_class": (
            config["evaluation"]["images_per_class"]
            if args.images_per_class is None
            else args.images_per_class
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
    if min(settings["images_per_class"], settings["batch_size"], settings["inference_steps"]) <= 0:
        raise ValueError(
            "Images per class, batch size, and inference steps must be positive"
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
    real = real_test_spectra(
        dataset,
        checkpoint,
        classes,
        settings["images_per_class"],
        config["data"]["normalization_max"],
    )

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
    metrics = save_plots(
        values,
        config,
        result_directory / "plots",
        title="Real and generated RMCs with random non-target class bases",
    )
    summary = {
        "checkpoint": str(checkpoint_path),
        "settings": settings,
        "base_class_counts": {
            name: dict(counts) for name, counts in base_counts.items()
        },
        "per_class_metrics": metrics,
        "macro_metrics": {
            key: sum(entry[key] for entry in metrics.values()) / len(metrics)
            for key in next(iter(metrics.values()))
        },
    }
    with (result_directory / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(json.dumps(summary["macro_metrics"], indent=2))
    print(f"Ablation results saved to {result_directory}")


if __name__ == "__main__":
    main()

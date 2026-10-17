import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.baseline.diffusion import create_sampling_scheduler, generate  
from experiments.baseline.model import create_model  
from experiments.baseline.utils import (  
    latest_checkpoint,
    load_config,
    save_generated_pair,
    set_seed,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate baseline samples")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--single", dest="mode", action="store_const", const="single",
        help="Generate samples for one class",
    )
    modes.add_argument(
        "--multi", dest="mode", action="store_const", const="multi",
        help="Generate all classes using the sampling configuration (default)",
    )
    parser.set_defaults(mode="multi")
    parser.add_argument("--class", dest="class_name", help="Class name for --single")
    parser.add_argument("--n", type=int, help="Number of samples for --single")
    args = parser.parse_args(argv)
    if args.mode == "single":
        if args.class_name is None or args.n is None:
            parser.error("--single requires --class and --n")
        if args.n <= 0:
            parser.error("--n must be positive")
    elif args.class_name is not None or args.n is not None:
        parser.error("--class and --n can only be used with --single")
    return args


def next_sampling_directory(root, run_name):
    root.mkdir(parents=True, exist_ok=True)
    index = 1
    while (root / f"baseline-sampling_{index}-{run_name}").exists():
        index += 1
    path = root / f"baseline-sampling_{index}-{run_name}"
    path.mkdir()
    return path


def main(args=None):
    # Calls from evaluation keep the default multi-class sampling behavior.
    if args is None:
        args = parse_args([])
    config = load_config()
    set_seed(config["experiment"]["seed"])
    checkpoint_path = latest_checkpoint(
        Path(config["paths"]["checkpoints"]), suffix="best"
    )
    if checkpoint_path is None:
        raise FileNotFoundError("Nessun checkpoint best della baseline")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    classes = checkpoint["classes"]
    if args.mode == "single":
        if args.class_name not in classes:
            raise ValueError(
                f"Unknown class {args.class_name!r}. Available classes: {', '.join(classes)}"
            )
        selected_classes = [(classes.index(args.class_name), args.class_name)]
        total_samples = samples_per_class = args.n
    else:
        selected_classes = list(enumerate(classes))
        total_samples = config["sampling"]["num_samples"]
        if total_samples % len(classes):
            raise ValueError("sampling.num_samples deve essere divisibile per le classi")
        samples_per_class = total_samples // len(classes)

    model = create_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    scheduler = create_sampling_scheduler(config)
    run_name = checkpoint_path.stem.removesuffix("-best")
    result_directory = next_sampling_directory(
        Path(config["paths"]["results"]), run_name
    )
    samples_directory = result_directory / "samples"
    samples_directory.mkdir()

    for label, class_name in selected_classes:
        ms_directory = samples_directory / class_name / "MS"
        rgb_directory = samples_directory / class_name / "RGB"
        ms_directory.mkdir(parents=True)
        rgb_directory.mkdir()
        progress = tqdm(total=samples_per_class, desc=class_name, dynamic_ncols=True)
        generator = torch.Generator(device=device).manual_seed(
            config["experiment"]["seed"] + label
        )
        for start in range(0, samples_per_class, config["sampling"]["batch_size"]):
            batch_size = min(
                config["sampling"]["batch_size"], samples_per_class - start
            )
            labels = torch.full((batch_size,), label, device=device, dtype=torch.long)
            images = generate(
                model,
                scheduler,
                labels,
                config,
                config["sampling"]["inference_steps"],
                generator,
            )
            for offset, image in enumerate(images, start=1):
                index = start + offset
                stem = f"{class_name}_{index:04d}"
                save_generated_pair(
                    image,
                    ms_directory / f"{stem}.tif",
                    rgb_directory / f"{stem}.png",
                    checkpoint["band_stats"],
                    config,
                )
            progress.update(batch_size)
        progress.close()

    with (result_directory / "generation_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            {
                "checkpoint": str(checkpoint_path.resolve()),
                "checkpoint_epoch": checkpoint.get("epoch"),
                "weights": checkpoint.get("weights"),
                "mode": args.mode,
                "classes": [class_name for _, class_name in selected_classes],
                "num_samples": total_samples,
                "inference_steps": config["sampling"]["inference_steps"],
                "guidance_scale": config["cfg"]["guidance_scale"],
                "seed": config["experiment"]["seed"],
            },
            file,
            indent=2,
        )
    print(f"Risultati salvati in {result_directory}")
    return result_directory


if __name__ == "__main__":
    main(parse_args())


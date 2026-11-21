"""Compute overall clean-FID from an existing FID_RGB comparison run."""

import argparse
import csv
import json
import tempfile
from pathlib import Path

import torch
from cleanfid import fid


PROJECT_ROOT = Path(__file__).resolve().parent
REAL_ROOT = PROJECT_ROOT / "data/EuroSAT_RGB"
RUNS_ROOT = PROJECT_ROOT / "outputs/results/FID_RGB"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def latest_run():
    runs = sorted(RUNS_ROOT.glob("FID_RGB-comparison_*"))
    if not runs:
        raise FileNotFoundError(f"Nessun run trovato in {RUNS_ROOT}")
    return max(runs, key=lambda path: int(path.name.rsplit("_", 1)[1]))


def images_in(directory):
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def link_dataset(source_directories, destination):
    destination.mkdir()
    count = 0
    for class_name, source in source_directories:
        for image in images_in(source):
            (destination / f"{class_name}_{image.name}").symlink_to(image.resolve())
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Calcola la clean-FID complessiva dai campioni RGB esistenti"
    )
    parser.add_argument(
        "run_directory",
        nargs="?",
        type=Path,
        help="Run da valutare; se omesso usa il run FID_RGB più recente",
    )
    args = parser.parse_args()
    run_directory = args.run_directory or latest_run()
    if not run_directory.is_absolute():
        run_directory = PROJECT_ROOT / run_directory
    run_directory = run_directory.resolve()

    real_classes = sorted(path for path in REAL_ROOT.iterdir() if path.is_dir())
    real_sources = [(path.name, path) for path in real_classes]
    real_counts = {name: len(images_in(path)) for name, path in real_sources}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = json.loads(
        (PROJECT_ROOT / "experiments/DDPM/config.json").read_text(encoding="utf-8")
    )
    settings = config["fid_evaluation"]
    results = []

    with tempfile.TemporaryDirectory(dir=run_directory, prefix="fid_overall_") as temp:
        temp = Path(temp)
        real_flat = temp / "real"
        real_count = link_dataset(real_sources, real_flat)

        for model_name in ("LLUM", "DDPM"):
            generated_root = run_directory / "samples" / model_name
            generated_sources = []
            generated_counts = {}
            for class_name in real_counts:
                rgb_directory = generated_root / class_name / "RGB"
                generated_sources.append((class_name, rgb_directory))
                generated_counts[class_name] = len(images_in(rgb_directory))
            if generated_counts != real_counts:
                raise ValueError(
                    f"Conteggi per classe diversi per {model_name}: "
                    f"reali={real_counts}, generati={generated_counts}"
                )

            generated_flat = temp / model_name
            generated_count = link_dataset(generated_sources, generated_flat)
            if generated_count != real_count:
                raise ValueError(
                    f"Conteggi totali diversi: reali={real_count}, "
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
            row = {"model": model_name, "fid": float(score), "images": real_count}
            results.append(row)
            print(f"{model_name} vs EuroSAT_RGB: FID = {score:.6f}")

    with (run_directory / "fid_overall.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=["model", "fid", "images"])
        writer.writeheader()
        writer.writerows(results)
    (run_directory / "fid_overall.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(f"Risultati salvati in {run_directory}")


if __name__ == "__main__":
    main()

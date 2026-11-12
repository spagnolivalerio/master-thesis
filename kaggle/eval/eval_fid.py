"""FID RGB River: generate–reali e controllo reali–reali sullo stesso set.

Esecuzione dalla root del progetto: python kaggle/eval/eval_fid.py
Dipendenze: pip install "scipy<1.18" pytorch-fid==0.3.0
Percorsi e batch size sono definiti in kaggle/config.json.
"""

import csv
import json
from pathlib import Path

import torch
from pytorch_fid.fid_score import calculate_fid_given_paths


KAGGLE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = KAGGLE_ROOT.parent
FEATURE_DIMENSIONS = 2048
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def load_config():
    """Legge il config locale; i percorsi sono relativi alla root del progetto."""
    path = KAGGLE_ROOT / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    for key in ("fid_real_images", "fid_fake_images", "fid_results"):
        config["paths"][key] = PROJECT_ROOT / config["paths"][key]
    return config


def image_paths(directory):
    """Elenca le immagini RGB della sola cartella richiesta, in ordine stabile."""
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def evaluate_class(real_directory, fake_directory, device, settings):
    """Confronta tutte le immagini reali e generate di una singola classe."""
    real_paths = image_paths(real_directory)
    fake_paths = image_paths(fake_directory)
    for directory, paths in ((real_directory, real_paths), (fake_directory, fake_paths)):
        if len(paths) < 2:
            raise ValueError(f"La FID richiede almeno due immagini RGB in {directory}")

    arguments = {
        "batch_size": settings["batch_size"],
        "device": device,
        "dims": FEATURE_DIMENSIONS,
        "num_workers": settings["num_workers"],
    }
    fid_value = calculate_fid_given_paths(
        [str(real_directory), str(fake_directory)], **arguments
    )
    fid_real_real = calculate_fid_given_paths(
        [str(real_directory), str(real_directory)], **arguments
    )
    return {
        "class": real_directory.name,
        "real_images": len(real_paths),
        "fake_images": len(fake_paths),
        "fid": float(fid_value),
        "fid_real_real": fid_real_real,
    }


def format_table(rows):
    """Tabella a colonne allineate, leggibile anche senza un foglio di calcolo."""
    width = max(len("Classe"), *(len(row["class"]) for row in rows))
    header = (
        f"{'Classe':<{width}} | {'Reali':>7} | {'Generate':>8} | "
        f"{'FID gen-real':>14} | {'FID real-real':>14}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row['class']:<{width}} | {row['real_images']:>7} | "
            f"{row['fake_images']:>8} | {row['fid']:>14.6f} | {row['fid_real_real']:>14.6f}"
        )
    return "\n".join(lines)


def save_results(rows, config):
    """Salva conteggi, FID e provenienza delle immagini in TXT e CSV."""
    paths = config["paths"]
    directory = paths["fid_results"]
    directory.mkdir(parents=True, exist_ok=True)
    report = "\n".join([
        "FID RGB per classe — feature pytorch-fid 0.3.0, Inception-v3 pool3 (2048)",
        "Calcolo end-to-end con calculate_fid_given_paths di pytorch-fid.",
        f"Reali: {paths['fid_real_images']} (dataset completo per classe)",
        f"Generate: {paths['fid_fake_images']}",
        "Tutte le immagini disponibili; nessun sottocampionamento.",
        "FID real-real: stesso identico set e stesse statistiche; controllo numerico atteso vicino a zero.",
        "FID inferiore indica distribuzioni RGB più simili; non misura la coerenza multispettrale.",
        "",
        format_table(rows),
        "",
    ])
    (directory / "fid_per_class.txt").write_text(report, encoding="utf-8")
    with (directory / "fid_per_class.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=["class", "real_images", "fake_images", "fid", "fid_real_real"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(report)
    print(f"Tabelle salvate in {directory}")


def main():
    config = load_config()
    paths = config["paths"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"FID end-to-end con pytorch-fid su {device} — {paths['fid_real_images'].name}", flush=True)

    row = evaluate_class(
        paths["fid_real_images"], paths["fid_fake_images"],
        device, config["fid_evaluation"],
    )
    print(
        f"{row['class']} — FID gen-real: {row['fid']:.6f}; "
        f"FID real-real: {row['fid_real_real']:.6f}", flush=True,
    )
    save_results([row], config)


if __name__ == "__main__":
    main()

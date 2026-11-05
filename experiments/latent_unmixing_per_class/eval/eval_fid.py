"""FID RGB per classe: generate–reali e controllo reali–reali sullo stesso set.

Esecuzione: python experiments/<esperimento>/eval/eval_fid.py
Dipendenza aggiuntiva: pip install pytorch-fid==0.3.0
Percorsi e batch size sono definiti nel config.json dell'esperimento.
Lo script può essere copiato senza modifiche nella cartella eval di un altro
esperimento con la stessa configurazione FID.
"""

import csv
import json
from pathlib import Path

import numpy as np
import torch
from pytorch_fid.fid_score import calculate_activation_statistics
from pytorch_fid.inception import InceptionV3


EXPERIMENT_DIRECTORY = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_DIRECTORY.parents[1]
FEATURE_DIMENSIONS = 2048
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def load_config():
    """Legge il config locale; i percorsi sono relativi alla root del progetto."""
    path = EXPERIMENT_DIRECTORY / "config.json"
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


def frechet_distance(real_mean, real_covariance, fake_mean, fake_covariance):
    """Distanza fra gaussiane: ||m_r-m_f||² + tr(C_r+C_f-2√(√C_r C_f √C_r))."""
    difference = real_mean.astype(np.float64) - fake_mean.astype(np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(real_covariance)
    # Le covarianze sono semidefinite positive; il clip elimina il roundoff negativo.
    real_sqrt = (eigenvectors * np.sqrt(eigenvalues.clip(min=0))) @ eigenvectors.T
    product = real_sqrt @ fake_covariance @ real_sqrt
    product = (product + product.T) / 2
    cross_trace = np.sqrt(np.linalg.eigvalsh(product).clip(min=0)).sum()
    return float(
        difference @ difference + np.trace(real_covariance)
        + np.trace(fake_covariance) - 2 * cross_trace
    )


def evaluate_class(real_directory, fake_directory, model, device, settings):
    """Confronta tutte le immagini reali e generate di una singola classe."""
    real_paths = image_paths(real_directory)
    fake_paths = image_paths(fake_directory)
    for directory, paths in ((real_directory, real_paths), (fake_directory, fake_paths)):
        if len(paths) < 2:
            raise ValueError(f"La FID richiede almeno due immagini RGB in {directory}")

    statistics = []
    for label, paths in (("Real", real_paths), ("Fake", fake_paths)):
        print(f"{fake_directory.name} — {label}: {len(paths)} immagini", flush=True)
        statistics.append(calculate_activation_statistics(
            paths, model,
            batch_size=settings["batch_size"],
            dims=FEATURE_DIMENSIONS,
            device=device,
            num_workers=settings["num_workers"],
        ))
    real_mean, real_covariance = statistics[0]
    fake_mean, fake_covariance = statistics[1]
    fid = frechet_distance(
        real_mean, real_covariance, fake_mean, fake_covariance,
    )
    # Riusa le statistiche dello stesso set reale, senza ripetere l'inferenza.
    fid_real_real = frechet_distance(
        real_mean, real_covariance, real_mean, real_covariance,
    )
    return {
        "class": fake_directory.name,
        "real_images": len(real_paths),
        "fake_images": len(fake_paths),
        "fid": float(fid),
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
        "Distanza Fréchet in float64 mediante covarianze simmetriche.",
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
    classes = sorted(path for path in paths["fid_fake_images"].iterdir() if path.is_dir())
    if not classes:
        raise ValueError(f"Nessuna cartella di classe in {paths['fid_fake_images']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    block = InceptionV3.BLOCK_INDEX_BY_DIM[FEATURE_DIMENSIONS]
    model = InceptionV3([block]).to(device).eval()
    print(f"FID su {device} — {len(classes)} classi", flush=True)

    rows = []
    for fake_directory in classes:
        row = evaluate_class(
            paths["fid_real_images"] / fake_directory.name,
            fake_directory, model, device, config["fid_evaluation"],
        )
        rows.append(row)
        print(
            f"{row['class']} — FID gen-real: {row['fid']:.6f}; "
            f"FID real-real: {row['fid_real_real']:.6f}", flush=True,
        )
    save_results(rows, config)


if __name__ == "__main__":
    main()

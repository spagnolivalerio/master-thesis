"""Sampling per classe: DDPM sui logits e proiezione softmax."""

import argparse
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.latent_unmixing_per_class_softmax.evaluation import generate_samples
from experiments.latent_unmixing_per_class_softmax.model import (
    ClassSpecificLatentUnmixingModel,
    create_ddpm_model,
)
from experiments.latent_unmixing_per_class_softmax.utils import (
    latest_checkpoint,
    load_config,
    set_seed,
)


def main():
    config = load_config()
    parser = argparse.ArgumentParser(description="Sampling di una classe EuroSAT")
    parser.add_argument("--class", dest="class_name", help="Omettere per tutte le classi")
    parser.add_argument("--n", dest="images_per_class", type=int, required=True)
    args = parser.parse_args()
    if args.images_per_class <= 0:
        parser.error("--n deve essere positivo")
    args.batch_size = config["evaluation"]["batch_size"]
    args.inference_steps = config["evaluation"]["inference_steps"]
    args.suffix = ""
    set_seed(config["experiment"]["seed"])
    prefix = "latent_unmixing_per_class_softmax_ddpm"
    checkpoint_path = latest_checkpoint(config["paths"]["ddpm_checkpoints"], prefix, "best")
    if checkpoint_path is None:
        raise FileNotFoundError("Nessun checkpoint best del DDPM per-class")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if args.class_name is not None and args.class_name not in checkpoint["classes"]:
        parser.error(f"Classe sconosciuta: {args.class_name}")
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
    root = Path(config["paths"]["results"])
    root.mkdir(parents=True, exist_ok=True)
    index = 1
    name = config["experiment"]["name"]
    while (root / f"{name}-sampling_{index}-{run_name}").exists():
        index += 1
    directory = root / f"{name}-sampling_{index}-{run_name}"
    generate_samples(ddpm, unmixing, checkpoint, config, args, directory / "samples", device)
    print(f"Risultati salvati in {directory}")


if __name__ == "__main__":
    main()

"""Orchestratore dei due stadi per la variante class-specific."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from experiments.latent_unmixing_per_class import train_ddpm, train_encoder  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Training latent unmixing per class")
    parser.add_argument(
        "--stage", choices=("all", "encoder", "ddpm"), default="all"
    )
    args = parser.parse_args()
    if args.stage in ("all", "encoder"):
        train_encoder.main()
    if args.stage in ("all", "ddpm"):
        train_ddpm.main()


if __name__ == "__main__":
    main()


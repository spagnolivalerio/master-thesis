import argparse
import sys
from pathlib import Path


NEWREPO_ROOT = Path(__file__).resolve().parents[2]
if str(NEWREPO_ROOT) not in sys.path:
    sys.path.insert(0, str(NEWREPO_ROOT))


from experiments.latent_unmixing import train_ddpm, train_encoder


def parse_args():
    parser = argparse.ArgumentParser(description="Training latent unmixing")
    parser.add_argument(
        "--stage",
        choices=("all", "encoder", "ddpm"),
        default="all",
        help="Stadio da eseguire; per default esegue encoder e poi DDPM",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.stage in ("all", "encoder"):
        train_encoder.main()
    if args.stage in ("all", "ddpm"):
        train_ddpm.main()


if __name__ == "__main__":
    main()


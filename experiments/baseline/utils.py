"""Utility locali della baseline; il modulo non dipende da ``common``."""

import json
import math
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader, Sampler, Subset
from torchvision.utils import save_image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).with_name("config.json")


class BalancedBatchSampler(Sampler):
    """Produce batch con lo stesso numero di esempi per ogni classe."""

    def __init__(self, indices, targets, batch_size, num_classes, seed):
        if batch_size % num_classes:
            raise ValueError("batch_size deve essere multiplo di num_classes")
        self.samples_per_class = batch_size // num_classes
        self.seed = seed
        self.epoch = 0
        self.class_indices = {label: [] for label in range(num_classes)}
        for index in indices:
            self.class_indices[targets[index]].append(index)
        empty = [str(label) for label, values in self.class_indices.items() if not values]
        if empty:
            raise ValueError("Classi prive di campioni nello split: " + ", ".join(empty))
        largest = max(len(values) for values in self.class_indices.values())
        self.num_batches = math.ceil(largest / self.samples_per_class)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        required = self.num_batches * self.samples_per_class
        pools = {}
        for label, indices in self.class_indices.items():
            pool = []
            for _ in range(math.ceil(required / len(indices))):
                order = torch.randperm(len(indices), generator=generator).tolist()
                pool.extend(indices[position] for position in order)
            pools[label] = pool[:required]

        for batch_index in range(self.num_batches):
            start = batch_index * self.samples_per_class
            end = start + self.samples_per_class
            batch = [
                index
                for label in sorted(pools)
                for index in pools[label][start:end]
            ]
            order = torch.randperm(len(batch), generator=generator).tolist()
            yield [batch[position] for position in order]


def load_config(path=CONFIG_PATH):
    with Path(path).open(encoding="utf-8") as file:
        config = json.load(file)
    config["data"]["root"] = str(PROJECT_ROOT / config["data"]["root"])
    for name, value in config["paths"].items():
        config["paths"][name] = str(PROJECT_ROOT / value)
    return config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_random_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_random_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def split_indices(dataset, config):
    split = config["data"]["split"]
    if not math.isclose(sum(split.values()), 1.0):
        raise ValueError("Le percentuali dello split devono sommare a 1")
    generator = torch.Generator().manual_seed(config["experiment"]["seed"])
    train, validation, test = [], [], []
    for label in range(len(dataset.classes)):
        indices = [i for i, target in enumerate(dataset.targets) if target == label]
        order = torch.randperm(len(indices), generator=generator).tolist()
        indices = [indices[position] for position in order]
        train_end = int(len(indices) * split["train"])
        validation_end = train_end + int(len(indices) * split["validation"])
        train.extend(indices[:train_end])
        validation.extend(indices[train_end:validation_end])
        test.extend(indices[validation_end:])
    return train, validation, test


def compute_band_stats(dataset, train_indices, config):
    loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=config["data"]["statistics_batch_size"],
        num_workers=config["training"]["num_workers"],
        pin_memory=config["training"]["pin_memory"],
    )
    channels = config["data"]["channels"]
    band_sum = torch.zeros(channels, dtype=torch.float64)
    squared_sum = torch.zeros_like(band_sum)
    pixel_count = 0
    scale = config["data"]["normalization_max"]
    for images, _ in loader:
        images = images.double() / scale
        band_sum += images.sum((0, 2, 3))
        squared_sum += images.square().sum((0, 2, 3))
        pixel_count += images.shape[0] * images.shape[2] * images.shape[3]
    mean = band_sum / pixel_count
    std = (squared_sum / pixel_count - mean.square()).clamp_min(0).sqrt()
    return {
        "mean": mean.float().tolist(),
        "standard_deviation": std.float().tolist(),
        "dataset": {
            "root": config["data"]["root"],
            "seed": config["experiment"]["seed"],
            "split": config["data"]["split"],
            "scale": scale,
        },
    }


def load_or_compute_band_stats(dataset, train_indices, config):
    path = Path(config["paths"]["checkpoints"]) / "band_stats.json"
    expected = {
        "root": config["data"]["root"],
        "seed": config["experiment"]["seed"],
        "split": config["data"]["split"],
        "scale": config["data"]["normalization_max"],
    }
    if path.exists():
        with path.open(encoding="utf-8") as file:
            stats = json.load(file)
        if stats.get("dataset") == expected:
            return stats
    stats = compute_band_stats(dataset, train_indices, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(stats, file, indent=2)
    return stats


def apply_normalization(dataset, stats, config):
    mean = torch.tensor(stats["mean"])[:, None, None]
    std = torch.tensor(stats["standard_deviation"])[:, None, None]
    scale = config["data"]["normalization_max"]

    def transform(image):
        return (image / scale - mean) / std

    dataset.transform = transform


def denormalize(images, stats):
    shape = (1, -1, 1, 1) if images.ndim == 4 else (-1, 1, 1)
    mean = torch.as_tensor(stats["mean"], device=images.device).view(shape)
    std = torch.as_tensor(
        stats["standard_deviation"], device=images.device
    ).view(shape)
    return images * std + mean


def append_log(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": datetime.now().isoformat(timespec="seconds"), **values}
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")


def next_run_directory(root, experiment_name):
    root.mkdir(parents=True, exist_ok=True)
    index = 1
    while (root / f"{experiment_name}-run_{index}").exists():
        index += 1
    path = root / f"{experiment_name}-run_{index}"
    path.mkdir()
    return path


def latest_checkpoint(directory, suffix="latest"):
    paths = list(directory.glob(f"baseline-run_*-{suffix}.pt"))
    return max(paths, key=lambda path: path.stat().st_mtime) if paths else None


def save_preview(images, path, stats, config):
    reflectance = denormalize(images.float().cpu(), stats).clamp_min(0)
    rgb = reflectance[:, [3, 2, 1]] / config["data"]["rgb_display_max"]
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(rgb.clamp(0, 1), path, nrow=5)


def save_generated_pair(image, ms_path, rgb_path, stats, config):
    reflectance = denormalize(image.float().cpu(), stats).clamp_min(0)
    tifffile.imwrite(
        ms_path,
        reflectance.permute(1, 2, 0).numpy().astype(np.float32),
    )
    rgb = reflectance[[3, 2, 1]] / config["data"]["rgb_display_max"]
    save_image(rgb.clamp(0, 1), rgb_path)


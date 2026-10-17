"""Utility locali, incluse le trasformazioni ILR esplicite e invertibili."""

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


def helmert_basis(num_components, *, dtype, device):
    """Base ortonormale [K, K-1] del sottospazio a somma zero."""
    if num_components < 2:
        raise ValueError("La trasformazione ILR richiede almeno due componenti")
    basis = torch.zeros(
        num_components, num_components - 1, dtype=dtype, device=device
    )
    for column in range(num_components - 1):
        denominator = math.sqrt((column + 1) * (column + 2))
        basis[: column + 1, column] = 1.0 / denominator
        basis[column + 1, column] = -(column + 1) / denominator
    return basis


def _raw_ILR(abundances, epsilon):
    if abundances.ndim != 4 or abundances.shape[1] < 2:
        raise ValueError("Le abundance devono avere forma [B, K, H, W], con K >= 2")
    probabilities = abundances.float().clamp_min(epsilon)
    probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)
    logarithms = probabilities.log()
    clr = logarithms - logarithms.mean(dim=1, keepdim=True)
    basis = helmert_basis(
        probabilities.shape[1], dtype=clr.dtype, device=clr.device
    )
    return torch.einsum("bkhw,kq->bqhw", clr, basis)


def _ILR_statistics_tensors(statistics, reference):
    channels = reference.shape[1]
    if statistics["latent_channels"] != channels:
        raise ValueError("Le statistiche ILR non corrispondono ai canali latenti")
    shape = (1, channels, 1, 1)
    mean = torch.as_tensor(
        statistics["mean"], dtype=reference.dtype, device=reference.device
    ).view(shape)
    standard_deviation = torch.as_tensor(
        statistics["standard_deviation"],
        dtype=reference.dtype,
        device=reference.device,
    ).view(shape)
    return mean, standard_deviation


def to_ILR(abundances, statistics):
    """Normalizza le abundance e restituisce coordinate ILR standardizzate.

    La funzione applica floor positivo, chiusura sul simplesso, trasformazione
    ILR di Helmert e standardizzazione con media/deviazione del training set.
    """
    raw = _raw_ILR(abundances, statistics["epsilon"])
    mean, standard_deviation = _ILR_statistics_tensors(statistics, raw)
    return (raw - mean) / standard_deviation


def from_ILR(latents, statistics):
    """De-standardizza coordinate ILR e ricostruisce abundance sul simplesso."""
    if latents.ndim != 4:
        raise ValueError("I latenti ILR devono avere forma [B, K-1, H, W]")
    if not torch.isfinite(latents).all():
        raise ValueError("I latenti ILR contengono NaN o Inf")
    mean, standard_deviation = _ILR_statistics_tensors(statistics, latents)
    raw = latents.float() * standard_deviation + mean
    num_components = statistics["num_components"]
    basis = helmert_basis(num_components, dtype=raw.dtype, device=raw.device)
    clr = torch.einsum("bqhw,kq->bkhw", raw, basis)
    abundances = torch.softmax(clr, dim=1)
    abundances = abundances.clamp_min(statistics["epsilon"])
    return abundances / abundances.sum(dim=1, keepdim=True)


@torch.inference_mode()
def compute_ILR_statistics(model, loader, config, device):
    """Stima media e deviazione ILR usando esclusivamente il training set."""
    model.eval()
    latent_channels = config["model"]["num_bases"] - 1
    values_sum = torch.zeros(latent_channels, dtype=torch.float64)
    squared_sum = torch.zeros_like(values_sum)
    count = 0
    epsilon = config["simplex"]["epsilon"]
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        _, abundances = model.encode(images)
        values = _raw_ILR(abundances, epsilon).double()
        values_sum += values.sum((0, 2, 3)).cpu()
        squared_sum += values.square().sum((0, 2, 3)).cpu()
        count += values.shape[0] * values.shape[2] * values.shape[3]
    mean = values_sum / count
    variance = (squared_sum / count - mean.square()).clamp_min(0)
    standard_deviation = variance.sqrt().clamp_min(
        config["simplex"]["statistics_epsilon"]
    )
    return {
        "transform": "ILR",
        "num_components": config["model"]["num_bases"],
        "latent_channels": latent_channels,
        "epsilon": epsilon,
        "mean": mean.float().tolist(),
        "standard_deviation": standard_deviation.float().tolist(),
    }


class BalancedBatchSampler(Sampler):
    def __init__(self, indices, targets, batch_size, num_classes, seed):
        if batch_size % num_classes:
            raise ValueError("batch_size deve essere multiplo di num_classes")
        self.samples_per_class = batch_size // num_classes
        self.seed = seed
        self.epoch = 0
        self.class_indices = {label: [] for label in range(num_classes)}
        for index in indices:
            self.class_indices[targets[index]].append(index)
        if any(not values for values in self.class_indices.values()):
            raise ValueError("Lo split non contiene tutte le classi")
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


def append_log(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": datetime.now().isoformat(timespec="seconds"), **values}
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")


def split_indices(dataset, config):
    split = config["data"]["split"]
    if not math.isclose(sum(split.values()), 1.0):
        raise ValueError("Le percentuali train/validation/test devono sommare a 1")
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
    settings = config["encoder_training"]
    loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=config["data"]["statistics_batch_size"],
        num_workers=settings["num_workers"],
        pin_memory=settings["pin_memory"],
    )
    total = torch.zeros(config["data"]["channels"], dtype=torch.float64)
    squared = torch.zeros_like(total)
    count = 0
    for images, _ in loader:
        images = images.double() / config["data"]["normalization_max"]
        total += images.sum((0, 2, 3))
        squared += images.square().sum((0, 2, 3))
        count += images.shape[0] * images.shape[2] * images.shape[3]
    mean = total / count
    std = (squared / count - mean.square()).clamp_min(0).sqrt()
    return {
        "mean": mean.float().tolist(),
        "standard_deviation": std.float().tolist(),
        "dataset": {
            "root": config["data"]["root"],
            "seed": config["experiment"]["seed"],
            "split": config["data"]["split"],
            "scale": config["data"]["normalization_max"],
        },
    }


def load_or_compute_band_stats(dataset, train_indices, config):
    path = Path(config["paths"]["encoder_checkpoints"]) / "band_stats.json"
    expected = {
        "root": config["data"]["root"],
        "seed": config["experiment"]["seed"],
        "split": config["data"]["split"],
        "scale": config["data"]["normalization_max"],
    }
    if path.exists():
        with path.open(encoding="utf-8") as file:
            statistics = json.load(file)
        if statistics.get("dataset") == expected:
            return statistics
    statistics = compute_band_stats(dataset, train_indices, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(statistics, file, indent=2)
    return statistics


def apply_band_normalization(dataset, statistics, config):
    mean = torch.tensor(statistics["mean"])[:, None, None]
    std = torch.tensor(statistics["standard_deviation"])[:, None, None]
    scale = config["data"]["normalization_max"]
    dataset.transform = lambda image: (image / scale - mean) / std


def denormalize_bands(images, statistics):
    shape = (1, -1, 1, 1) if images.ndim == 4 else (-1, 1, 1)
    mean = torch.as_tensor(
        statistics["mean"], dtype=images.dtype, device=images.device
    ).view(shape)
    std = torch.as_tensor(
        statistics["standard_deviation"],
        dtype=images.dtype,
        device=images.device,
    ).view(shape)
    return images * std + mean


def sample_training_pixels(dataset, train_indices, num_bases, seed):
    generator = torch.Generator().manual_seed(seed)
    positions = torch.randperm(len(train_indices), generator=generator)[:num_bases]
    basis, sources = [], []
    for position in positions.tolist():
        dataset_index = train_indices[position]
        image, label = dataset[dataset_index]
        height, width = image.shape[1:]
        flat = torch.randint(height * width, (1,), generator=generator).item()
        row, column = divmod(flat, width)
        basis.append(image[:, row, column])
        sources.append(
            {
                "dataset_index": dataset_index,
                "class": dataset.classes[label],
                "row": row,
                "column": column,
            }
        )
    return torch.stack(basis), sources


def next_run_directory(root, prefix):
    root.mkdir(parents=True, exist_ok=True)
    index = 1
    while (root / f"{prefix}-run_{index}").exists():
        index += 1
    path = root / f"{prefix}-run_{index}"
    path.mkdir()
    return path


def latest_checkpoint(directory, prefix, kind="latest"):
    candidates = list(Path(directory).glob(f"{prefix}-run_*-{kind}.pt"))
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def save_generated_pair(image, ms_path, rgb_path, band_stats, config):
    reflectance = denormalize_bands(image.float().cpu(), band_stats).clamp_min(0)
    if not torch.isfinite(reflectance).all():
        raise FloatingPointError("Il sampling ha prodotto NaN o Inf")
    tifffile.imwrite(
        ms_path, reflectance.permute(1, 2, 0).numpy().astype(np.float32)
    )
    rgb = reflectance[[3, 2, 1]] / config["data"]["rgb_display_max"]
    save_image(rgb.clamp(0, 1), rgb_path)


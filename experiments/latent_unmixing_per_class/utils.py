"""Preparazione dei dati, coordinate ILR e gestione delle run dell'esperimento.

Le bande vengono standardizzate con statistiche comuni a tutte le classi.
Le abundance vengono invece trasformate in coordinate ILR e standardizzate
con statistiche distinte per classe, prima di essere usate dal DDPM.

Forme dei tensori: B = batch, C = bande, K = basi, H/W = dimensioni spaziali.
Le immagini hanno forma [B,C,H,W], le abundance [B,K,H,W] e le coordinate
ILR [B,K-1,H,W]. Le statistiche sono stimate esclusivamente sul training set.
"""

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
    """Costruisce K-1 colonne ortonormali a somma zero, in una matrice [K,K-1].

    La base permette di rappresentare i logaritmi centrati delle K abundance
    con K-1 coordinate indipendenti, senza cambiare la risoluzione spaziale.
    """
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
    """Converte le abundance in coordinate ILR, senza standardizzarle."""
    if abundances.ndim != 4 or abundances.shape[1] < 2:
        raise ValueError("Le abundance devono avere forma [B,K,H,W]")
    # Evita log(0).
    positive_abundances = abundances.float().clamp_min(epsilon)
    logarithms = positive_abundances.log()
    # CLR (centered log-ratio): i K logaritmi centrati hanno somma zero.
    clr = logarithms - logarithms.mean(dim=1, keepdim=True)
    basis = helmert_basis(
        positive_abundances.shape[1], dtype=clr.dtype, device=clr.device
    )
    # Proietta da K logaritmi centrati a K-1 coordinate ILR per pixel.
    return torch.einsum("bkhw,kq->bqhw", clr, basis)


def _class_statistics(statistics, labels, reference):
    """Seleziona media/std ILR per campione e le rende broadcastabili sui pixel."""
    if labels.ndim != 1 or len(labels) != len(reference):
        raise ValueError("Servono le label per selezionare le statistiche ILR")
    mean = torch.as_tensor(
        statistics["mean"], dtype=reference.dtype, device=reference.device
    )
    std = torch.as_tensor(
        statistics["standard_deviation"],
        dtype=reference.dtype,
        device=reference.device,
    )
    # Da [num_classes,K-1] a [B,K-1,1,1], sul device e nel dtype del latente.
    class_mean = mean.index_select(0, labels)
    class_std = std.index_select(0, labels)
    return class_mean[:, :, None, None], class_std[:, :, None, None]


def to_ILR(abundances, labels, statistics):
    """Da abundance [B,K,H,W] a coordinate ILR standardizzate [B,K-1,H,W]."""
    raw_latents = _raw_ILR(abundances, statistics["epsilon"])
    mean, std = _class_statistics(statistics, labels, raw_latents)
    return (raw_latents - mean) / std


def from_ILR(latents, labels, statistics):
    """Da ILR standardizzate a abundance positive con somma unitaria per pixel.

    L'inversione recupera le abundance originarie entro gli effetti numerici
    e del clamp a epsilon, usato per evitare componenti nulle.
    """
    if latents.ndim != 4 or not torch.isfinite(latents).all():
        raise ValueError("Latenti ILR non validi")
    mean, std = _class_statistics(statistics, labels, latents)
    raw_latents = latents.float() * std + mean
    basis = helmert_basis(
        statistics["num_components"],
        dtype=raw_latents.dtype,
        device=raw_latents.device,
    )
    # Ricostruisce i K logaritmi centrati; la softmax li riporta sul simplesso,
    # cioè nell'insieme dei pesi positivi con somma uno.
    clr = torch.einsum("bqhw,kq->bkhw", raw_latents, basis)
    abundances = torch.softmax(clr, dim=1).clamp_min(statistics["epsilon"])
    return abundances / abundances.sum(dim=1, keepdim=True)


@torch.inference_mode()
def compute_ILR_statistics(model, loader, config, device):
    """Stima media/std per classe e coordinata ILR su tutti i pixel di training.

    Il chiamante deve fornire un loader del solo training set. Il modello
    viene impostato in eval e non vengono calcolati gradienti.
    """
    model.eval()
    num_classes = config["data"]["num_classes"]
    latent_channels = config["model"]["num_bases"] - 1
    # Accumula somme in float64 su CPU, senza conservare tutti i latenti.
    coordinate_sum = torch.zeros(num_classes, latent_channels, dtype=torch.float64)
    squared_sum = torch.zeros_like(coordinate_sum)
    pixel_counts = torch.zeros(num_classes, dtype=torch.float64)
    epsilon = config["simplex"]["epsilon"]
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        _, abundances = model.encode(images, labels)
        raw_latents = _raw_ILR(abundances, epsilon).double()
        pixels_per_image = raw_latents.shape[2] * raw_latents.shape[3]
        for class_index in labels.unique().tolist():
            selected = labels == class_index
            class_latents = raw_latents[selected]
            coordinate_sum[class_index] += class_latents.sum((0, 2, 3)).cpu()
            squared_sum[class_index] += class_latents.square().sum((0, 2, 3)).cpu()
            pixel_counts[class_index] += int(selected.sum()) * pixels_per_image
    if (pixel_counts == 0).any():
        raise RuntimeError("Impossibile stimare ILR: manca almeno una classe")
    mean = coordinate_sum / pixel_counts[:, None]
    # Varianza di popolazione: E[x²] - E[x]². Il clamp elimina piccoli valori
    # negativi dovuti all'arrotondamento, non una varianza realmente negativa.
    variance = (squared_sum / pixel_counts[:, None] - mean.square()).clamp_min(0)
    std = variance.sqrt().clamp_min(config["simplex"]["statistics_epsilon"])
    return {
        "transform": "ILR_per_class",
        "num_components": config["model"]["num_bases"],
        "latent_channels": latent_channels,
        "num_classes": num_classes,
        "epsilon": epsilon,
        "mean": mean.float().tolist(),
        "standard_deviation": std.float().tolist(),
    }


class BalancedBatchSampler(Sampler):
    """Genera batch con lo stesso numero di immagini per ciascuna classe.

    Gli indici si riferiscono al dataset completo, ma appartengono allo split
    fornito. Le classi più piccole vengono ripetute per riempire tutti i batch.
    L'ordine è riproducibile e cambia con set_epoch().
    """

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
        largest_class_size = max(len(values) for values in self.class_indices.values())
        self.num_batches = math.ceil(largest_class_size / self.samples_per_class)

    def set_epoch(self, epoch):
        """Seleziona l'ordine dei campioni dell'epoca, anche dopo un resume."""
        self.epoch = epoch

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        required_per_class = self.num_batches * self.samples_per_class
        class_pools = {}
        for label, indices in self.class_indices.items():
            pool = []
            # Ogni passaggio mescola tutti gli indici della classe. Ripete i
            # passaggi finché ci sono abbastanza campioni per un'epoca intera.
            num_repeats = math.ceil(required_per_class / len(indices))
            for _ in range(num_repeats):
                order = torch.randperm(len(indices), generator=generator).tolist()
                pool.extend(indices[position] for position in order)
            class_pools[label] = pool[:required_per_class]
        for batch_index in range(self.num_batches):
            start = batch_index * self.samples_per_class
            end = start + self.samples_per_class
            batch = []
            for label in sorted(class_pools):
                batch.extend(class_pools[label][start:end])
            # Mescola anche le posizioni nel batch, senza cambiarne il bilanciamento.
            order = torch.randperm(len(batch), generator=generator).tolist()
            yield [batch[position] for position in order]


def load_config(path=CONFIG_PATH):
    """Legge il JSON e risolve i percorsi rispetto alla root del progetto."""
    with Path(path).open(encoding="utf-8") as file:
        config = json.load(file)
    config["data"]["root"] = str(PROJECT_ROOT / config["data"]["root"])
    for name, relative_path in config["paths"].items():
        config["paths"][name] = str(PROJECT_ROOT / relative_path)
    return config


def set_seed(seed):
    """Inizializza i generatori globali di Python, NumPy e PyTorch (CPU/CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_random_state():
    """Cattura gli stati RNG da inserire nel checkpoint per un resume fedele."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_random_state(state):
    """Ripristina il punto raggiunto dai generatori, anziché riavviarli dal seed."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def append_log(path, values):
    """Aggiunge subito un record JSONL al log della run, con timestamp locale."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": datetime.now().isoformat(timespec="seconds"), **values}
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")


def split_indices(dataset, config):
    """Crea uno split stratificato e riproducibile usando le quote del config.

    Ogni classe viene mescolata separatamente. Training e validation usano
    quote arrotondate per difetto; il test riceve gli indici rimanenti.
    """
    split = config["data"]["split"]
    generator = torch.Generator().manual_seed(config["experiment"]["seed"])
    train_indices, validation_indices, test_indices = [], [], []
    for label in range(len(dataset.classes)):
        indices = [
            index for index, target in enumerate(dataset.targets) if target == label
        ]
        order = torch.randperm(len(indices), generator=generator).tolist()
        indices = [indices[position] for position in order]
        train_end = int(len(indices) * split["train"])
        validation_end = train_end + int(len(indices) * split["validation"])
        train_indices.extend(indices[:train_end])
        validation_indices.extend(indices[train_end:validation_end])
        test_indices.extend(indices[validation_end:])
    return train_indices, validation_indices, test_indices


def compute_band_stats(dataset, train_indices, config):
    """Stima media/std per banda sui pixel di training, nello spazio di riflettanza.

    Il dataset deve fornire i valori originali, prima della standardizzazione.
    Le statistiche sono comuni a tutte le classi e vengono accumulate in float64.
    """
    settings = config["encoder_training"]
    loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=config["data"]["statistics_batch_size"],
        num_workers=settings["num_workers"],
        pin_memory=settings["pin_memory"],
    )
    band_sum = torch.zeros(config["data"]["channels"], dtype=torch.float64)
    squared_sum = torch.zeros_like(band_sum)
    pixel_count = 0
    for images, _ in loader:
        reflectance = images.double() / config["data"]["normalization_max"]
        # Somma su batch, altezza e larghezza: resta un valore per banda.
        band_sum += reflectance.sum((0, 2, 3))
        squared_sum += reflectance.square().sum((0, 2, 3))
        pixel_count += images.shape[0] * images.shape[2] * images.shape[3]
    mean = band_sum / pixel_count
    variance = (squared_sum / pixel_count - mean.square()).clamp_min(0)
    std = variance.sqrt()
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
    """Riusa band_stats.json se la root del dataset coincide, altrimenti lo ricalcola."""
    path = Path(config["paths"]["encoder_checkpoints"]) / "band_stats.json"
    if path.exists():
        with path.open(encoding="utf-8") as file:
            statistics = json.load(file)
        if statistics.get("dataset", {}).get("root") == config["data"]["root"]:
            return statistics
    statistics = compute_band_stats(dataset, train_indices, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(statistics, file, indent=2)
    return statistics


def apply_band_normalization(dataset, statistics, config):
    """Imposta la trasformazione: valore originale → riflettanza → valore standardizzato.

    Media/std hanno forma [C,1,1] e si applicano a tutti i pixel della banda.
    """
    mean = torch.tensor(statistics["mean"])[:, None, None]
    std = torch.tensor(statistics["standard_deviation"])[:, None, None]
    scale = config["data"]["normalization_max"]
    dataset.transform = lambda image: (image / scale - mean) / std


def denormalize_bands(images, statistics):
    """Riporta un'immagine [C,H,W] o un batch [B,C,H,W] in riflettanza.

    Non riapplica il fattore di scala intero del dataset: il risultato contiene
    direttamente valori di riflettanza, come richiesto per i TIFF generati.
    """
    broadcast_shape = (1, -1, 1, 1) if images.ndim == 4 else (-1, 1, 1)
    mean = torch.as_tensor(
        statistics["mean"], dtype=images.dtype, device=images.device
    ).view(broadcast_shape)
    std = torch.as_tensor(
        statistics["standard_deviation"],
        dtype=images.dtype,
        device=images.device,
    ).view(broadcast_shape)
    return images * std + mean


def sample_class_training_pixels(dataset, train_indices, config):
    """Inizializza i banchi [num_classes,K,C] con pixel di training per classe.

    Sceglie K immagini distinte per classe e un pixel casuale da ciascuna.
    Il dataset deve essere già standardizzato. Restituisce anche le posizioni
    dei pixel scelti, da salvare nel checkpoint per tracciarne la provenienza.
    """
    generator = torch.Generator().manual_seed(config["experiment"]["seed"])
    num_classes = config["data"]["num_classes"]
    num_bases = config["model"]["num_bases"]
    basis_banks, sources = [], []
    for class_index in range(num_classes):
        candidates = [
            index for index in train_indices if dataset.targets[index] == class_index
        ]
        positions = torch.randperm(len(candidates), generator=generator)[:num_bases]
        class_spectra, class_sources = [], []
        for position in positions.tolist():
            dataset_index = candidates[position]
            image, _ = dataset[dataset_index]
            height, width = image.shape[1:]
            pixel_index = torch.randint(height * width, (1,), generator=generator).item()
            row, column = divmod(pixel_index, width)
            class_spectra.append(image[:, row, column])
            class_sources.append(
                {"dataset_index": dataset_index, "row": row, "column": column}
            )
        basis_banks.append(torch.stack(class_spectra))
        sources.append(class_sources)
    return torch.stack(basis_banks), sources


def next_run_directory(root, prefix):
    """Crea la prima directory libera con nome <prefix>-run_i, partendo da 1."""
    root.mkdir(parents=True, exist_ok=True)
    index = 1
    while (root / f"{prefix}-run_{index}").exists():
        index += 1
    path = root / f"{prefix}-run_{index}"
    path.mkdir()
    return path


def latest_checkpoint(directory, prefix, kind="latest"):
    """Trova il checkpoint del tipo richiesto modificato più recentemente."""
    paths = list(Path(directory).glob(f"{prefix}-run_*-{kind}.pt"))
    if not paths:
        return None
    return max(paths, key=lambda path: path.stat().st_mtime)


def save_generated_pair(image, ms_path, rgb_path, band_stats, config):
    """Salva un'immagine standardizzata come TIFF MS float32 e preview RGB.

    Il chiamante prepara i percorsi e le directory sotto outputs/. I valori
    negativi di riflettanza vengono portati a zero; l'RGB usa B4/B3/B2 e una
    scala di visualizzazione che non modifica il TIFF multispettrale.
    """
    reflectance = denormalize_bands(image.float().cpu(), band_stats).clamp_min(0)
    if not torch.isfinite(reflectance).all():
        raise FloatingPointError("Il sampling ha prodotto NaN o Inf")
    # I TIFF memorizzano [H,W,C], mentre PyTorch usa [C,H,W].
    ms_array = reflectance.permute(1, 2, 0).numpy().astype(np.float32)
    tifffile.imwrite(ms_path, ms_array)
    rgb = reflectance[[3, 2, 1]] / config["data"]["rgb_display_max"]
    save_image(rgb.clamp(0, 1), rgb_path)

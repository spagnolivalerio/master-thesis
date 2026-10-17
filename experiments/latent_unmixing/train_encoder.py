import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import EuroSATMSDataset  # noqa: E402
from experiments.latent_unmixing.model import LatentUnmixingModel  # noqa: E402
from experiments.latent_unmixing.utils import (  # noqa: E402
    BalancedBatchSampler,
    append_log,
    apply_band_normalization,
    denormalize_bands,
    get_random_state,
    latest_checkpoint,
    load_config,
    load_or_compute_band_stats,
    next_run_directory,
    sample_training_pixels,
    set_random_state,
    set_seed,
    split_indices,
)


@torch.inference_mode()
def validate(model, loader, device, use_fp16):
    model.eval()
    total, count = 0.0, 0
    for images, _ in tqdm(loader, desc="Validazione encoder", leave=False):
        images = images.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_fp16):
            reconstruction = model(images)["reconstruction"]
            loss = F.mse_loss(reconstruction, images)
        total += loss.item() * len(images)
        count += len(images)
    return total / count


def spectral_axis(config):
    wavelengths = np.asarray(config["data"]["wavelengths_nm"])
    order = np.argsort(wavelengths)
    return wavelengths[order], order


def save_basis_plot(model, band_stats, config, directory, epoch):
    mean = torch.tensor(band_stats["mean"])[None, :]
    std = torch.tensor(band_stats["standard_deviation"])[None, :]
    basis = model.latent_basis.detach().float().cpu() * std + mean
    wavelengths, order = spectral_axis(config)
    figure, axis = plt.subplots(figsize=(11, 6.5))
    for index, spectrum in enumerate(basis.numpy()):
        axis.plot(
            wavelengths,
            spectrum[order],
            "o-",
            linewidth=1.8,
            markersize=3,
            label=f"E{index + 1}",
        )
    axis.set_title(f"Evoluzione delle basi latenti — epoca {epoch}")
    axis.set_xlabel("Lunghezza d'onda (nm)")
    axis.set_ylabel("Riflettanza")
    axis.grid(alpha=0.25, linestyle="--")
    axis.legend(ncol=4)
    figure.tight_layout()
    directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(directory / f"epoch_{epoch:04d}.png", dpi=180)
    plt.close(figure)


@torch.inference_mode()
def save_rmc_monitor(model, loader, band_stats, classes, config, directory, epoch, device):
    model.eval()
    num_classes = len(classes)
    channels = config["data"]["channels"]
    real_sum = torch.zeros(num_classes, channels, dtype=torch.float64)
    reconstruction_sum = torch.zeros_like(real_sum)
    pixel_count = torch.zeros(num_classes, dtype=torch.float64)
    squared_error, value_count = 0.0, 0

    for images, labels in tqdm(loader, desc="Monitor RMC", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        reconstruction = model(images)["reconstruction"]
        real = denormalize_bands(images.float(), band_stats)
        reconstructed = denormalize_bands(reconstruction.float(), band_stats)
        error = reconstructed - real
        squared_error += error.square().sum().item()
        value_count += error.numel()
        pixels = images.shape[2] * images.shape[3]
        for class_index in labels.unique().tolist():
            selected = labels == class_index
            real_sum[class_index] += real[selected].sum((0, 2, 3)).double().cpu()
            reconstruction_sum[class_index] += (
                reconstructed[selected].sum((0, 2, 3)).double().cpu()
            )
            pixel_count[class_index] += int(selected.sum()) * pixels

    real_rmc = real_sum / pixel_count[:, None]
    reconstructed_rmc = reconstruction_sum / pixel_count[:, None]
    wavelengths, order = spectral_axis(config)
    maximum = max(real_rmc.max().item(), reconstructed_rmc.max().item())
    margin = max(maximum * 0.10, 0.01)
    figure, axes = plt.subplots(2, 5, figsize=(24, 9), sharex=True, sharey=True)
    axes.flat[0].set_ylim(0, maximum + margin)
    for class_index, axis in enumerate(axes.flat):
        axis.plot(
            wavelengths,
            real_rmc[class_index, order],
            "o-",
            color="black",
            label="Real",
        )
        axis.plot(
            wavelengths,
            reconstructed_rmc[class_index, order],
            "o-",
            color="#E15759",
            label="Reconstructed",
        )
        axis.set_title(classes[class_index])
        axis.grid(alpha=0.25, linestyle="--")
    axes.flat[0].legend()
    figure.supxlabel("Wavelength (nm)")
    figure.supylabel("Reflectance")
    figure.suptitle(f"Real and reconstructed RMCs — epoch {epoch}")
    figure.tight_layout()
    directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(directory / f"epoch_{epoch:04d}.png", dpi=180)
    plt.close(figure)
    return {
        "reflectance_rmse": (squared_error / value_count) ** 0.5,
        "rmc_mae": (reconstructed_rmc - real_rmc).abs().mean().item(),
    }


def main():
    config = load_config()
    settings = config["encoder_training"]
    seed = config["experiment"]["seed"]
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = device.type == "cuda" and settings["mixed_precision"] == "fp16"
    checkpoint_directory = Path(config["paths"]["encoder_checkpoints"])
    log_directory = Path(config["paths"]["encoder_logs"])
    preview_root = Path(config["paths"]["encoder_previews"])
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    log_directory.mkdir(parents=True, exist_ok=True)
    preview_root.mkdir(parents=True, exist_ok=True)

    resume_path = latest_checkpoint(
        checkpoint_directory, "latent_unmixing_encoder", "latest"
    )
    checkpoint = None
    if settings["resume"] and resume_path is not None:
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)

    dataset = EuroSATMSDataset(config["data"]["root"])
    if checkpoint is None:
        train_indices, validation_indices, test_indices = split_indices(dataset, config)
        band_stats = load_or_compute_band_stats(dataset, train_indices, config)
    else:
        train_indices = checkpoint["train_indices"]
        validation_indices = checkpoint["validation_indices"]
        test_indices = checkpoint["test_indices"]
        band_stats = checkpoint["band_stats"]
    apply_band_normalization(dataset, band_stats, config)

    if checkpoint is None:
        initial_basis, basis_sources = sample_training_pixels(
            dataset, train_indices, config["model"]["num_bases"], seed
        )
    else:
        initial_basis = checkpoint["model"]["latent_basis"]
        basis_sources = checkpoint["basis_sources"]
    model = LatentUnmixingModel(config, initial_basis).to(device)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.encoder.parameters(),
                "lr": settings["encoder_learning_rate"],
                "weight_decay": settings["weight_decay"],
            },
            {
                "params": [model.latent_basis],
                "lr": settings["basis_learning_rate"],
                "weight_decay": 0.0,
            },
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=settings["epochs"]
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    batch_sampler = BalancedBatchSampler(
        train_indices,
        dataset.targets,
        settings["batch_size"],
        config["data"]["num_classes"],
        seed,
    )
    train_loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=settings["num_workers"],
        pin_memory=settings["pin_memory"],
    )
    validation_loader = DataLoader(
        Subset(dataset, validation_indices),
        batch_size=settings["validation_batch_size"],
        shuffle=False,
        num_workers=settings["num_workers"],
        pin_memory=settings["pin_memory"],
    )
    monitor_loader = DataLoader(
        Subset(dataset, validation_indices),
        batch_size=config["monitor"]["batch_size"],
        shuffle=False,
        num_workers=settings["num_workers"],
        pin_memory=settings["pin_memory"],
    )

    start_epoch, best_validation = 1, float("inf")
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        set_random_state(checkpoint["random_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_validation = checkpoint["best_validation"]
        run_name = checkpoint["run_name"]
        run_directory = preview_root / run_name
        run_directory.mkdir(parents=True, exist_ok=True)
    else:
        run_directory = next_run_directory(preview_root, "latent_unmixing_encoder")
        run_name = run_directory.name

    latest_path = checkpoint_directory / f"{run_name}-latest.pt"
    best_path = checkpoint_directory / f"{run_name}-best.pt"
    log_path = log_directory / f"{run_name}.jsonl"
    append_log(
        log_path,
        {
            "event": "resume" if checkpoint is not None else "start",
            "epoch": start_epoch,
            "train_samples": len(train_indices),
            "validation_samples": len(validation_indices),
            "test_samples": len(test_indices),
            "config": config,
        },
    )

    for epoch in range(start_epoch, settings["epochs"] + 1):
        model.train()
        batch_sampler.set_epoch(epoch)
        total, count = 0.0, 0
        progress = tqdm(
            train_loader,
            desc=f"Encoder {epoch}/{settings['epochs']}",
            dynamic_ncols=True,
        )
        for images, _ in progress:
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_fp16):
                reconstruction = model(images)["reconstruction"]
                loss = F.mse_loss(reconstruction, images)
            if not torch.isfinite(loss):
                raise FloatingPointError("Loss encoder non finita")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * len(images)
            count += len(images)
            progress.set_postfix(mse=f"{total / count:.5f}")

        train_mse = total / count
        validation_mse = validate(
            model, validation_loader, device, use_fp16
        )
        monitor_metrics = save_rmc_monitor(
            model,
            monitor_loader,
            band_stats,
            dataset.classes,
            config,
            run_directory / "rmc",
            epoch,
            device,
        )
        save_basis_plot(
            model,
            band_stats,
            config,
            run_directory / "latent_basis",
            epoch,
        )
        scheduler.step()
        is_best = validation_mse < best_validation
        if is_best:
            best_validation = validation_mse
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "validation_mse": validation_mse,
                    "classes": dataset.classes,
                    "train_indices": train_indices,
                    "validation_indices": validation_indices,
                    "test_indices": test_indices,
                    "band_stats": band_stats,
                    "basis_sources": basis_sources,
                    "config": config,
                },
                best_path,
            )
        append_log(
            log_path,
            {
                "event": "epoch",
                "epoch": epoch,
                "train_mse": train_mse,
                "validation_mse": validation_mse,
                "best_validation_mse": best_validation,
                "is_best": is_best,
                **monitor_metrics,
            },
        )
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "run_name": run_name,
                "best_validation": best_validation,
                "classes": dataset.classes,
                "train_indices": train_indices,
                "validation_indices": validation_indices,
                "test_indices": test_indices,
                "band_stats": band_stats,
                "basis_sources": basis_sources,
                "random_state": get_random_state(),
                "config": config,
            },
            latest_path,
        )
        tqdm.write(
            f"Encoder epoca {epoch}: train {train_mse:.5f}, "
            f"validation {validation_mse:.5f}"
        )

    return best_path


if __name__ == "__main__":
    main()


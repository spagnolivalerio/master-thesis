import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import EuroSATMSDataset
from experiments.latent_unmixing_per_class_large.model import (
    ClassSpecificLatentUnmixingModel,
)
from experiments.latent_unmixing_per_class_large.utils import (
    BalancedBatchSampler,
    TrainingLogger,
    append_log,
    apply_band_normalization,
    denormalize_bands,
    get_random_state,
    latest_checkpoint,
    load_config,
    load_or_compute_band_stats,
    next_run_directory,
    sample_class_training_pixels,
    set_random_state,
    set_seed,
    split_indices,
)


@torch.inference_mode()
def validate(model, loader, device, use_fp16):
    model.eval()
    total, count = 0.0, 0
    progress = tqdm(loader, desc="Validazione encoder", leave=False)
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_fp16):
            reconstruction = model(images, labels)["reconstruction"]
            loss = F.mse_loss(reconstruction, images)
        total += loss.item() * len(images)
        count += len(images)
        progress.set_postfix(loss=f"{total / count:.5f}", mse=f"{total / count:.5f}")
    return total / count


def spectral_axis(config):
    wavelengths = np.asarray(config["data"]["wavelengths_nm"])
    order = np.argsort(wavelengths)
    return wavelengths[order], order


def save_bases(model, band_stats, classes, config, directory, epoch):
    mean = torch.tensor(band_stats["mean"])[None, None, :]
    std = torch.tensor(band_stats["standard_deviation"])[None, None, :]
    bases = model.latent_bases.detach().float().cpu() * std + mean
    wavelengths, order = spectral_axis(config)
    figure, axes = plt.subplots(2, 5, figsize=(24, 10), sharex=True, sharey=False)
    for class_index, axis in enumerate(axes.flat):
        for basis_index, spectrum in enumerate(bases[class_index].numpy()):
            axis.plot(
                wavelengths,
                spectrum[order],
                "o-",
                linewidth=1.3,
                markersize=2.5,
                label=f"E{basis_index + 1}",
            )
        axis.set_title(classes[class_index])
        axis.grid(alpha=0.25, linestyle="--")
    axes.flat[0].legend(ncol=2, fontsize=8)
    figure.supxlabel("Lunghezza d'onda (nm)")
    figure.supylabel("Riflettanza")
    figure.suptitle(f"Basi latenti specifiche per classe — epoca {epoch}")
    figure.tight_layout()
    directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(directory / f"epoch_{epoch:04d}.png", dpi=180)
    plt.close(figure)


@torch.inference_mode()
def save_rmc(model, loader, band_stats, classes, config, directory, epoch, device):
    model.eval()
    num_classes = len(classes)
    channels = config["data"]["channels"]
    real_sum = torch.zeros(num_classes, channels, dtype=torch.float64)
    reconstructed_sum = torch.zeros_like(real_sum)
    pixel_count = torch.zeros(num_classes, dtype=torch.float64)
    squared_error, value_count = 0.0, 0
    for images, labels in tqdm(loader, desc="Monitor RMC", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        reconstruction = model(images, labels)["reconstruction"]
        real = denormalize_bands(images.float(), band_stats)
        reconstructed = denormalize_bands(reconstruction.float(), band_stats)
        squared_error += (reconstructed - real).square().sum().item()
        value_count += real.numel()
        pixels = images.shape[2] * images.shape[3]
        for class_index in labels.unique().tolist():
            selected = labels == class_index
            real_sum[class_index] += real[selected].sum((0, 2, 3)).double().cpu()
            reconstructed_sum[class_index] += (
                reconstructed[selected].sum((0, 2, 3)).double().cpu()
            )
            pixel_count[class_index] += int(selected.sum()) * pixels

    real_rmc = real_sum / pixel_count[:, None]
    reconstructed_rmc = reconstructed_sum / pixel_count[:, None]
    wavelengths, order = spectral_axis(config)
    maximum = max(real_rmc.max().item(), reconstructed_rmc.max().item())
    margin = max(maximum * config["monitor"]["axis_margin"], 0.01)
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

    prefix = "latent_unmixing_per_class_large_encoder"
    resume_path = latest_checkpoint(checkpoint_directory, prefix, "latest")
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
        initial_bases, basis_sources = sample_class_training_pixels(
            dataset, train_indices, config
        )
    else:
        initial_bases = checkpoint["model"]["latent_bases"]
        basis_sources = checkpoint["basis_sources"]
    model = ClassSpecificLatentUnmixingModel(config, initial_bases).to(device)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.encoders.parameters(),
                "lr": settings["encoder_learning_rate"],
                "weight_decay": settings["weight_decay"],
            },
            {
                "params": [model.latent_bases],
                "lr": settings["basis_learning_rate"],
                "weight_decay": 0.0,
            },
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=settings["epochs"]
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    sampler = BalancedBatchSampler(
        train_indices,
        dataset.targets,
        settings["batch_size"],
        config["data"]["num_classes"],
        seed,
    )
    train_loader = DataLoader(
        dataset,
        batch_sampler=sampler,
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
        run_directory = next_run_directory(preview_root, prefix)
        run_name = run_directory.name
    latest_path = checkpoint_directory / f"{run_name}-latest.pt"
    best_path = checkpoint_directory / f"{run_name}-best.pt"
    log_path = log_directory / f"{run_name}.jsonl"
    logger = TrainingLogger(log_path, optimizer, device)
    global_step = checkpoint["global_step"] if checkpoint is not None else 0
    append_log(
        log_path,
        {
            "event": "resume" if checkpoint is not None else "start",
            "epoch": start_epoch,
            "config": config,
        },
    )

    # Start training
    for epoch in range(start_epoch, settings["epochs"] + 1):
        model.train()
        sampler.set_epoch(epoch)
        logger.start_epoch(epoch, global_step)
        total, count = 0.0, 0
        progress = tqdm(
            train_loader,
            desc=f"Encoder per class {epoch}/{settings['epochs']}",
            dynamic_ncols=True,
        )
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_fp16):
                reconstruction = model(images, labels)["reconstruction"]
                loss = F.mse_loss(reconstruction, images)
            if not torch.isfinite(loss):
                raise FloatingPointError("Loss encoder non finita")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), settings["max_grad_norm"]
            )
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * len(images)
            count += len(images)
            means = logger.update(
                {"loss": loss, "mse_loss": loss}, len(images), gradient_norm
            )
            progress.set_postfix(
                loss=f"{means['loss']:.5f}", mse=f"{means['mse_loss']:.5f}"
            )

        train_mse = total / count
        global_step = logger.global_step
        training_metrics = logger.summary()
        validation_mse = validate(model, validation_loader, device, use_fp16)
        monitor_metrics = save_rmc(
            model,
            validation_loader,
            band_stats,
            dataset.classes,
            config,
            run_directory / "rmc",
            epoch,
            device,
        )
        save_bases(
            model,
            band_stats,
            dataset.classes,
            config,
            run_directory / "latent_bases",
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
                "train_loss": train_mse,
                "validation_loss": validation_mse,
                "best_validation_mse": best_validation,
                "is_best": is_best,
                **training_metrics,
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
                "global_step": global_step,
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


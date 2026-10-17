import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import EuroSATMSDataset  # noqa: E402
from experiments.latent_unmixing.model import (  # noqa: E402
    LatentUnmixingModel,
    create_ddpm_model,
    create_sampling_scheduler,
    create_training_scheduler,
    generate_ILR,
)
from experiments.latent_unmixing.utils import (  # noqa: E402
    BalancedBatchSampler,
    append_log,
    apply_band_normalization,
    compute_ILR_statistics,
    denormalize_bands,
    from_ILR,
    get_random_state,
    latest_checkpoint,
    load_config,
    next_run_directory,
    set_random_state,
    set_seed,
    to_ILR,
)


@torch.inference_mode()
def validate(ddpm, encoder, loader, scheduler, ilr_stats, config, device, use_fp16):
    ddpm.eval()
    total, count = 0.0, 0
    generator = torch.Generator(device=device).manual_seed(
        config["experiment"]["seed"]
    )
    for images, labels in tqdm(loader, desc="Validazione DDPM", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        _, abundances = encoder.encode(images)
        clean = to_ILR(abundances, ilr_stats)
        noise = torch.randn(clean.shape, generator=generator, device=device)
        timesteps = torch.randint(
            0,
            scheduler.config.num_train_timesteps,
            (len(images),),
            generator=generator,
            device=device,
        )
        noisy = scheduler.add_noise(clean, noise, timesteps)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_fp16):
            prediction = ddpm(noisy, timesteps, class_labels=labels).sample
            loss = F.mse_loss(prediction, noise)
        total += loss.item() * len(images)
        count += len(images)
    return total / count


@torch.inference_mode()
def save_preview(ddpm, encoder, scheduler, ilr_stats, band_stats, config, path, device):
    labels = torch.arange(config["data"]["num_classes"], device=device)
    generator = torch.Generator(device=device).manual_seed(
        config["experiment"]["seed"]
    )
    standardized_ilr = generate_ILR(
        ddpm,
        scheduler,
        labels,
        config,
        config["ddpm_training"]["preview_inference_steps"],
        generator,
    )
    abundances = from_ILR(standardized_ilr, ilr_stats)
    reconstructed = encoder.decode(abundances)
    reflectance = denormalize_bands(reconstructed.float(), band_stats).clamp_min(0)
    rgb = reflectance[:, [3, 2, 1]] / config["data"]["rgb_display_max"]
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(rgb.clamp(0, 1), path, nrow=5)


def load_encoder_state(config):
    path = latest_checkpoint(
        config["paths"]["encoder_checkpoints"],
        "latent_unmixing_encoder",
        "best",
    )
    if path is None:
        raise FileNotFoundError(
            "Checkpoint best dell'encoder assente: eseguire prima train_encoder"
        )
    return path, torch.load(path, map_location="cpu", weights_only=False)


def main():
    config = load_config()
    settings = config["ddpm_training"]
    seed = config["experiment"]["seed"]
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = device.type == "cuda" and settings["mixed_precision"] == "fp16"
    checkpoint_directory = Path(config["paths"]["ddpm_checkpoints"])
    log_directory = Path(config["paths"]["ddpm_logs"])
    preview_root = Path(config["paths"]["ddpm_previews"])
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    log_directory.mkdir(parents=True, exist_ok=True)
    preview_root.mkdir(parents=True, exist_ok=True)

    resume_path = latest_checkpoint(
        checkpoint_directory, "latent_unmixing_ddpm", "latest"
    )
    checkpoint = None
    if settings["resume"] and resume_path is not None:
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)

    dataset = EuroSATMSDataset(config["data"]["root"])
    if checkpoint is None:
        encoder_path, encoder_checkpoint = load_encoder_state(config)
        encoder_state = encoder_checkpoint["model"]
        train_indices = encoder_checkpoint["train_indices"]
        validation_indices = encoder_checkpoint["validation_indices"]
        test_indices = encoder_checkpoint["test_indices"]
        band_stats = encoder_checkpoint["band_stats"]
        classes = encoder_checkpoint["classes"]
        source_encoder_checkpoint = str(encoder_path.resolve())
    else:
        encoder_state = checkpoint["encoder_model"]
        train_indices = checkpoint["train_indices"]
        validation_indices = checkpoint["validation_indices"]
        test_indices = checkpoint["test_indices"]
        band_stats = checkpoint["band_stats"]
        classes = checkpoint["classes"]
        source_encoder_checkpoint = checkpoint["source_encoder_checkpoint"]
    if dataset.classes != classes:
        raise ValueError("Le classi del dataset non corrispondono all'encoder")
    apply_band_normalization(dataset, band_stats, config)

    encoder = LatentUnmixingModel(config, encoder_state["latent_basis"])
    encoder.load_state_dict(encoder_state)
    encoder.to(device).eval()
    encoder.requires_grad_(False)

    statistics_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=settings["validation_batch_size"],
        shuffle=False,
        num_workers=settings["num_workers"],
        pin_memory=settings["pin_memory"],
    )
    ilr_stats = (
        checkpoint["ilr_stats"]
        if checkpoint is not None
        else compute_ILR_statistics(encoder, statistics_loader, config, device)
    )

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

    ddpm = create_ddpm_model(config).to(device)
    if settings["gradient_checkpointing"]:
        ddpm.enable_gradient_checkpointing()
    optimizer = torch.optim.AdamW(ddpm.parameters(), lr=settings["learning_rate"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    training_scheduler = create_training_scheduler(config)
    sampling_scheduler = create_sampling_scheduler(config)
    accumulation = settings["gradient_accumulation_steps"]
    updates_per_epoch = math.ceil(len(train_loader) / accumulation)
    learning_rate_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=settings["warmup_steps"],
        num_training_steps=updates_per_epoch * settings["epochs"],
    )
    ema = EMAModel(
        ddpm.parameters(), decay=settings["ema_decay"], use_ema_warmup=True
    )
    ema.to(device)

    start_epoch, best_validation = 1, float("inf")
    if checkpoint is not None:
        ddpm.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        learning_rate_scheduler.load_state_dict(checkpoint["scheduler"])
        ema.load_state_dict(checkpoint["ema"])
        ema.to(device)
        set_random_state(checkpoint["random_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_validation = checkpoint["best_validation"]
        run_name = checkpoint["run_name"]
        run_directory = preview_root / run_name
        run_directory.mkdir(parents=True, exist_ok=True)
    else:
        run_directory = next_run_directory(preview_root, "latent_unmixing_ddpm")
        run_name = run_directory.name

    latest_path = checkpoint_directory / f"{run_name}-latest.pt"
    best_path = checkpoint_directory / f"{run_name}-best.pt"
    log_path = log_directory / f"{run_name}.jsonl"
    append_log(
        log_path,
        {
            "event": "resume" if checkpoint is not None else "start",
            "epoch": start_epoch,
            "source_encoder_checkpoint": source_encoder_checkpoint,
            "ilr_stats": ilr_stats,
            "config": config,
        },
    )

    for epoch in range(start_epoch, settings["epochs"] + 1):
        ddpm.train()
        batch_sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        total, count = 0.0, 0
        progress = tqdm(
            train_loader,
            desc=f"DDPM ILR {epoch}/{settings['epochs']}",
            dynamic_ncols=True,
        )
        for step, (images, labels) in enumerate(progress, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.no_grad():
                _, abundances = encoder.encode(images)
                clean = to_ILR(abundances, ilr_stats)
            noise = torch.randn_like(clean)
            timesteps = torch.randint(
                0,
                training_scheduler.config.num_train_timesteps,
                (len(images),),
                device=device,
            )
            noisy = training_scheduler.add_noise(clean, noise, timesteps)
            model_labels = labels.clone()
            dropped = torch.rand(len(labels), device=device) < config["cfg"][
                "condition_dropout"
            ]
            model_labels[dropped] = config["data"]["num_classes"]
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_fp16):
                prediction = ddpm(
                    noisy, timesteps, class_labels=model_labels
                ).sample
                mse = F.mse_loss(prediction, noise)
            if not torch.isfinite(mse):
                raise FloatingPointError("Loss DDPM non finita")
            group_start = ((step - 1) // accumulation) * accumulation
            group_size = min(accumulation, len(train_loader) - group_start)
            scaler.scale(mse / group_size).backward()
            update = step % accumulation == 0 or step == len(train_loader)
            if update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    ddpm.parameters(), settings["max_grad_norm"]
                )
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if not use_fp16 or scaler.get_scale() >= scale_before:
                    learning_rate_scheduler.step()
                    ema.step(ddpm.parameters())
                optimizer.zero_grad(set_to_none=True)
            total += mse.item() * len(images)
            count += len(images)
            progress.set_postfix(mse=f"{total / count:.5f}")

        train_mse = total / count
        ema.store(ddpm.parameters())
        ema.copy_to(ddpm.parameters())
        validation_mse = validate(
            ddpm,
            encoder,
            validation_loader,
            training_scheduler,
            ilr_stats,
            config,
            device,
            use_fp16,
        )
        save_preview(
            ddpm,
            encoder,
            sampling_scheduler,
            ilr_stats,
            band_stats,
            config,
            run_directory / f"epoch_{epoch:04d}.png",
            device,
        )
        is_best = validation_mse < best_validation
        if is_best:
            best_validation = validation_mse
            torch.save(
                {
                    "model": ddpm.state_dict(),
                    "encoder_model": encoder.state_dict(),
                    "epoch": epoch,
                    "validation_mse": validation_mse,
                    "classes": classes,
                    "train_indices": train_indices,
                    "validation_indices": validation_indices,
                    "test_indices": test_indices,
                    "band_stats": band_stats,
                    "ilr_stats": ilr_stats,
                    "weights": "ema",
                    "source_encoder_checkpoint": source_encoder_checkpoint,
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
                "learning_rate": learning_rate_scheduler.get_last_lr()[0],
            },
        )
        ema.restore(ddpm.parameters())
        torch.save(
            {
                "model": ddpm.state_dict(),
                "encoder_model": encoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "scheduler": learning_rate_scheduler.state_dict(),
                "ema": ema.state_dict(),
                "epoch": epoch,
                "run_name": run_name,
                "best_validation": best_validation,
                "classes": classes,
                "train_indices": train_indices,
                "validation_indices": validation_indices,
                "test_indices": test_indices,
                "band_stats": band_stats,
                "ilr_stats": ilr_stats,
                "source_encoder_checkpoint": source_encoder_checkpoint,
                "random_state": get_random_state(),
                "config": config,
            },
            latest_path,
        )
        tqdm.write(
            f"DDPM epoca {epoch}: train {train_mse:.5f}, "
            f"validation {validation_mse:.5f}"
        )

    return best_path


if __name__ == "__main__":
    main()


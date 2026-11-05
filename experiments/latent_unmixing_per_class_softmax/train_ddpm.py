"""Training del DDPM sui K logits degli encoder per classe."""

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

from dataset import EuroSATMSDataset
from experiments.latent_unmixing_per_class_softmax.model import (
    ClassSpecificLatentUnmixingModel,
    create_ddpm_model,
    create_sampling_scheduler,
    create_training_scheduler,
    generate_logits,
)
from experiments.latent_unmixing_per_class_softmax.utils import (
    BalancedBatchSampler,
    TrainingLogger,
    append_log,
    apply_band_normalization,
    compute_logit_statistics,
    denormalize_bands,
    from_latents,
    get_random_state,
    latest_checkpoint,
    load_config,
    next_run_directory,
    set_random_state,
    set_seed,
    to_latents,
)


@torch.inference_mode()
def validate(ddpm, unmixing, loader, scheduler, stats, config, device, use_fp16):
    ddpm.eval()
    total, count = 0.0, 0
    generator = torch.Generator(device=device).manual_seed(
        config["experiment"]["seed"]
    )
    progress = tqdm(loader, desc="Validazione DDPM", leave=False)
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits, _ = unmixing.encode(images, labels)
        clean = to_latents(logits, labels, stats)
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
        progress.set_postfix(loss=f"{total / count:.5f}", mse=f"{total / count:.5f}")
    return total / count


@torch.inference_mode()
def save_preview(ddpm, unmixing, scheduler, stats, band_stats, config, path, device):
    labels = torch.arange(config["data"]["num_classes"], device=device)
    generator = torch.Generator(device=device).manual_seed(
        config["experiment"]["seed"]
    )
    latents = generate_logits(
        ddpm,
        scheduler,
        labels,
        config,
        config["ddpm_training"]["preview_inference_steps"],
        generator,
    )
    abundances = from_latents(latents, labels, stats)
    images = unmixing.decode(abundances, labels)
    reflectance = denormalize_bands(images.float(), band_stats).clamp_min(0)
    rgb = reflectance[:, [3, 2, 1]] / config["data"]["rgb_display_max"]
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(rgb.clamp(0, 1), path, nrow=5)
    return {
        "preview_reflectance_mean": reflectance.mean().item(),
        "preview_reflectance_std": reflectance.std().item(),
        "preview_reflectance_min": reflectance.min().item(),
        "preview_reflectance_max": reflectance.max().item(),
    }


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

    encoder_prefix = "latent_unmixing_per_class_softmax_encoder"
    ddpm_prefix = "latent_unmixing_per_class_softmax_ddpm"
    resume_path = latest_checkpoint(checkpoint_directory, ddpm_prefix, "latest")
    checkpoint = None
    if settings["resume"] and resume_path is not None:
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)

    dataset = EuroSATMSDataset(config["data"]["root"])
    if checkpoint is None:
        encoder_path = latest_checkpoint(
            config["paths"]["encoder_checkpoints"], encoder_prefix, "best"
        )
        if encoder_path is None:
            raise FileNotFoundError("Manca il checkpoint best degli encoder per classe")
        source = torch.load(encoder_path, map_location="cpu", weights_only=False)
        unmixing_state = source["model"]
        train_indices = source["train_indices"]
        validation_indices = source["validation_indices"]
        test_indices = source["test_indices"]
        band_stats = source["band_stats"]
        classes = source["classes"]
        source_encoder_checkpoint = str(encoder_path.resolve())
    else:
        unmixing_state = checkpoint["unmixing_model"]
        train_indices = checkpoint["train_indices"]
        validation_indices = checkpoint["validation_indices"]
        test_indices = checkpoint["test_indices"]
        band_stats = checkpoint["band_stats"]
        classes = checkpoint["classes"]
        source_encoder_checkpoint = checkpoint["source_encoder_checkpoint"]
    if dataset.classes != classes:
        raise ValueError("Le classi non corrispondono al checkpoint degli encoder")
    apply_band_normalization(dataset, band_stats, config)

    unmixing = ClassSpecificLatentUnmixingModel(
        config, unmixing_state["latent_bases"]
    )
    unmixing.load_state_dict(unmixing_state)
    unmixing.to(device).eval().requires_grad_(False)

    statistics_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=settings["validation_batch_size"],
        shuffle=False,
        num_workers=settings["num_workers"],
        pin_memory=settings["pin_memory"],
    )
    logit_stats = (
        checkpoint["logit_stats"]
        if checkpoint is not None
        else compute_logit_statistics(unmixing, statistics_loader, config, device)
    )

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

    ddpm = create_ddpm_model(config).to(device)
    if settings["gradient_checkpointing"]:
        ddpm.enable_gradient_checkpointing()
    optimizer = torch.optim.AdamW(ddpm.parameters(), lr=settings["learning_rate"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    training_scheduler = create_training_scheduler(config)
    sampling_scheduler = create_sampling_scheduler(config)
    accumulation = settings["gradient_accumulation_steps"]
    updates_per_epoch = math.ceil(len(train_loader) / accumulation)
    lr_scheduler = get_cosine_schedule_with_warmup(
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
        lr_scheduler.load_state_dict(checkpoint["scheduler"])
        ema.load_state_dict(checkpoint["ema"])
        ema.to(device)
        set_random_state(checkpoint["random_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_validation = checkpoint["best_validation"]
        run_name = checkpoint["run_name"]
        run_directory = preview_root / run_name
        run_directory.mkdir(parents=True, exist_ok=True)
    else:
        run_directory = next_run_directory(preview_root, ddpm_prefix)
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
            "source_encoder_checkpoint": source_encoder_checkpoint,
            "logit_stats": logit_stats,
            "config": config,
        },
    )

    for epoch in range(start_epoch, settings["epochs"] + 1):
        ddpm.train()
        sampler.set_epoch(epoch)
        logger.start_epoch(epoch, global_step)
        optimizer.zero_grad(set_to_none=True)
        total, count = 0.0, 0
        progress = tqdm(
            train_loader,
            desc=f"DDPM per class {epoch}/{settings['epochs']}",
            dynamic_ncols=True,
        )
        for step, (images, labels) in enumerate(progress, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.no_grad():
                logits, _ = unmixing.encode(images, labels)
                clean = to_latents(logits, labels, logit_stats)
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
            gradient_norm = None
            if step % accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    ddpm.parameters(), settings["max_grad_norm"]
                )
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if not use_fp16 or scaler.get_scale() >= scale_before:
                    lr_scheduler.step()
                    ema.step(ddpm.parameters())
                optimizer.zero_grad(set_to_none=True)
            total += mse.item() * len(images)
            count += len(images)
            means = logger.update(
                {"loss": mse, "mse_loss": mse}, len(images), gradient_norm
            )
            progress.set_postfix(
                loss=f"{means['loss']:.5f}", mse=f"{means['mse_loss']:.5f}"
            )

        train_mse = total / count
        global_step = logger.global_step
        training_metrics = logger.summary()
        ema.store(ddpm.parameters())
        ema.copy_to(ddpm.parameters())
        validation_mse = validate(
            ddpm,
            unmixing,
            validation_loader,
            training_scheduler,
            logit_stats,
            config,
            device,
            use_fp16,
        )
        preview_metrics = save_preview(
            ddpm,
            unmixing,
            sampling_scheduler,
            logit_stats,
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
                    "unmixing_model": unmixing.state_dict(),
                    "epoch": epoch,
                    "validation_mse": validation_mse,
                    "classes": classes,
                    "train_indices": train_indices,
                    "validation_indices": validation_indices,
                    "test_indices": test_indices,
                    "band_stats": band_stats,
                    "logit_stats": logit_stats,
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
                "train_loss": train_mse,
                "validation_loss": validation_mse,
                "best_validation_mse": best_validation,
                "is_best": is_best,
                **training_metrics,
                **preview_metrics,
            },
        )
        ema.restore(ddpm.parameters())
        torch.save(
            {
                "model": ddpm.state_dict(),
                "unmixing_model": unmixing.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "scheduler": lr_scheduler.state_dict(),
                "ema": ema.state_dict(),
                "epoch": epoch,
                "run_name": run_name,
                "global_step": global_step,
                "best_validation": best_validation,
                "classes": classes,
                "train_indices": train_indices,
                "validation_indices": validation_indices,
                "test_indices": test_indices,
                "band_stats": band_stats,
                "logit_stats": logit_stats,
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


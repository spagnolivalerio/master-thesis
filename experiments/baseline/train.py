import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import EuroSATMSDataset  
from experiments.baseline.diffusion import (  
    create_sampling_scheduler,
    create_training_scheduler,
    generate,
)
from experiments.baseline.model import create_model  
from experiments.baseline.utils import (  
    BalancedBatchSampler,
    append_log,
    apply_normalization,
    get_random_state,
    latest_checkpoint,
    load_config,
    load_or_compute_band_stats,
    next_run_directory,
    save_preview,
    set_random_state,
    set_seed,
    split_indices,
)


@torch.inference_mode()
def compute_mse(model, loader, scheduler, config, device, use_fp16):
    model.eval()
    total = 0.0
    count = 0
    generator = torch.Generator(device=device).manual_seed(
        config["experiment"]["seed"]
    )
    for images, labels in tqdm(loader, desc="Validazione", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        noise = torch.randn(images.shape, generator=generator, device=device)
        timesteps = torch.randint(
            0,
            scheduler.config.num_train_timesteps,
            (len(images),),
            generator=generator,
            device=device,
        )
        noisy = scheduler.add_noise(images, noise, timesteps)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_fp16):
            prediction = model(noisy, timesteps, class_labels=labels).sample
            loss = F.mse_loss(prediction, noise)
        total += loss.item() * len(images)
        count += len(images)
    return total / count


def main():
    config = load_config()
    seed = config["experiment"]["seed"]
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = device.type == "cuda" and config["training"]["mixed_precision"] == "fp16"
    print(f"Device: {device}")

    checkpoint_directory = Path(config["paths"]["checkpoints"])
    log_directory = Path(config["paths"]["logs"])
    preview_root = Path(config["paths"]["previews"])
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    log_directory.mkdir(parents=True, exist_ok=True)
    preview_root.mkdir(parents=True, exist_ok=True)

    resume_path = latest_checkpoint(checkpoint_directory)
    checkpoint = None
    if config["training"]["resume"] and resume_path is not None:
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)

    dataset = EuroSATMSDataset(config["data"]["root"])

    if checkpoint is None:
        train_indices, validation_indices, test_indices = split_indices(dataset, config)
    else:
        train_indices = checkpoint["train_indices"]
        validation_indices = checkpoint["validation_indices"]
        test_indices = checkpoint["test_indices"]

    stats = (
        checkpoint["band_stats"]
        if checkpoint is not None
        else load_or_compute_band_stats(dataset, train_indices, config)
    )
    apply_normalization(dataset, stats, config)

    batch_sampler = BalancedBatchSampler(
        train_indices,
        dataset.targets,
        config["training"]["batch_size"],
        config["data"]["num_classes"],
        seed,
    )
    train_loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=config["training"]["num_workers"],
        pin_memory=config["training"]["pin_memory"],
    )
    validation_loader = DataLoader(
        Subset(dataset, validation_indices),
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=config["training"]["pin_memory"],
    )
    test_loader = DataLoader(
        Subset(dataset, test_indices),
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=config["training"]["pin_memory"],
    )

    model = create_model(config).to(device)
    if config["training"]["gradient_checkpointing"]:
        model.enable_gradient_checkpointing()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["training"]["learning_rate"]
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    training_scheduler = create_training_scheduler(config)
    sampling_scheduler = create_sampling_scheduler(config)
    accumulation = config["training"]["gradient_accumulation_steps"]
    updates_per_epoch = math.ceil(len(train_loader) / accumulation)
    learning_rate_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["training"]["warmup_steps"],
        num_training_steps=updates_per_epoch * config["training"]["epochs"],
    )
    ema = EMAModel(
        model.parameters(),
        decay=config["training"]["ema_decay"],
        use_ema_warmup=True,
    )
    ema.to(device)

    start_epoch = 1
    best_validation_mse = float("inf")
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        learning_rate_scheduler.load_state_dict(checkpoint["learning_rate_scheduler"])
        ema.load_state_dict(checkpoint["ema"])
        ema.to(device)
        set_random_state(checkpoint["random_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_validation_mse = checkpoint["best_validation_mse"]
        run_name = checkpoint["run_name"]
        preview_directory = preview_root / run_name
        preview_directory.mkdir(parents=True, exist_ok=True)
        print(f"Ripresa di {run_name} dall'epoca {start_epoch}")
    else:
        preview_directory = next_run_directory(
            preview_root, config["experiment"]["name"]
        )
        run_name = preview_directory.name
        print(f"Nuova run: {run_name}")

    latest_path = checkpoint_directory / f"{run_name}-latest.pt"
    best_path = checkpoint_directory / f"{run_name}-best.pt"
    log_path = log_directory / f"{run_name}.jsonl"
    append_log(
        log_path,
        {
            "event": "resume" if checkpoint is not None else "start",
            "epoch": start_epoch,
            "config": config,
            "train_samples": len(train_indices),
            "validation_samples": len(validation_indices),
            "test_samples": len(test_indices),
        },
    )

    # --------- START TRAINING PHASE ---------#
    for epoch in range(start_epoch, config["training"]["epochs"] + 1):
        model.train()
        batch_sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        total_mse = 0.0
        sample_count = 0
        progress = tqdm(
            train_loader,
            desc=f"Epoca {epoch}/{config['training']['epochs']}",
            dynamic_ncols=True,
        )

        for step, (images, labels) in enumerate(progress, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            noise = torch.randn_like(images)
            timesteps = torch.randint(
                0,
                training_scheduler.config.num_train_timesteps,
                (len(images),),
                device=device,
            )
            noisy = training_scheduler.add_noise(images, noise, timesteps)
            model_labels = labels.clone()

            # cfg condition
            if config["cfg"].get("enabled", True):

                # Boolean mask 
                dropped = torch.rand(len(labels), device=device) < config["cfg"]["condition_dropout"]
                # Applying the boolean mask on the batch labels
                model_labels[dropped] = config["data"]["num_classes"]

            # Training step
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_fp16):
                prediction = model(
                    noisy,
                    timesteps,
                    class_labels=model_labels,
                ).sample
                mse = F.mse_loss(prediction, noise)
                loss = mse / accumulation
            scaler.scale(loss).backward()
            # --------- END TRAINING PHASE ---------#

            # Audite phase
            update = step % accumulation == 0 or step == len(train_loader)
            if update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config["training"]["max_grad_norm"]
                )
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if not use_fp16 or scaler.get_scale() >= scale_before:
                    ema.step(model.parameters())
                    learning_rate_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            total_mse += mse.item() * len(images)
            sample_count += len(images)
            progress.set_postfix(mse=f"{total_mse / sample_count:.4f}")

        train_mse = total_mse / sample_count
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        validation_mse = compute_mse(
            model,
            validation_loader,
            training_scheduler,
            config,
            device,
            use_fp16,
        )
        append_log(
            log_path,
            {
                "event": "epoch",
                "epoch": epoch,
                "train_mse": train_mse,
                "validation_mse": validation_mse,
                "learning_rate": learning_rate_scheduler.get_last_lr()[0],
            },
        )
        tqdm.write(
            f"Epoca {epoch}: MSE {train_mse:.4f}, val MSE {validation_mse:.4f}"
        )

        if validation_mse < best_validation_mse:
            best_validation_mse = validation_mse
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "validation_mse": validation_mse,
                    "classes": dataset.classes,
                    "class_to_idx": dataset.class_to_idx,
                    "band_stats": stats,
                    "weights": "ema",
                    "config": config,
                    "test_indices": test_indices,
                },
                best_path,
            )

        labels = torch.arange(config["data"]["num_classes"], device=device)
        generator = torch.Generator(device=device).manual_seed(seed)
        preview = generate(
            model,
            sampling_scheduler,
            labels,
            config,
            config["training"]["preview_inference_steps"],
            generator,
        )
        save_preview(
            preview,
            preview_directory / f"epoch_{epoch:04d}.png",
            stats,
            config,
        )
        ema.restore(model.parameters())

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "learning_rate_scheduler": learning_rate_scheduler.state_dict(),
                "ema": ema.state_dict(),
                "epoch": epoch,
                "run_name": run_name,
                "train_indices": train_indices,
                "validation_indices": validation_indices,
                "test_indices": test_indices,
                "best_validation_mse": best_validation_mse,
                "band_stats": stats,
                "random_state": get_random_state(),
                "config": config,
            },
            latest_path,
        )

    best = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best["model"])
    test_mse = compute_mse(
        model, test_loader, training_scheduler, config, device, use_fp16
    )
    append_log(log_path, {"event": "test", "test_mse": test_mse})
    print(f"Test MSE: {test_mse:.4f}")


if __name__ == "__main__":
    main()


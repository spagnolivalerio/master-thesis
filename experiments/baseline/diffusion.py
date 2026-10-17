import torch
from diffusers import DDIMScheduler, DDPMScheduler


def create_training_scheduler(config):
    diffusion = config["diffusion"]
    return DDPMScheduler(
        num_train_timesteps=diffusion["timesteps"],
        beta_schedule=diffusion["beta_schedule"],
        prediction_type=diffusion["prediction_type"],
        clip_sample=diffusion["clip_sample"],
    )


def create_sampling_scheduler(config):
    return DDIMScheduler.from_config(create_training_scheduler(config).config)


@torch.inference_mode()
def generate(model, scheduler, labels, config, inference_steps, generator=None):
    data = config["data"]
    images = torch.randn(
        (len(labels), data["channels"], data["image_size"], data["image_size"]),
        generator=generator,
        device=labels.device,
    )
    scheduler.set_timesteps(inference_steps, device=labels.device)
    use_cfg = config["cfg"].get("enabled", True)

    for timestep in scheduler.timesteps:
        if use_cfg:
            null_labels = torch.full_like(labels, data["num_classes"])
            model_input = torch.cat((images, images))
            label_input = torch.cat((null_labels, labels))
            prediction = model(
                model_input, timestep, class_labels=label_input
            ).sample
            unconditional, conditional = prediction.chunk(2)
            prediction = unconditional + config["cfg"]["guidance_scale"] * (
                conditional - unconditional
            )
        else:
            prediction = model(images, timestep).sample
        images = scheduler.step(prediction, timestep, images).prev_sample

    return images


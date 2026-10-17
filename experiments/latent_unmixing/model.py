"""Encoder, decoder lineare e DDPM dell'esperimento latent unmixing."""

import torch
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels, norm_num_groups):
        super().__init__()
        self.norm_1 = nn.GroupNorm(norm_num_groups, channels)
        self.conv_1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, padding_mode="reflect"
        )
        self.norm_2 = nn.GroupNorm(norm_num_groups, channels)
        self.conv_2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, padding_mode="reflect"
        )
        self.activation = nn.SiLU()

    def forward(self, inputs):
        hidden = self.conv_1(self.activation(self.norm_1(inputs)))
        hidden = self.conv_2(self.activation(self.norm_2(hidden)))
        return inputs + hidden


class AbundanceEncoder(nn.Module):
    """Predice K abundance positive, con somma unitaria per ogni pixel."""

    def __init__(self, config):
        super().__init__()
        data, settings = config["data"], config["model"]
        hidden = settings["hidden_channels"]
        self.temperature = settings["abundance_temperature"]
        self.input_projection = nn.Conv2d(
            data["channels"], hidden, kernel_size=3, padding=1, padding_mode="reflect"
        )
        self.blocks = nn.Sequential(
            *[
                ResidualBlock(hidden, settings["norm_num_groups"])
                for _ in range(settings["num_res_blocks"])
            ]
        )
        self.output_norm = nn.GroupNorm(settings["norm_num_groups"], hidden)
        self.output = nn.Conv2d(hidden, settings["num_bases"], kernel_size=1)
        self.activation = nn.SiLU()

    def forward(self, images):
        hidden = self.input_projection(images)
        hidden = self.blocks(hidden)
        logits = self.output(self.activation(self.output_norm(hidden)))
        abundances = torch.softmax(logits.float() / self.temperature, dim=1)
        return logits, abundances


class LatentUnmixingModel(nn.Module):
    """Encoder delle abundance con K basi globali e decoder strettamente lineare."""

    def __init__(self, config, initial_basis):
        super().__init__()
        expected = (config["model"]["num_bases"], config["data"]["channels"])
        if tuple(initial_basis.shape) != expected:
            raise ValueError(f"Forma basi {tuple(initial_basis.shape)}, attesa {expected}")
        self.num_bases = expected[0]
        self.encoder = AbundanceEncoder(config)
        self.latent_basis = nn.Parameter(initial_basis.detach().clone().float())

    def encode(self, images):
        return self.encoder(images)

    def decode(self, abundances):
        if abundances.ndim != 4 or abundances.shape[1] != self.num_bases:
            raise ValueError("Forma delle abundance non valida")
        return torch.einsum("bkhw,kc->bchw", abundances, self.latent_basis)

    def forward(self, images):
        logits, abundances = self.encode(images)
        return {
            "logits": logits,
            "abundances": abundances,
            "reconstruction": self.decode(abundances),
            "latent_basis": self.latent_basis,
        }


def create_ddpm_model(config):
    settings = config["ddpm_model"]
    channels = tuple(
        settings["base_channels"] * multiplier
        for multiplier in settings["channel_multipliers"]
    )
    levels = len(channels)
    latent_channels = config["model"]["num_bases"] - 1
    return UNet2DModel(
        sample_size=config["data"]["image_size"],
        in_channels=latent_channels,
        out_channels=latent_channels,
        down_block_types=("DownBlock2D",) * (levels - 1) + ("AttnDownBlock2D",),
        up_block_types=("AttnUpBlock2D",) + ("UpBlock2D",) * (levels - 1),
        block_out_channels=channels,
        layers_per_block=settings["num_res_blocks"],
        dropout=settings["dropout"],
        attention_head_dim=settings["attention_head_dim"],
        norm_num_groups=settings["norm_num_groups"],
        resnet_time_scale_shift=settings["resnet_time_scale_shift"],
        num_class_embeds=config["data"]["num_classes"] + 1,
        num_train_timesteps=config["diffusion"]["timesteps"],
    )


def create_training_scheduler(config):
    settings = config["diffusion"]
    return DDPMScheduler(
        num_train_timesteps=settings["timesteps"],
        beta_schedule=settings["beta_schedule"],
        prediction_type=settings["prediction_type"],
        clip_sample=settings["clip_sample"],
    )


def create_sampling_scheduler(config):
    return DDIMScheduler.from_config(create_training_scheduler(config).config)


@torch.inference_mode()
def generate_ILR(model, scheduler, labels, config, inference_steps, generator=None):
    latent_channels = config["model"]["num_bases"] - 1
    size = config["data"]["image_size"]
    latents = torch.randn(
        (len(labels), latent_channels, size, size),
        generator=generator,
        device=labels.device,
    )
    scheduler.set_timesteps(inference_steps, device=labels.device)
    for timestep in scheduler.timesteps:
        null_labels = torch.full_like(labels, config["data"]["num_classes"])
        model_input = torch.cat((latents, latents))
        label_input = torch.cat((null_labels, labels))
        prediction = model(model_input, timestep, class_labels=label_input).sample
        unconditional, conditional = prediction.chunk(2)
        prediction = unconditional + config["cfg"]["guidance_scale"] * (
            conditional - unconditional
        )
        latents = scheduler.step(prediction, timestep, latents).prev_sample
    return latents


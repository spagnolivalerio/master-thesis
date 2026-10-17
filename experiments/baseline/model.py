from diffusers import UNet2DModel

# Factory for the ddpm
def create_model(config):
    data = config["data"]
    model = config["model"]
    channels = tuple(
        model["base_channels"] * multiplier
        for multiplier in model["channel_multipliers"]
    )
    levels = len(channels)
    conditioning = config["cfg"].get("enabled", True)
    
    return UNet2DModel(
        sample_size=data["image_size"],
        in_channels=data["channels"],
        out_channels=data["channels"],
        down_block_types=("DownBlock2D",) * (levels - 1) + ("AttnDownBlock2D",),
        up_block_types=("AttnUpBlock2D",) + ("UpBlock2D",) * (levels - 1),
        block_out_channels=channels,
        layers_per_block=model["num_res_blocks"],
        dropout=model["dropout"],
        attention_head_dim=model["attention_head_dim"],
        norm_num_groups=model["norm_num_groups"],
        num_class_embeds=data["num_classes"] + 1 if conditioning else None,
        num_train_timesteps=config["diffusion"]["timesteps"],
    )


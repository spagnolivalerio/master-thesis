"""Unmixing per classe e diffusione sulle coordinate latenti delle abundance.

Il modello di unmixing ricostruisce le immagini come miscele di basi spettrali.
Ogni classe ha un encoder e un banco di basi indipendenti. Una sola U-Net
condizionale impara invece a generare le coordinate ILR standardizzate delle
abundance; le trasformazioni da e verso ILR sono definite in utils.py.

Forme dei tensori: B = batch, C = bande, K = basi per classe, H/W = dimensioni
spaziali. Immagini e basi sono nello spazio delle bande standardizzate.
"""

import torch
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel
from torch import nn


class ResidualBlock(nn.Module):
    """Due convoluzioni con connessione residua, senza cambiare la forma."""

    def __init__(self, channels, groups):
        super().__init__()
        self.norm_1 = nn.GroupNorm(groups, channels)
        self.conv_1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, padding_mode="reflect"
        )
        self.norm_2 = nn.GroupNorm(groups, channels)
        self.conv_2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, padding_mode="reflect"
        )
        self.activation = nn.SiLU()

    def forward(self, inputs):
        hidden = self.norm_1(inputs)
        hidden = self.activation(hidden)
        hidden = self.conv_1(hidden)
        hidden = self.norm_2(hidden)
        hidden = self.activation(hidden)
        hidden = self.conv_2(hidden)
        return inputs + hidden


class AbundanceEncoder(nn.Module):
    """Da immagini [B,C,H,W] a K pesi di miscela per ciascun pixel.

    La risoluzione spaziale resta invariata. La softmax lungo le K basi
    produce abundance positive con somma unitaria per ciascun pixel.
    """

    def __init__(self, config):
        super().__init__()
        data = config["data"]
        settings = config["model"]
        hidden_channels = settings["hidden_channels"]
        self.temperature = settings["abundance_temperature"]
        self.input_projection = nn.Conv2d(
            data["channels"],
            hidden_channels,
            kernel_size=3,
            padding=1,
            padding_mode="reflect",
        )
        self.blocks = nn.Sequential(
            *[
                ResidualBlock(hidden_channels, settings["norm_num_groups"])
                for _ in range(settings["num_res_blocks"])
            ]
        )
        self.output_norm = nn.GroupNorm(settings["norm_num_groups"], hidden_channels)
        self.output = nn.Conv2d(hidden_channels, settings["num_bases"], kernel_size=1)
        self.activation = nn.SiLU()

    def forward(self, images):
        """Restituisce logits liberi e abundance, entrambi [B,K,H,W]."""
        hidden = self.input_projection(images)
        hidden = self.blocks(hidden)
        hidden = self.output_norm(hidden)
        hidden = self.activation(hidden)
        logits = self.output(hidden)

        # La softmax usa float32 anche durante il training in mixed precision.
        # Una temperatura più bassa concentra i pesi sulle basi dominanti.
        abundances = torch.softmax(logits.float() / self.temperature, dim=1)
        return logits, abundances


class ClassSpecificLatentUnmixingModel(nn.Module):
    """Seleziona encoder e basi con la classe reale di ogni immagine.

    Il decoder è una miscela lineare: non contiene una rete aggiuntiva.
    Le basi sono parametri liberi [num_classes,K,C]; non hanno un vincolo
    di positività e non sono necessariamente spettri di materiali reali.
    """

    def __init__(self, config, initial_bases):
        super().__init__()
        self.num_classes = config["data"]["num_classes"]
        self.num_bases = config["model"]["num_bases"]
        self.channels = config["data"]["channels"]
        expected = (self.num_classes, self.num_bases, self.channels)
        if tuple(initial_bases.shape) != expected:
            raise ValueError(
                f"Forma dei banchi di basi {tuple(initial_bases.shape)}, attesa {expected}"
            )
        self.encoders = nn.ModuleList(
            [AbundanceEncoder(config) for _ in range(self.num_classes)]
        )
        # Ogni riga del banco di una classe è uno spettro di C bande.
        self.latent_bases = nn.Parameter(initial_bases.detach().clone().float())

    def _validate_labels(self, images, labels):
        if labels is None or labels.ndim != 1 or len(labels) != len(images):
            raise ValueError("Serve una label per ogni immagine")
        if labels.min() < 0 or labels.max() >= self.num_classes:
            raise ValueError("Label fuori intervallo")

    def encode(self, images, labels):
        """Applica gli encoder per classe e conserva l'ordine del batch.

        images: [B,C,H,W]; labels: [B], interi da 0 a num_classes - 1
        sullo stesso device delle immagini. Restituisce due tensori [B,K,H,W].
        """
        self._validate_labels(images, labels)
        batch_size = images.shape[0]
        height, width = images.shape[2:]
        abundance_shape = (batch_size, self.num_bases, height, width)
        logits = images.new_zeros(abundance_shape)
        abundances = images.new_zeros(abundance_shape, dtype=torch.float32)

        for class_index, encoder in enumerate(self.encoders):
            indices = torch.nonzero(labels == class_index, as_tuple=False).flatten()
            if indices.numel() == 0:
                continue

            class_images = images.index_select(0, indices)
            class_logits, class_abundances = encoder(class_images)
            # Ricompone il batch nelle posizioni originali, mantenendo il grafo
            # per il backward verso gli encoder delle classi presenti.
            logits = logits.index_copy(0, indices, class_logits.to(logits.dtype))
            abundances = abundances.index_copy(0, indices, class_abundances)
        return logits, abundances

    def decode(self, abundances, labels):
        """Ricostruisce [B,C,H,W] usando il banco della classe di ogni campione.

        Per ogni banda c e pixel (h,w):
        reconstruction[b,c,h,w] = sum_k abundances[b,k,h,w] * bases[b,k,c].
        """
        if abundances.ndim != 4 or abundances.shape[1] != self.num_bases:
            raise ValueError("Forma delle abundance non valida")
        self._validate_labels(abundances, labels)
        class_bases = self.latent_bases.index_select(0, labels)  # [B,K,C]
        return torch.einsum("bkhw,bkc->bchw", abundances, class_bases)

    def forward(self, images, labels):
        """Esegue encoding e ricostruzione, esponendo i valori per la loss."""
        logits, abundances = self.encode(images, labels)
        reconstruction = self.decode(abundances, labels)
        return {
            "logits": logits,
            "abundances": abundances,
            "reconstruction": reconstruction,
            "latent_bases": self.latent_bases,
        }


def create_ddpm_model(config):
    """Crea la U-Net condizionale che predice il rumore nello spazio ILR."""
    settings = config["ddpm_model"]
    block_channels = tuple(
        settings["base_channels"] * multiplier
        for multiplier in settings["channel_multipliers"]
    )
    num_levels = len(block_channels)
    # K abundance con somma 1 hanno K-1 coordinate ILR indipendenti.
    latent_channels = config["model"]["num_bases"] - 1
    # Attention al livello di risoluzione più basso della U-Net.
    down_blocks = ("DownBlock2D",) * (num_levels - 1) + ("AttnDownBlock2D",)
    up_blocks = ("AttnUpBlock2D",) + ("UpBlock2D",) * (num_levels - 1)
    # La classe aggiuntiva rappresenta il caso senza condizionamento per CFG.
    num_class_embeddings = config["data"]["num_classes"] + 1

    return UNet2DModel(
        sample_size=config["data"]["image_size"],
        in_channels=latent_channels,
        out_channels=latent_channels,
        down_block_types=down_blocks,
        up_block_types=up_blocks,
        block_out_channels=block_channels,
        layers_per_block=settings["num_res_blocks"],
        dropout=settings["dropout"],
        attention_head_dim=settings["attention_head_dim"],
        norm_num_groups=settings["norm_num_groups"],
        resnet_time_scale_shift=settings["resnet_time_scale_shift"],
        num_class_embeds=num_class_embeddings,
        num_train_timesteps=config["diffusion"]["timesteps"],
    )


def create_training_scheduler(config):
    """Definisce il processo di rumore DDPM usato durante il training."""
    settings = config["diffusion"]
    return DDPMScheduler(
        num_train_timesteps=settings["timesteps"],
        beta_schedule=settings["beta_schedule"],
        prediction_type=settings["prediction_type"],
        clip_sample=settings["clip_sample"],
    )


def create_sampling_scheduler(config):
    """Usa DDIM per generare, con la stessa configurazione di rumore del DDPM."""
    return DDIMScheduler.from_config(create_training_scheduler(config).config)


@torch.inference_mode()
def generate_ILR(model, scheduler, labels, config, inference_steps, generator=None):
    """Dal rumore a coordinate ILR standardizzate [B,K-1,H,W] tramite CFG.

    Il chiamante imposta model.eval() e converte poi il risultato in abundance
    con from_ILR; il decoder ricostruisce infine le immagini multispettrali.
    """
    latent_channels = config["model"]["num_bases"] - 1
    image_size = config["data"]["image_size"]
    guidance_scale = config["cfg"]["guidance_scale"]
    latents = torch.randn(
        (len(labels), latent_channels, image_size, image_size),
        generator=generator,
        device=labels.device,
    )
    null_labels = torch.full_like(labels, config["data"]["num_classes"])
    model_labels = torch.cat((null_labels, labels))
    scheduler.set_timesteps(inference_steps, device=labels.device)
    for timestep in scheduler.timesteps:
        # Un solo forward calcola entrambe le predizioni sullo stesso latente:
        # la prima metà senza classe, la seconda con la classe richiesta.
        model_input = torch.cat((latents, latents))
        prediction = model(model_input, timestep, class_labels=model_labels).sample
        unconditional_prediction, conditional_prediction = prediction.chunk(2)
        guided_prediction = unconditional_prediction + guidance_scale * (
            conditional_prediction - unconditional_prediction
        )
        latents = scheduler.step(guided_prediction, timestep, latents).prev_sample
    return latents

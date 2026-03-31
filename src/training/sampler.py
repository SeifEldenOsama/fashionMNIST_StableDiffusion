import os

import torch
from torchvision.utils import save_image

from config import config
from src.models import VAE, LatentEpsModel, ConditionalDenoiseDiffusion
from src.training.checkpoints import load_checkpoint


def sample_latent_images(
    device: torch.device,
    vae: VAE | None = None,
    model: LatentEpsModel | None = None,
    suffix: str = "final",
) -> str:
    if vae is None:
        vae = VAE().to(device)
        if not load_checkpoint(vae, config.VAE_PATH, device=str(device)):
            raise FileNotFoundError(f"VAE checkpoint not found at '{config.VAE_PATH}'.")
        for p in vae.parameters():
            p.requires_grad_(False)
    vae.eval()

    if model is None:
        model = LatentEpsModel().to(device)
        if not load_checkpoint(model, config.LATENT_EMA_PATH, device=str(device)):
            raise FileNotFoundError(f"EMA checkpoint not found at '{config.LATENT_EMA_PATH}'.")
    model.eval()

    scheduler     = ConditionalDenoiseDiffusion(model, device=device)
    n_per_class   = config.N_SAMPLES // config.NUM_CLASSES
    target_labels = torch.arange(config.NUM_CLASSES, device=device).repeat_interleave(n_per_class)
    latent_shape  = (config.N_SAMPLES, config.LATENT_CHANNELS, config.LATENT_SIZE, config.LATENT_SIZE)

    with torch.no_grad():
        z        = scheduler.sample(latent_shape, device=device, c=target_labels)
        x_images = vae.decode(z).clamp(-1.0, 1.0)

    x_images = (x_images + 1.0) * 0.5

    os.makedirs(config.SAMPLES_DIR, exist_ok=True)
    out_path = os.path.join(config.SAMPLES_DIR, f"samples_{suffix}.png")
    save_image(x_images, out_path, nrow=n_per_class)
    print(f"Saved -> {out_path}")
    return out_path

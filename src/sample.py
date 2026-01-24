import torch
from torchvision.utils import save_image
from .config import config
from .models.vae import VAE
from .models.unet import LatentEpsModel
from .models.diffusion import ConditionalDenoiseDiffusion
from .utils import load_checkpoint

def sample_latent_images(device, model=None):
    vae = VAE().to(device)
    if not load_checkpoint(vae, config.VAE_PATH, device=device):
        print("ERROR: VAE weights not found. Cannot sample.")
        return

    if model is None:
        model = LatentEpsModel().to(device)
        if not load_checkpoint(model, config.LATENT_EMA_PATH, device=device):
            print("ERROR: EMA Latent model not found. Cannot sample.")
            return

    for p in vae.parameters():
        p.requires_grad_(False)
    vae.eval()
    model.eval()

    sched = ConditionalDenoiseDiffusion(model, device=device)

    n_per_class = config.N_SAMPLES // config.NUM_CLASSES
    target_labels = torch.arange(config.NUM_CLASSES, device=device).repeat_interleave(n_per_class)

    latent_shape = (config.N_SAMPLES, config.LATENT_CHANNELS, config.LATENT_SIZE, config.LATENT_SIZE)
    print(f"Generating {config.N_SAMPLES} samples in latent space {latent_shape}...")

    with torch.no_grad():
        z_samples = sched.sample(
            shape=latent_shape,
            device=device,
            c=target_labels
        )
        x_samples = vae.decode(z_samples).clamp(-1, 1)

    x_samples = (x_samples + 1) * 0.5
    out_path = "fashion_latent_samples.png"
    save_image(x_samples, out_path, nrow=n_per_class)
    print(f"Generated samples saved to {out_path}.")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_latent_images(device)

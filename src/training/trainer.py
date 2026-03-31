import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import config
from src.data import get_dataloader
from src.models import VAE, LatentEpsModel, ConditionalDenoiseDiffusion, build_ema, update_ema
from src.training.checkpoints import save_checkpoint, load_checkpoint
from src.training.sampler import sample_latent_images


def train_vae(device: torch.device) -> VAE:
    vae       = VAE().to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    loader    = get_dataloader(train=True)

    for epoch in range(config.VAE_EPOCHS):
        vae.train()
        pbar       = tqdm(loader, desc=f"VAE [{epoch+1:02d}/{config.VAE_EPOCHS}]", leave=False)
        epoch_loss = 0.0

        for imgs, _ in pbar:
            imgs = imgs.to(device)
            optimizer.zero_grad()

            recon, mu, logvar, _ = vae(imgs)
            recon_loss = F.mse_loss(recon, imgs)
            kl_div     = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss       = recon_loss + config.KL_WEIGHT * kl_div

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(recon=f"{recon_loss.item():.4f}", kl=f"{kl_div.item():.4f}")

        print(f"VAE Epoch {epoch+1:02d}/{config.VAE_EPOCHS} avg_loss={epoch_loss / len(loader):.4f}")

    save_checkpoint(vae, config.VAE_PATH)
    return vae


def train_latent_ddpm(device: torch.device) -> tuple[LatentEpsModel, LatentEpsModel]:
    vae = VAE().to(device)
    if not load_checkpoint(vae, config.VAE_PATH, device=str(device)):
        raise FileNotFoundError(f"VAE checkpoint not found at '{config.VAE_PATH}'.")
    for p in vae.parameters():
        p.requires_grad_(False)
    vae.eval()

    model     = LatentEpsModel().to(device)
    ema_model = build_ema(model)
    load_checkpoint(model, config.LATENT_MODEL_PATH, device=str(device))

    scheduler = ConditionalDenoiseDiffusion(model, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    loader    = get_dataloader(train=True)

    for epoch in range(config.N_EPOCHS):
        model.train()
        pbar       = tqdm(loader, desc=f"CLDM [{epoch+1:03d}/{config.N_EPOCHS}]", leave=False)
        epoch_loss = 0.0

        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            with torch.no_grad():
                mu, logvar = vae.encode(imgs)
                z0 = mu + torch.randn_like(mu) * (0.5 * logvar).exp()

            optimizer.zero_grad()
            loss = scheduler.loss(z0, labels)
            loss.backward()
            optimizer.step()

            update_ema(ema_model, model)

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        print(f"CLDM Epoch {epoch+1:03d}/{config.N_EPOCHS} avg_loss={epoch_loss / len(loader):.4f}")

        should_save = (
            (epoch + 1) % config.SAVE_EVERY_N_EPOCHS == 0
            or (epoch + 1) == config.N_EPOCHS
        )
        if should_save:
            save_checkpoint(model,     config.LATENT_MODEL_PATH)
            save_checkpoint(ema_model, config.LATENT_EMA_PATH)
            sample_latent_images(device, vae=vae, model=ema_model, suffix=f"ep{epoch+1:03d}")

    return model, ema_model

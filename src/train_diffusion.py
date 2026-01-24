import torch
from tqdm import tqdm
from copy import deepcopy
from .config import config
from .dataset import get_dataloader
from .models.vae import VAE
from .models.unet import LatentEpsModel
from .models.diffusion import ConditionalDenoiseDiffusion
from .utils import save_checkpoint, load_checkpoint

def train_latent_ddpm(device):
    vae = VAE().to(device)
    if not load_checkpoint(vae, config.VAE_PATH, device=device):
        print("ERROR: VAE weights not found. Please run 'train_vae' first!")
        return

    for p in vae.parameters():
        p.requires_grad_(False)
    vae.eval()
    print("Attention-Enhanced VAE is loaded and frozen.")

    model = LatentEpsModel().to(device)
    ema_model = deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad_(False)

    load_checkpoint(model, config.LATENT_MODEL_PATH, device=device)

    sched = ConditionalDenoiseDiffusion(model, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    dataloader = get_dataloader(train=True)

    print("--- Starting Fashion CLDM Training ---")
    for epoch in range(config.N_EPOCHS):
        model.train()
        pbar = tqdm(dataloader, desc=f"CLDM Epoch {epoch+1}/{config.N_EPOCHS}")
        total_loss = 0.0

        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            with torch.no_grad():
                mu, log_var = vae.encode(imgs)
                z0 = mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)

            optimizer.zero_grad()
            loss = sched.loss(z0, labels)

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                    ema_p.mul_(config.EMA_DECAY).add_(p * (1 - config.EMA_DECAY))

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        print(f"\n--- Epoch {epoch+1} finished. Avg Loss: {total_loss / len(dataloader):.4f} ---")

        if (epoch+1) % config.SAVE_EVERY_N_EPOCHS == 0 or (epoch+1) == config.N_EPOCHS:
            save_checkpoint(model, config.LATENT_MODEL_PATH)
            save_checkpoint(ema_model, config.LATENT_EMA_PATH)
            print(f"Fashion CLDM Checkpoint saved at epoch {epoch+1}")

    return model, ema_model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_latent_ddpm(device)

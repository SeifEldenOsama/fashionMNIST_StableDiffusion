import torch
import torch.nn.functional as F
from tqdm import tqdm
from .config import config
from .dataset import get_dataloader
from .models.vae import VAE
from .utils import save_checkpoint

def train_vae(device):
    vae = VAE().to(device)
    vae_optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    dataloader = get_dataloader(train=True)

    print("--- Starting Attention-Enhanced VAE Pre-training (Fashion MNIST) ---")
    for epoch in range(config.VAE_EPOCHS):
        pbar = tqdm(dataloader, desc=f"VAE Epoch {epoch+1}/{config.VAE_EPOCHS}")
        for imgs, _ in pbar:
            imgs = imgs.to(device)
            vae_optimizer.zero_grad()

            recon, mu, logvar, _ = vae(imgs)

            recon_loss = F.mse_loss(recon, imgs, reduction='mean')
            kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + config.KL_WEIGHT * kl_div

            loss.backward()
            vae_optimizer.step()
            pbar.set_postfix(recon=f"{recon_loss.item():.4f}", kl=f"{kl_div.item():.4f}")

    save_checkpoint(vae, config.VAE_PATH)
    print(f"Fashion VAE weights saved to {config.VAE_PATH}. VAE training complete.")
    return vae

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_vae(device)

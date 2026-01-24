import torch
import torch.nn as nn
from ..config import config

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class Encoder(nn.Module):
    def __init__(self, latent_channels=config.LATENT_CHANNELS):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(config.IMAGE_CHANNELS, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),

            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            ChannelAttention(64),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            ChannelAttention(128)
        )

        self.to_mu = nn.Conv2d(128, latent_channels, 1)
        self.to_logvar = nn.Conv2d(128, latent_channels, 1)

    def forward(self, x):
        h = self.net(x)
        return self.to_mu(h), self.to_logvar(h)

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

class Decoder(nn.Module):
    def __init__(self, latent_channels=config.LATENT_CHANNELS):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(latent_channels, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            ChannelAttention(64),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),

            nn.Conv2d(32, config.IMAGE_CHANNELS, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)

class VAE(nn.Module):
    def __init__(self, latent_channels=config.LATENT_CHANNELS):
        super().__init__()
        self.encoder = Encoder(latent_channels)
        self.decoder = Decoder(latent_channels)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar, z

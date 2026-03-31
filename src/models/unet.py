import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from config import config


def get_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    exp  = torch.exp(-math.log(10_000) * torch.arange(half, device=timesteps.device) / half)
    emb  = timesteps.float().unsqueeze(-1) * exp.unsqueeze(0)
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1).float()


class CrossAttention(nn.Module):
    def __init__(self, query_dim: int, context_dim: int, num_heads: int = 4):
        super().__init__()
        self.q_proj   = nn.Linear(query_dim, query_dim)
        self.k_proj   = nn.Linear(context_dim, query_dim)
        self.v_proj   = nn.Linear(context_dim, query_dim)
        self.attn     = nn.MultiheadAttention(query_dim, num_heads, batch_first=True)
        self.proj_out = nn.Linear(query_dim, query_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(context)
        v = self.v_proj(context)
        out, _ = self.attn(q, k, v)
        return x + self.proj_out(out)


class LatentResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, context_dim: int):
        super().__init__()
        self.conv1    = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_ch))
        self.conv2    = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.norm_ca  = nn.GroupNorm(8, out_ch)
        self.ca       = CrossAttention(out_ch, context_dim)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        out_ch = self.conv1.out_channels

        out = self.conv1(F.silu(F.group_norm(x, min(8, x.shape[1]))))
        out = out + self.time_mlp(t_emb)[:, :, None, None]

        h_ca = F.silu(self.norm_ca(out))
        h_ca = h_ca.permute(0, 2, 3, 1).reshape(b, h * w, out_ch)
        h_ca = self.ca(h_ca, context)
        out  = out + h_ca.transpose(1, 2).reshape(b, out_ch, h, w)

        out = self.conv2(F.silu(F.group_norm(out, min(8, out.shape[1]))))
        return out + self.shortcut(x)


class LatentEpsModel(nn.Module):
    def __init__(self):
        super().__init__()
        time_dim    = config.TIME_EMBED_DIM
        context_dim = config.CONTEXT_DIM

        self.time_mlp  = nn.Sequential(
            nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim),
        )
        self.class_emb = nn.Embedding(config.NUM_CLASSES, context_dim)
        self.conv_in   = nn.Conv2d(config.LATENT_CHANNELS, config.BASE_CHANNELS, 3, padding=1)

        self.downs = nn.ModuleList()
        ch = config.BASE_CHANNELS
        for mult in config.CHANNEL_MULT:
            out    = config.BASE_CHANNELS * mult
            blocks = nn.ModuleList([LatentResBlock(ch, out, time_dim, context_dim)])
            for _ in range(config.NUM_RES_BLOCKS - 1):
                blocks.append(LatentResBlock(out, out, time_dim, context_dim))
            self.downs.append(nn.ModuleDict({
                "blocks": blocks,
                "down":   nn.Conv2d(out, out, 3, 1, 1),
            }))
            ch = out

        self.bot1 = LatentResBlock(ch, ch, time_dim, context_dim)
        self.bot2 = LatentResBlock(ch, ch, time_dim, context_dim)

        self.ups = nn.ModuleList()
        for mult in reversed(config.CHANNEL_MULT):
            skip_ch = config.BASE_CHANNELS * mult
            out     = skip_ch
            blocks  = nn.ModuleList([LatentResBlock(out + skip_ch, out, time_dim, context_dim)])
            for _ in range(config.NUM_RES_BLOCKS - 1):
                blocks.append(LatentResBlock(out, out, time_dim, context_dim))
            self.ups.append(nn.ModuleDict({
                "blocks": blocks,
                "up":     nn.ConvTranspose2d(ch, out, 3, 1, 1),
            }))
            ch = out

        self.conv_out = nn.Conv2d(ch, config.LATENT_CHANNELS, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        t_emb   = self.time_mlp(get_timestep_embedding(t, config.TIME_EMBED_DIM))
        context = self.class_emb(y).unsqueeze(1)

        skips = []
        h = self.conv_in(x)
        for module in self.downs:
            for block in module["blocks"]:
                h = block(h, t_emb, context)
            skips.append(h)
            h = module["down"](h)

        h = self.bot1(h, t_emb, context)
        h = self.bot2(h, t_emb, context)

        for module in self.ups:
            h = module["up"](h)
            h = torch.cat([h, skips.pop()], dim=1)
            for block in module["blocks"]:
                h = block(h, t_emb, context)

        return self.conv_out(h)


class ConditionalDenoiseDiffusion:
    def __init__(self, eps_model: nn.Module, n_steps: int = config.N_STEPS, device=None):
        self.eps_model = eps_model
        self.device    = device or torch.device("cpu")

        beta           = torch.linspace(0.0001, 0.02, n_steps).to(self.device)
        alpha          = 1.0 - beta
        alpha_bar      = torch.cumprod(alpha, dim=0)
        alpha_bar_prev = F.pad(alpha_bar[:-1], (1, 0), value=1.0)

        self.n_steps                  = n_steps
        self.beta                     = beta
        self.alpha                    = alpha
        self.alpha_bar                = alpha_bar
        self.sqrt_alpha_bar           = alpha_bar.sqrt()
        self.sqrt_one_minus_alpha_bar = (1.0 - alpha_bar).sqrt()
        self.post_variance            = beta * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps=None) -> torch.Tensor:
        if eps is None:
            eps = torch.randn_like(x0)
        a  = self.sqrt_alpha_bar[t].reshape(-1, 1, 1, 1)
        om = self.sqrt_one_minus_alpha_bar[t].reshape(-1, 1, 1, 1)
        return a * x0 + om * eps

    def p_sample(self, xt: torch.Tensor, t, c=None) -> torch.Tensor:
        if not isinstance(t, torch.Tensor):
            t = torch.full((xt.shape[0],), t, device=xt.device, dtype=torch.long)

        eps_theta = self.eps_model(xt, t, c)

        x0_pred = (
            (xt - self.sqrt_one_minus_alpha_bar[t].reshape(-1, 1, 1, 1) * eps_theta)
            / self.sqrt_alpha_bar[t].reshape(-1, 1, 1, 1)
        ).clamp(-1.0, 1.0)

        alpha_bar_t  = self.alpha_bar[t].reshape(-1, 1, 1, 1)
        alpha_bar_tp = F.pad(self.alpha_bar[:-1], (1, 0), value=1.0)[t].reshape(-1, 1, 1, 1)
        alpha_t      = self.alpha[t].reshape(-1, 1, 1, 1)

        mean = (
            alpha_bar_tp.sqrt() * self.beta[t].reshape(-1, 1, 1, 1) / (1.0 - alpha_bar_t) * x0_pred
            + alpha_t.sqrt() * (1.0 - alpha_bar_tp) / (1.0 - alpha_bar_t) * xt
        )

        if t[0] > 0:
            return mean + self.post_variance[t].reshape(-1, 1, 1, 1).sqrt() * torch.randn_like(xt)
        return mean

    @torch.no_grad()
    def sample(self, shape: tuple, device: torch.device, c=None) -> torch.Tensor:
        x = torch.randn(shape, device=device)
        for step in tqdm(reversed(range(self.n_steps)), total=self.n_steps, desc="Sampling"):
            x = self.p_sample(x, step, c)
        return x

    def loss(self, x0: torch.Tensor, labels=None) -> torch.Tensor:
        t   = torch.randint(0, self.n_steps, (x0.shape[0],), device=x0.device, dtype=torch.long)
        eps = torch.randn_like(x0)
        xt  = self.q_sample(x0, t, eps)
        return F.mse_loss(eps, self.eps_model(xt, t, labels))


def build_ema(model: nn.Module) -> nn.Module:
    ema = deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema


def update_ema(ema_model: nn.Module, model: nn.Module, decay: float = config.EMA_DECAY) -> None:
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.mul_(decay).add_(p.detach() * (1.0 - decay))

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..config import config

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, num_heads=4):
        super().__init__()
        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(context_dim, query_dim)
        self.v_proj = nn.Linear(context_dim, query_dim)
        self.attn = nn.MultiheadAttention(query_dim, num_heads, batch_first=True)
        self.proj_out = nn.Linear(query_dim, query_dim)

    def forward(self, x, context):
        q = self.q_proj(x)
        k = self.k_proj(context)
        v = self.v_proj(context)
        attn_output, _ = self.attn(q, k, v)
        return x + self.proj_out(attn_output)

class LatentResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, context_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_ch))
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.norm_ca = nn.GroupNorm(8, out_ch)
        self.ca = CrossAttention(out_ch, context_dim)

    def forward(self, x, t_emb, context):
        b, _, h, w = x.shape
        out_ch = self.conv1.out_channels
        h_temp = F.group_norm(x, min(8, x.shape[1]))
        h_temp = self.conv1(F.silu(h_temp))
        h_temp = h_temp + self.time_mlp(t_emb)[:, :, None, None]
        h_ca = F.silu(self.norm_ca(h_temp))
        new_hw = int(h) * int(w)
        h_ca = h_ca.permute(0, 2, 3, 1).reshape(b, new_hw, out_ch)
        h_ca = self.ca(h_ca, context)
        h_temp = h_temp + h_ca.transpose(1, 2).reshape(b, out_ch, h, w)
        h_temp = self.conv2(F.silu(F.group_norm(h_temp, min(8, h_temp.shape[1]))))
        return h_temp + self.shortcut(x)

class LatentEpsModel(nn.Module):
    def __init__(self):
        super().__init__()
        time_dim = config.TIME_EMBED_DIM
        context_dim = config.CONTEXT_DIM
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.class_emb = nn.Embedding(config.NUM_CLASSES, context_dim)
        self.conv_in = nn.Conv2d(config.LATENT_CHANNELS, config.BASE_CHANNELS, 3, padding=1)

        self.downs = nn.ModuleList()
        ch = config.BASE_CHANNELS
        for mult in config.CHANNEL_MULT:
            out = config.BASE_CHANNELS * mult
            blocks = nn.ModuleList()
            blocks.append(LatentResBlock(ch, out, time_dim, context_dim))
            for _ in range(config.NUM_RES_BLOCKS - 1):
                blocks.append(LatentResBlock(out, out, time_dim, context_dim))
            downsample = nn.Conv2d(out, out, 3, 1, 1)
            self.downs.append(nn.ModuleDict({"blocks": blocks, "down": downsample}))
            ch = out

        self.bot1 = LatentResBlock(ch, ch, time_dim, context_dim)
        self.bot2 = LatentResBlock(ch, ch, time_dim, context_dim)

        self.ups = nn.ModuleList()
        for mult in reversed(config.CHANNEL_MULT):
            out = config.BASE_CHANNELS * mult
            blocks = nn.ModuleList()
            blocks.append(LatentResBlock(ch + out, out, time_dim, context_dim))
            for _ in range(config.NUM_RES_BLOCKS - 1):
                blocks.append(LatentResBlock(out + out, out, time_dim, context_dim))
            upsample = nn.ConvTranspose2d(out, out, 3, 1, 1)
            self.ups.append(nn.ModuleDict({"blocks": blocks, "up": upsample}))
            ch = out

        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, config.LATENT_CHANNELS, 3, padding=1)
        )

    def forward(self, x, t, c):
        from ..utils import get_timestep_embedding
        t_emb = self.time_mlp(get_timestep_embedding(t, config.TIME_EMBED_DIM))
        c_emb = self.class_emb(c).unsqueeze(1)
        x = self.conv_in(x)
        skips = []
        for level in self.downs:
            for block in level["blocks"]:
                x = block(x, t_emb, c_emb)
                skips.append(x)
            x = level["down"](x)
        x = self.bot1(x, t_emb, c_emb)
        x = self.bot2(x, t_emb, c_emb)
        for level in self.ups:
            x = level["up"](x)
            for block in level["blocks"]:
                x = block(torch.cat([x, skips.pop()], dim=1), t_emb, c_emb)
        return self.conv_out(x)

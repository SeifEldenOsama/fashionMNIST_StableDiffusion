import torch
import os
import math

def get_timestep_embedding(timesteps, dim):
    half_dim = dim // 2
    exponent = torch.exp(-math.log(10000) * torch.arange(half_dim, device=timesteps.device) / half_dim)
    timesteps = timesteps.float().unsqueeze(-1)
    emb = timesteps * exponent.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return emb.to(dtype=torch.float32)

def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_checkpoint(model, path, device="cpu"):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"Checkpoint loaded from {path}")
        return True
    return False

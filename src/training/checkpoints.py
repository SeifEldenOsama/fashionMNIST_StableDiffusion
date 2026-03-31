import os
import torch
import torch.nn as nn


def save_checkpoint(model: nn.Module, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Saved -> {path}")


def load_checkpoint(model: nn.Module, path: str, device: str = "cpu") -> bool:
    if not os.path.exists(path):
        return False
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"Loaded <- {path}")
    return True

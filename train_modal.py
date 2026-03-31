import os
import sys

import modal

app = modal.App("fashion-ldm")

volume = modal.Volume.from_name("fashion-ldm-vol", create_if_missing=True)

# Use a single mount point for the volume
VOLUME_MOUNT = "/mnt/fashion-ldm"

image = (
    modal.Image.debian_slim(python_version="3.11")
    # 1. Install PyTorch from the specialized CUDA index
    .pip_install(
        "torch==2.3.1",
        "torchvision==0.18.1",
        extra_options="--index-url https://download.pytorch.org/whl/cu121",
    )
    # 2. Install other packages from the default PyPI index
    .pip_install(
        "tqdm",
        "streamlit",
        "pillow",
        "numpy",
    )
    .add_local_dir(".", remote_path="/root/project")
)


def _setup_path():
    sys.path.insert(0, "/root/project")
    # Dynamically update the config object to use the new volume mount
    from config import config
    config.DATA_DIR = f"{VOLUME_MOUNT}/data"
    config.VAE_PATH = f"{VOLUME_MOUNT}/checkpoints/fashion_vae.pt"
    config.LATENT_MODEL_PATH = f"{VOLUME_MOUNT}/checkpoints/latent_model.pt"
    config.LATENT_EMA_PATH = f"{VOLUME_MOUNT}/checkpoints/latent_model_ema.pt"
    config.SAMPLES_DIR = f"{VOLUME_MOUNT}/checkpoints/samples"
    
    # Ensure directories exist on the volume
    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(config.SAMPLES_DIR, exist_ok=True)


def _get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60 * 2,
    volumes={VOLUME_MOUNT: volume},
)
def train_vae_remote():
    _setup_path()
    from src.training import train_vae
    device = _get_device()
    train_vae(device)
    volume.commit()


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60 * 6,
    volumes={VOLUME_MOUNT: volume},
)
def train_diffusion_remote():
    _setup_path()
    from src.training import train_latent_ddpm
    device = _get_device()
    train_latent_ddpm(device)
    volume.commit()


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 30,
    volumes={VOLUME_MOUNT: volume},
)
def sample_remote():
    _setup_path()
    from src.training import sample_latent_images
    device = _get_device()
    sample_latent_images(device, suffix="manual")
    volume.commit()


@app.local_entrypoint()
def main():
    train_vae_remote.remote()
    train_diffusion_remote.remote()


@app.function(image=image, volumes={VOLUME_MOUNT: volume})
def _list_volume_files() -> list[tuple[str, int]]:
    results = []
    for root, _, files in os.walk(VOLUME_MOUNT):
        for f in files:
            full = os.path.join(root, f)
            results.append((full, os.path.getsize(full)))
    return results


@app.function(image=image, volumes={VOLUME_MOUNT: volume})
def _read_file(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


@app.local_entrypoint()
def download_outputs():
    out_dir = "./outputs"
    os.makedirs(out_dir, exist_ok=True)

    files = _list_volume_files.remote()
    if not files:
        print("Volume is empty.")
        return

    for remote_path, _ in files:
        rel   = os.path.relpath(remote_path, VOLUME_MOUNT)
        local = os.path.join(out_dir, rel)
        os.makedirs(os.path.dirname(local), exist_ok=True)
        data  = _read_file.remote(remote_path)
        with open(local, "wb") as fh:
            fh.write(data)
        print(f"{remote_path} -> {local}")


# Streamlit App Hosting
@app.function(
    image=image,
    gpu="H100",
    volumes={VOLUME_MOUNT: volume},
    timeout=60 * 60,
)
@modal.wsgi_app()
def streamlit_app():
    import streamlit.web.cli as stcli
    import sys

    _setup_path()
    
    # Set the streamlit app file path
    sys.argv = [
        "streamlit",
        "run",
        "/root/project/streamlit_app.py",
        "--server.port=8000",
        "--server.address=0.0.0.0",
    ]
    
    return stcli.main()

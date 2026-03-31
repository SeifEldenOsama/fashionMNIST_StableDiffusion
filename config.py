from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    DATA_DIR: str = "/data"
    IMAGE_SIZE: int = 28
    IMAGE_CHANNELS: int = 1
    NUM_CLASSES: int = 10

    BATCH_SIZE: int = 128
    NUM_WORKERS: int = 4
    LEARNING_RATE: float = 1e-4
    N_EPOCHS: int = 100
    SAVE_EVERY_N_EPOCHS: int = 10
    EMA_DECAY: float = 0.999

    LATENT_CHANNELS: int = 4
    LATENT_SIZE: int = 7
    KL_WEIGHT: float = 1e-4
    VAE_EPOCHS: int = 35
    VAE_PATH: str = "/checkpoints/fashion_vae.pt"

    N_STEPS: int = 1000
    LATENT_MODEL_PATH: str = "/checkpoints/latent_model.pt"
    LATENT_EMA_PATH: str = "/checkpoints/latent_model_ema.pt"

    BASE_CHANNELS: int = 64
    CHANNEL_MULT: List[int] = field(default_factory=lambda: [1, 2, 2])
    NUM_RES_BLOCKS: int = 2

    @property
    def TIME_EMBED_DIM(self) -> int:
        return self.BASE_CHANNELS * 4

    @property
    def CONTEXT_DIM(self) -> int:
        return self.TIME_EMBED_DIM

    N_SAMPLES: int = 80
    GUIDANCE_WEIGHT: float = 0.0
    SAMPLES_DIR: str = "/checkpoints/samples"


config = Config()

class Config:
    DATA_DIR = "./data"
    IMAGE_SIZE = 28
    IMAGE_CHANNELS = 1
    NUM_CLASSES = 10
    BATCH_SIZE = 128
    NUM_WORKERS = 2

    N_STEPS = 1000
    LEARNING_RATE = 1e-4
    N_EPOCHS = 100
    SAVE_EVERY_N_EPOCHS = 10
    EMA_DECAY = 0.999
    KL_WEIGHT = 1e-4

    LATENT_CHANNELS = 4
    LATENT_SIZE = 7
    VAE_EPOCHS = 35
    VAE_PATH = "checkpoints/fashion_vae.pt"

    LATENT_MODEL_PATH = "checkpoints/latent_model.pt"
    LATENT_EMA_PATH = "checkpoints/latent_model.ema.pt"

    N_SAMPLES = 80
    GUIDANCE_WEIGHT = 0.0

    BASE_CHANNELS = 64
    CHANNEL_MULT = [1, 2, 2]
    NUM_RES_BLOCKS = 2
    TIME_EMBED_DIM = BASE_CHANNELS * 4
    CONTEXT_DIM = TIME_EMBED_DIM

config = Config()

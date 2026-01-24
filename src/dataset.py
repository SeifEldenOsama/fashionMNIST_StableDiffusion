from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from .config import config

def get_dataloader(train: bool = True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.FashionMNIST(
        root=config.DATA_DIR,
        train=train,
        download=True,
        transform=transform
    )
    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=train,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

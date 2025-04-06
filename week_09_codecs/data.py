import torch
from lightning import LightningDataModule
from torch.utils.data import random_split, DataLoader

from torchvision.datasets import MNIST
from torchvision import transforms


class MNISTDataModule(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

        if stage == "predict":
            self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=32, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=32, num_workers=8)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=32, num_workers=8)
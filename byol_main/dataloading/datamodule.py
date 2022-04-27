import torch
import torchvision
import numpy as np
import torchvision.transforms as T
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch.utils.data as D
from galaxy_mnist import GalaxyMNIST

from torchvision.datasets import ImageFolder, STL10, CIFAR10

from paths import Path_Handler
from dataloading.utils import compute_mu_sig, _get_imagenet_norms
from dataloading.transforms import MultiView, SimpleView, ReduceView

paths = Path_Handler()
path_dict = paths._dict()


class imagenette_DataModule(pl.LightningDataModule):
    def __init__(self, config, path=path_dict["imagenette"]):
        super().__init__()
        self.path = path
        self.config = config

        self.mu = (0.485, 0.456, 0.406)
        self.sig = (0.229, 0.224, 0.225)

        self.T_train = MultiView(config, mu=self.mu, sig=self.sig)
        self.T_test = SimpleView(config, rotate=False, mu=self.mu, sig=self.sig)

        self.data = {}

    def prepare_data(self):
        return

    def setup(self):
        self.data["train"] = ImageFolder(self.path / "train", transform=self.T_train)
        self.data["labelled"] = ImageFolder(self.path / "train", transform=self.T_test)

        self.data["val"] = ImageFolder(self.path / "val", transform=self.T_test)

    def train_dataloader(self):
        # Batch all data together
        batch_size = self.config["batch_size"]
        loader = DataLoader(self.data["train"], batch_size, shuffle=True)

        return loader

    def val_dataloader(self):
        loader = DataLoader(self.data["val"], 1000)
        return loader

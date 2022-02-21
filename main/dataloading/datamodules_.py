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
from dataloading.datasets import MB_nohybrids, RGZ20k
from dataloading.utils import compute_mu_sig, rgz_cut, _get_imagenet_norms
from dataloading.transforms import MultiView, SimpleView, ReduceView

paths = Path_Handler()
path_dict = paths._dict()


class mb_DataModule(pl.LightningDataModule):
    def __init__(self, config, path=path_dict["data"]):
        super().__init__()
        self.path = path
        self.config = config

        self.T_train = MultiView(config)
        self.T_test = SimpleView(config, rotate=False)

    def prepare_data(self):
        MB_nohybrids(self.path, train=False, download=True)
        MB_nohybrids(self.path, train=True, download=True)
        RGZ20k(self.path, train=True, download=True)
        self.data = {}

    def setup(self):
        D_train = self._train_set(self.T_train)

        # Calculate mean and std of data
        mu, sig = compute_mu_sig(D_train, batch_size=1000, multiview=True)
        self.mu, self.sig = mu, sig

        # Define transforms with calculated values
        self.T_train.update_normalization((mu,), (sig,))
        self.T_test.update_normalization((mu,), (sig,))

        # Re-initialise dataset with new mu and sig values
        self.data["train"] = self._train_set(self.T_train)
        # Initialise individual datasets with test transform (for evaluation)
        self.data["val"] = MB_nohybrids(self.path, train=False, transform=self.T_test)
        self.data["test"] = MB_nohybrids(self.path, train=False, transform=self.T_test)
        self.data["l"] = MB_nohybrids(self.path, train=True, transform=self.T_test)
        self.data["u"] = RGZ20k(self.path, train=True, transform=self.T_test)

    def train_dataloader(self):
        # Batch all data together
        batch_size = self.config["batch_size"]
        loader = DataLoader(self.data["train"], batch_size, shuffle=True)

        return loader

    def val_dataloader(self):
        loader = DataLoader(self.data["val"], 1000)
        return loader

    ####### HELPER FUNCTIONS ########

    def _train_set(self, transform):
        """Load MiraBest & RGZ datasets, cut MiraBest by angular size, remove duplicates from RGZ and concatenate the two"""
        D_rgz = RGZ20k(self.path, train=True, transform=transform)
        D_rgz = rgz_cut(D_rgz, self.config["cut_threshold"], mb_cut=True)
        D_mb = MB_nohybrids(self.path, train=True, transform=transform)

        # Concatenate datasets
        return D.ConcatDataset([D_rgz, D_mb])


class mb_DataModule_eval(pl.LightningDataModule):
    def __init__(self, encoder, config, path=path_dict["data"]):
        super().__init__()
        self.path = path
        self.config = config

        mu = (self.config["data"]["mu"],)
        sig = (self.config["data"]["sig"],)
        self.T_train = ReduceView(encoder, config, rot=True, mu=mu, sig=sig)
        self.T_test = ReduceView(encoder, config, rot=False, mu=mu, sig=sig)

        self.data = {}

    def prepare_data(self):
        MB_nohybrids(self.path, train=False, download=True)
        MB_nohybrids(self.path, train=True, download=True)
        RGZ20k(self.path, train=True, download=True)

    def setup(self, stage=None):
        # D_train = self.cut_and_cat()

        # Calculate mean and std of data
        D_train = MB_nohybrids(self.path, train=True, transform=self.T_train)
        mu, sig = compute_mu_sig(D_train)
        self.mu, self.sig = mu, sig

        # Define transforms with calculated values
        self.T_train.update_normalization((mu,), (sig,))
        self.T_test.update_normalization((mu,), (sig,))

        # Initialise individual datasets
        self.data["train"] = MB_nohybrids(self.path, train=True, transform=self.T_train)
        self.data["val"] = MB_nohybrids(self.path, train=False, transform=self.T_test)
        self.data["test"] = MB_nohybrids(self.path, train=False, transform=self.T_test)
        self.data["mb"] = MB_nohybrids(self.path, train=True, transform=self.T_test)
        self.data["rgz"] = RGZ20k(self.path, train=True, transform=self.T_test)

    def train_dataloader(self):
        # Batch only labelled data
        batch_size = self.config["linear"]["batch_size"]
        loader = DataLoader(self.data["train"], batch_size, shuffle=True)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.data["test"], len(self.data["test"]))
        return loader

    def test_dataloader(self):
        loader = DataLoader(self.data["test"], len(self.data["test"]))
        return loader


class imagenette_DataModule(pl.LightningDataModule):
    def __init__(self, config, path=path_dict["imagenette"]):
        super().__init__()
        self.path = path
        self.config = config

        self.mu = (0.485, 0.456, 0.406)
        self.sig = (0.229, 0.224, 0.225)

        self.T_train = MultiView(config, mu=self.mu, sig=self.sig)
        self.T_test = SimpleView(config, mu=self.mu, sig=self.sig)

        self.data = {}

    def prepare_data(self):
        return

    def setup(self):
        self.data["train"] = ImageFolder(self.path / "train", transform=self.T_train)
        self.data["l"] = ImageFolder(self.path / "train", transform=self.T_test)

        self.data["val"] = ImageFolder(self.path / "val", transform=self.T_test)

    def train_dataloader(self):
        # Batch all data together
        batch_size = self.config["batch_size"]
        loader = DataLoader(self.data["train"], batch_size, shuffle=True)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.data["val"], 1000)
        return loader


class imagenette_DataModule_eval(pl.LightningDataModule):
    def __init__(self, encoder, config, path=path_dict["imagenette"]):
        super().__init__()
        self.path = path
        self.config = config

        norms = _get_imagenet_norms()
        self.T_train = ReduceView(encoder, config, mu=norms["mean"], sig=norms["sig"])
        self.T_test = ReduceView(encoder, config, mu=norms["mean"], sig=norms["sig"])

        self.data = {}

    def prepare_data(self):
        return

    def setup(self):
        # Initialise individual datasets with identity transform (for evaluation)
        D_train = ImageFolder(self.path / "train", transform=self.T_train)
        self.update_transforms(D_train)
        self.data["train"] = ImageFolder(self.path / "train", transform=self.T_train)
        self.data["val"] = ImageFolder(self.path / "val", transform=self.T_test)

    def train_dataloader(self):
        # Batch only labelled data
        batch_size = self.config["linear"]["batch_size"]
        loader = DataLoader(self.data["train"], batch_size, shuffle=True)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.data["val"], len(self.data["val"]))
        return loader

    def update_transforms(self, D_train):
        if not self.config["debug"]:
            mu, sig = compute_mu_sig(D_train)
            self.mu, self.sig = mu, sig

            # Define transforms with calculated values
            self.T_train.update_normalization((mu,), (sig,))
            self.T_test.update_normalization((mu,), (sig,))


class stl10_DataModule(pl.LightningDataModule):
    def __init__(self, config, path=path_dict["stl10"]):
        super().__init__()
        self.path = path
        self.config = config

        self.mu = (0.485, 0.456, 0.406)
        self.sig = (0.229, 0.224, 0.225)

        self.T_train = MultiView(config, mu=self.mu, sig=self.sig)
        self.T_test = SimpleView(config, rotate=False, mu=self.mu, sig=self.sig)

        self.data = {}

    def prepare_data(self):
        STL10(root=self.path, split="train+unlabeled", download=True)
        STL10(root=self.path, split="test", download=True)

    def setup(self):
        path = self.path
        T_train = self.T_train
        T_test = self.T_test

        self.data["train"] = STL10(
            root=path, split="train+unlabeled", transform=T_train
        )
        self.data["u"] = STL10(root=path, split="unlabeled", transform=T_test)
        self.data["l"] = STL10(root=path, split="train", transform=T_test)
        self.data["val"] = STL10(root=path, split="test", transform=T_test)
        self.data["test"] = STL10(root=path, split="test", transform=T_test)

    def train_dataloader(self):
        # Batch all data together
        batch_size = self.config["batch_size"]
        loader = DataLoader(self.data["train"], batch_size, shuffle=True)
        return loader

    def val_dataloader(self):
        batch_size = 1000
        loader = DataLoader(self.data["val"], batch_size)
        return loader


class stl10_DataModule_eval(pl.LightningDataModule):
    def __init__(self, encoder, config, path=path_dict["stl10"]):
        super().__init__()
        self.path = path
        self.config = config

        mu = (0.485, 0.456, 0.406)
        sig = (0.229, 0.224, 0.225)

        self.T_train = ReduceView(encoder, config, mu=mu, sig=sig)
        self.T_test = ReduceView(encoder, config, mu=mu, sig=sig)

        self.data = {}

    def prepare_data(self):
        return

    def setup(self):
        # Initialise individual datasets with identity transform (for evaluation)

        path = self.path
        T_train = self.T_train
        T_test = self.T_test

        D_train = STL10(root=path, split="train+unlabeled", transform=T_train)

        if not self.config["debug"]:
            self.mu, self.sig = compute_mu_sig(D_train)

            # Define transforms with calculated values
            self.T_train.update_normalization((self.mu,), (self.sig,))
            self.T_test.update_normalization((self.mu,), (self.sig,))

        self.data["train"] = STL10(root=path, split="train", transform=T_train)
        self.data["u"] = STL10(root=path, split="unlabeled", transform=T_test)
        self.data["val"] = STL10(root=path, split="test", transform=T_test)
        self.data["test"] = STL10(root=path, split="test", transform=T_test)

    def train_dataloader(self):
        # Batch only labelled data
        batch_size = self.config["linear"]["batch_size"]
        loader = DataLoader(self.data["train"], batch_size, shuffle=True)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.data["val"], len(self.data["test"]))
        return loader

    def test_dataloader(self):
        loader = DataLoader(self.data["test"], len(self.data["test"]))
        return loader


class cifar10_DataModule(pl.LightningDataModule):
    def __init__(self, config, path=path_dict["cifar10"]):
        super().__init__()
        self.path = path
        self.config = config

        self.mu = (0.49139968, 0.48215827, 0.44653124)
        self.sig = (0.24703233, 0.24348505, 0.26158768)

        self.T_train = MultiView(config, mu=self.mu, sig=self.sig)

        self.data = {}

    def prepare_data(self):
        CIFAR10(root=self.path, train=True, download=True)
        CIFAR10(root=self.path, train=False, download=True)

    def setup(self):
        path = self.path
        T_train = self.T_train

        self.data["train"] = CIFAR10(root=path, train=True, transform=T_train)

    def train_dataloader(self):
        # Batch all data together
        batch_size = self.config["batch_size"]
        loader = DataLoader(self.data["train"], batch_size, shuffle=True)

        return loader


class cifar10_DataModule_eval(pl.LightningDataModule):
    def __init__(self, encoder, config, path=path_dict["cifar10"]):
        super().__init__()
        self.path = path
        self.config = config

        mu = (0.49139968, 0.48215827, 0.44653124)
        sig = (0.24703233, 0.24348505, 0.26158768)

        self.T_train = ReduceView(encoder, config, mu=mu, sig=sig)
        self.T_test = ReduceView(encoder, config, mu=mu, sig=sig)

        self.data = {}

    def prepare_data(self):
        return

    def setup(self):
        # Initialise individual datasets with identity transform (for evaluation)
        path = self.path
        T_train = self.T_train
        T_test = self.T_test

        D_train = CIFAR10(root=path, train=True, transform=T_train)
        mu, sig = compute_mu_sig(D_train, batch_size=1000)
        self.mu, self.sig = mu, sig

        # Define transforms with calculated values
        self.T_train.update_normalization((mu,), (sig,))
        self.T_test.update_normalization((mu,), (sig,))

        self.data["train"] = CIFAR10(root=path, train=True, transform=T_train)
        self.data["l"] = CIFAR10(root=path, train=True, transform=T_test)
        self.data["u"] = CIFAR10(root=path, train=True, transform=T_test)
        self.data["val"] = CIFAR10(root=path, train=False, transform=T_test)
        self.data["test"] = CIFAR10(root=path, train=False, transform=T_test)

    def train_dataloader(self):
        # Batch only labelled data
        batch_size = self.config["linear"]["batch_size"]
        loader = DataLoader(self.data["train"], batch_size, shuffle=True)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.data["val"], len(self.data["test"]))
        return loader

    def test_dataloader(self):
        loader = DataLoader(self.data["test"], len(self.data["test"]))
        return loader

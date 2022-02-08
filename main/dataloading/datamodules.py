import torch
import torchvision
import numpy as np
import torchvision.transforms as T
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch.utils.data as D
from galaxy_mnist import GalaxyMNIST

from torchvision.datasets import ImageFolder

from paths import Path_Handler
from dataloading.datasets import MB_nohybrids, RGZ20k, MiraBest_full
from dataloading.utils import size_cut, compute_mu_sig, mb_cut, rgz_cut, rgz_cut
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
        mu, sig = compute_mu_sig(D_train, batch_size=1000)
        self.mu, self.sig = mu, sig

        # Define transforms with calculated values
        self.T_train.update_normalization((mu,), (sig,))
        self.T_test.update_normalization((mu,), (sig,))

        # Re-initialise dataset with new mu and sig values
        self.data["train"] = self._train_set(self.T_train)
        # Initialise individual datasets with test transform (for evaluation)
        self.data["bank"] = MB_nohybrids(self.path, train=True, transform=self.T_test)
        self.data["val"] = MB_nohybrids(self.path, train=False, transform=self.T_test)
        self.data["test"] = MB_nohybrids(self.path, train=False, transform=self.T_test)
        self.data["mb"] = MB_nohybrids(self.path, train=True, transform=self.T_test)
        self.data["rgz"] = RGZ20k(self.path, train=True, transform=self.T_test)

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

        self.mu = (self.config["data"]["mu"],)
        self.sig = (self.config["data"]["sig"],)
        self.T_train = ReduceView(
            encoder, config, rotate=True, mu=self.mu, sig=self.sig
        )
        self.T_test = ReduceView(encoder, config, rotate=True, mu=self.mu, sig=self.sig)

        self.data = {}

    def prepare_data(self):
        MB_nohybrids(self.path, train=False, download=True)
        MB_nohybrids(self.path, train=True, download=True)
        RGZ20k(self.path, train=True, download=True)

    def setup(self):
        # Initialise individual datasets with identity transform (for evaluation)
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
        self.T_test = SimpleView(config, rotate=False, mu=self.mu, sig=self.sig)

        self.data = {}

    def prepare_data(self):
        return

    def setup(self):
        self.data["train"] = ImageFolder(self.path / "train", transform=self.T_train)
        self.data["bank"] = ImageFolder(self.path / "train", transform=self.T_test)

        self.data["val"] = ImageFolder(self.path / "val", transform=self.T_test)

    def train_dataloader(self):
        # Batch all data together
        batch_size = self.config["batch_size"]
        loader = DataLoader(self.data["train"], batch_size, shuffle=True)

        return loader


class imagenette_DataModule_eval(pl.LightningDataModule):
    def __init__(self, encoder, config, path=path_dict["imagenette"]):
        super().__init__()
        self.path = path
        self.config = config

        self.mu = (0.485, 0.456, 0.406)
        self.sig = (0.229, 0.224, 0.225)

        self.T_train = ReduceView(encoder, config, mu=self.mu, sig=self.sig)
        self.T_test = ReduceView(encoder, config, mu=self.mu, sig=self.sig)

        self.data = {}

    def prepare_data(self):
        return

    def setup(self):
        # Initialise individual datasets with identity transform (for evaluation)
        self.data["train"] = ImageFolder(self.path / "train", transform=self.T_train)
        self.data["val"] = ImageFolder(self.path / "val", transform=self.T_test)

    def train_dataloader(self):
        # Batch only labelled data
        batch_size = self.config["linear"]["batch_size"]
        loader = DataLoader(self.data["train"], batch_size, shuffle=True)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.data["val"], len(self.data["test"]))
        return loader

    def test_dataloader(self):
        return


class GZMNIST_DataModule(pl.LightningDataModule):
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
        mu, sig = compute_mu_sig(D_train, batch_size=1000)
        self.mu, self.sig = mu, sig

        # Define transforms with calculated values
        self.T_train.update_normalization((mu,), (sig,))
        self.T_test.update_normalization((mu,), (sig,))

        # Re-initialise dataset with new mu and sig values
        self.data["train"] = self._train_set(self.T_train)

        # Initialise individual datasets with identity transform (for evaluation)
        self.data["test"] = MB_nohybrids(self.path, train=False, transform=self.T_test)
        self.data["mb"] = MB_nohybrids(self.path, train=True, transform=self.T_test)
        self.data["rgz"] = RGZ20k(self.path, train=True, transform=self.T_test)

    def train_dataloader(self):
        # Batch all data together
        batch_size = self.config["batch_size"]
        loader = DataLoader(self.data["train"], batch_size, shuffle=True)

        return loader

    ####### HELPER FUNCTIONS ########

    def _train_set(self, transform):
        """Load MiraBest & RGZ datasets, cut MiraBest by angular size, remove duplicates from RGZ and concatenate the two"""
        D_rgz = RGZ20k(self.path, train=True, transform=transform)
        D_rgz = rgz_cut(D_rgz, self.config["cut_threshold"], mb_cut=True)
        D_mb = MB_nohybrids(self.path, train=True, transform=transform)

        # Concatenate datasets
        return D.ConcatDataset([D_rgz, D_mb])


class GZMNIST_DataModule_eval(pl.LightningDataModule):
    def __init__(self, encoder, config, path=path_dict["data"]):
        super().__init__()
        self.path = path
        self.config = config

        self.mu = (self.config["data"]["mu"],)
        self.sig = (self.config["data"]["sig"],)
        self.T_train = ReduceView(encoder, config, train=True, mu=self.mu, sig=self.sig)
        self.T_test = ReduceView(encoder, config, train=False, mu=self.mu, sig=self.sig)

        self.data = {}

    def prepare_data(self):
        MB_nohybrids(self.path, train=False, download=True)
        MB_nohybrids(self.path, train=True, download=True)
        RGZ20k(self.path, train=True, download=True)

    def setup(self):
        # Initialise individual datasets with identity transform (for evaluation)
        self.data["train"] = MB_nohybrids(self.path, train=True, transform=self.T_train)
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

import torch
import torchvision
import numpy as np
import torchvision.transforms as T
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch.utils.data as D

from paths import Path_Handler
from dataloading.datasets import MB_nohybrids, RGZ20k, MiraBest_full
from dataloading.utils import size_cut, compute_mu_sig, mb_cut, rgz_cut, rgz_cut
from dataloading.transforms import MultiView, Identity, ReduceView

paths = Path_Handler()
path_dict = paths._dict()


class mbDataModule(pl.LightningDataModule):
    def __init__(self, config, path=path_dict["data"]):
        super().__init__()
        self.path = path
        self.config = config

        # Define different transforms for different algorithms
        train_transforms = {
            "byol": MultiView(config, n_views=2),
            "pca": Identity(config, train=True),
        }

        self.T_train = train_transforms[self.config["type"]]
        self.T_test = Identity(config, train=False)

        # self.save_hyperparameters(ignore=["T_test", "T_train"])

    def prepare_data(self):
        MB_nohybrids(self.path, train=False, download=True)
        MB_nohybrids(self.path, train=True, download=True)
        RGZ20k(self.path, train=True, download=True)
        self.data = {}

    def setup(self, stage=None):
        D_train = self.cut_and_cat(self.T_train)

        # Calculate mean and std of data
        mu, sig = compute_mu_sig(D_train, batch_size=1000)
        self.mu, self.sig = mu, sig

        # Define transforms with calculated values
        self.T_train.update_normalization(mu, sig)

        # Re-initialise dataset with new mu and sig values
        self.data["train"] = self.cut_and_cat(self.T_train)

        # Initialise individual datasets with identity transform (for evaluation)
        self.data["test"] = MB_nohybrids(self.path, train=False, transform=self.T_test)
        self.data["mb"] = MB_nohybrids(self.path, train=True, transform=self.T_test)
        self.data["rgz"] = RGZ20k(self.path, train=True, transform=self.T_test)

    def train_dataloader(self):
        # Batch all data together
        batch_size = self.config["batch_size"]
        loader = DataLoader(self.data["train"], batch_size, shuffle=True)

        return loader

    def val_dataloader(self):
        loader = DataLoader(self.data["test"], len(self.data["test"]))
        return loader

    def test_dataloader(self):
        loader = DataLoader(self.data["test"], len(self.data["test"]))
        return loader

    #################################
    ####### HELPER FUNCTIONS ########
    #################################
    def cut_and_cat(self, transform):
        """Load MiraBest & RGZ datasets, cut MiraBest by angular size, remove duplicates from RGZ and concatenate the two"""
        D_rgz = RGZ20k(self.path, train=True, transform=transform)
        D_rgz = rgz_cut(D_rgz, self.config["cut_threshold"], mb_cut=True)
        D_mb = MB_nohybrids(self.path, train=True, transform=transform)

        # Concatenate datasets
        return D.ConcatDataset([D_rgz, D_mb])


class reduce_mbDataModule(pl.LightningDataModule):
    def __init__(self, encoder, config, path=path_dict["data"]):
        super().__init__()
        self.path = path
        self.config = config
        self.aug = ReduceView(encoder, config)
        self.data = {}
        # self.identity = Identity()

    def prepare_data(self):
        MB_nohybrids(self.path, train=False, download=True)
        MB_nohybrids(self.path, train=True, download=True)
        RGZ20k(self.path, train=True, download=True)

    def setup(self, stage=None):
        # D_train = self.cut_and_cat()

        # Calculate mean and std of data
        D_train = MB_nohybrids(self.path, train=True, transform=self.aug)
        mu, sig = compute_mu_sig(D_train)
        self.mu, self.sig = mu, sig

        # Define transforms with calculated values
        self.aug.update_normalization(mu, sig)
        # identity = Identity(self.config["center_crop_size"], mu=mu, sig=sig)

        # Re-initialise dataset with new mu and sig values
        # self.data["train"] = self.cut_and_cat()
        self.data["train"] = MB_nohybrids(self.path, train=True, transform=self.aug)

        # Initialise individual datasets with identity transform (for evaluation)
        self.data["test"] = MB_nohybrids(self.path, train=False, transform=self.aug)
        self.data["mb"] = MB_nohybrids(self.path, train=True, transform=self.aug)
        self.data["rgz"] = RGZ20k(self.path, train=True, transform=self.aug)

    def train_dataloader(self):
        # Batch only labelled data
        loader = DataLoader(
            self.data["train"], self.config["linear"]["batch_size"], shuffle=True
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.data["test"], len(self.data["test"]))
        return loader

    def test_dataloader(self):
        loader = DataLoader(self.data["test"], len(self.data["test"]))
        return loader

    #################################
    ####### HELPER FUNCTIONS ########
    #################################

    def cut_and_cat(self):
        # Load and cut data-sets
        D_rgz = RGZ20k(self.path, train=True, transform=self.aug)
        size_cut(self.config["cut_threshold"], D_rgz)
        mb_cut(D_rgz)
        D_mb = MB_nohybrids(self.path, train=True, transform=self.aug)

        # Concatenate datasets
        return D.ConcatDataset([D_rgz, D_mb])

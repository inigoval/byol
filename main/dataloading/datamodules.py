import torch
import torchvision
import numpy as np
import torchvision.transforms as T
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch.utils.data as D

from paths import Path_Handler
from dataloading.datasets import MB_nohybrids, RGZ20k, MiraBest_full
from dataloading.utils import size_cut, compute_mu_sig, mb_cut
from dataloading.transforms import MultiViewTransform, IdentityTransform

paths = Path_Handler()
path_dict = paths._dict()


class mbDataModule(pl.LightningDataModule):
    def __init__(self, config, path=path_dict["data"]):
        super().__init__()
        self.path = path
        self.config = config
        self.hyperparams = {}
        self.view_transform = MultiViewTransform(config)

    def prepare_data(self):
        MB_nohybrids(self.path, train=False, download=True)
        MB_nohybrids(self.path, train=True, download=True)
        RGZ20k(self.path, train=True, download=True)
        self.data = {}

    def setup(self, stage=None):
        D_train = self.cut_and_cat()

        # Calculate mean and std of data
        mu, sig = compute_mu_sig(D_train)
        self.mu, self.sig = mu, sig

        # Define transforms with calculated values
        self.view_transform.update_normalization(mu, sig)
        identity = IdentityTransform(self.config["center_crop_size"], mu=mu, sig=sig)

        # Re-initialise dataset with new mu and sig values
        self.data["train"] = self.cut_and_cat()

        # Initialise individual datasets with identity transform (for evaluation)
        self.data["test"] = MB_nohybrids(self.path, train=False, transform=identity)
        self.data["mb"] = MB_nohybrids(self.path, train=True, transform=identity)
        self.data["rgz"] = RGZ20k(self.path, train=True, transform=identity)

    def train_dataloader(self):
        """Batch unlabelled and labelled data together"""

        loader = DataLoader(self.data["train"], self.config["batch_size"], shuffle=True)
        return loader

    def test_dataloader(self):
        loader = DataLoader(self.data["test"], len(self.data["test"]))
        return loader

    #################################
    ####### HELPER FUNCTIONS ########
    #################################

    def cut_and_cat(self):
        # Load and cut data-sets
        D_rgz = RGZ20k(self.path, train=True, transform=self.view_transform)
        size_cut(self.config["cut_threshold"], D_rgz)
        mb_cut(D_rgz)
        D_mb = MB_nohybrids(self.path, train=True, transform=self.view_transform)

        # Concatenate datasets
        return D.ConcatDataset([D_rgz, D_mb])

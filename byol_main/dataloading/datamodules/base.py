import logging

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from byol_main.paths import Path_Handler
from byol_main.dataloading.utils import compute_mu_sig_features, compute_mu_sig_images
from byol_main.dataloading.transforms import ReduceView, MultiView, SimpleView, SupervisedView


class Base_DataModule(pl.LightningDataModule):
    def __init__(self, config, mu, sig):
        super().__init__()

        # override default paths via config if desired
        paths = Path_Handler(**config.get("paths_to_override", {}))
        path_dict = paths._dict()
        self.path = path_dict[config["dataset"]]

        self.config = config

        self.mu, self.sig = mu, sig

        self.T_train = MultiView(config, mu=self.mu, sig=self.sig)
        self.T_test = SimpleView(config, mu=self.mu, sig=self.sig)

        self.data = {}

    def prepare_data(self):
        return

    def train_dataloader(self):
        loader = DataLoader(
            self.data["train"],
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["num_workers"],
            # num_workers=1,
            prefetch_factor=self.config["prefetch_factor"],
            pin_memory=self.config["pin_memory"],
            persistent_workers=self.config["persistent_workers"],
        )
        return loader

    def update_transforms(self, D_train):
        # if mu (and sig, implicitly) has been explicitly set, trust it is correct
        if self.mu != ((0,)):
            logging.info(
                "Skipping mu/sig calculation - mu, sig explicitly set to {}, {}".format(
                    self.mu, self.sig
                )
            )
        elif self.config["debug"]:
            logging.info("Skipping mu/sig calculation - debug mode")

        else:
            original_T_train_views = self.T_train.n_views
            # temporarily set one view to calculate mu, sig easily
            self.T_train.n_views = 1

            mu, sig = compute_mu_sig_images(D_train, batch_size=1000)
            self.mu, self.sig = mu, sig
            logging.info("mu, sig re-calculated as set to {}, {}".format(self.mu, self.sig))

            # Define transforms with calculated values
            self.T_train.update_normalization(mu, sig)
            self.T_test.update_normalization(mu, sig)

            # restore to normal 2-view mode (assumed the only sensible option)
            self.T_train.n_views = original_T_train_views


class Base_DataModule_Eval(pl.LightningDataModule):
    """mu sig initially passed are for pre-normalization in pixel space, update_normalization updates normalizatio in representation space"""

    def __init__(self, encoder, config):
        super().__init__()
        self.config = config
        paths = Path_Handler()
        path_dict = paths._dict()
        self.path = path_dict[config["dataset"]]

        self.T_train = ReduceView(encoder, config, train=True)
        self.T_test = ReduceView(encoder, config, train=False)

        # hardcoded default for now, currently always 2 except within mu/sig calculation
        self.T_train.n_views = 2

        self.data = {}

    def prepare_data(self):
        return

    def train_dataloader(self):
        # Batch only labelled data
        n_workers = self.config["num_workers"]
        batch_size = self.config["linear"]["batch_size"]
        loader = DataLoader(
            self.data["train"],
            batch_size,
            shuffle=True,
            num_workers=n_workers,
            prefetch_factor=20,
        )
        return loader

    def val_dataloader(self):
        n_workers = self.config["num_workers"]
        loader = DataLoader(
            self.data["val"],
            batch_size=self.config["data"]["val_batch_size"],
            num_workers=n_workers,
            prefetch_factor=20,
        )
        return loader

    def test_dataloader(self):
        n_workers = self.config["num_workers"]
        loader = DataLoader(
            self.data["test"],
            batch_size=self.config["data"]["val_batch_size"],
            num_workers=n_workers,
            prefetch_factor=20,
            persistent_workers=self.config["persistent_workers"],
        )
        return loader

    def update_transforms(self, D_train):
        if not self.config["debug"]:
            mu, sig = compute_mu_sig_features(D_train)
            self.mu, self.sig = mu, sig
            logging.info("Set mu {:3.2f}, sig {:3.2f}".format(self.mu, self.sig))

            # Define transforms with calculated values
            self.T_train.update_normalization(mu, sig)
            self.T_test.update_normalization(mu, sig)


class Base_DataModule_Supervised(pl.LightningDataModule):
    def __init__(self, config, mu, sig):
        super().__init__()
        paths = Path_Handler()
        path_dict = paths._dict()
        self.path = path_dict[config["dataset"]]

        self.config = config

        self.mu, self.sig = mu, sig

        self.T_train = SupervisedView(config, mu=self.mu, sig=self.sig)
        self.T_test = SimpleView(config, mu=self.mu, sig=self.sig)

        self.data = {}

    def prepare_data(self):
        return

    def train_dataloader(self):
        # Batch all data together
        batch_size = self.config["batch_size"]
        n_workers = self.config["num_workers"]
        loader = DataLoader(
            self.data["train"],
            batch_size,
            shuffle=True,
            num_workers=n_workers,
            prefetch_factor=20,
        )
        return loader

    def val_dataloader(self):
        n_workers = self.config["num_workers"]
        loader = DataLoader(
            self.data["val"],
            batch_size=self.config["data"]["val_batch_size"],
            num_workers=n_workers,
            prefetch_factor=20,
            persistent_workers=self.config["persistent_workers"],
        )
        return loader

    def test_dataloader(self):
        n_workers = self.config["num_workers"]
        loader = DataLoader(
            self.data["test"],
            batch_size=self.config["data"]["val_batch_size"],
            num_workers=n_workers,
            prefetch_factor=20,
            persistent_workers=self.config["persistent_workers"],
        )
        return loader

    def update_transforms(self, D_train):
        if not self.config["debug"]:
            mu, sig = compute_mu_sig_images(D_train, batch_size=1000)
            self.mu, self.sig = mu, sig

            # Define transforms with calculated values
            self.T_train.update_normalization(mu, sig)
            self.T_test.update_normalization(mu, sig)

        self.T_train.n_views = 2

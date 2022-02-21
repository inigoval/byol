import pytorch_lightning as pl
from torch.utils.data import DataLoader

from paths import Path_Handler
from dataloading.utils import compute_mu_sig_features, compute_mu_sig_images
from dataloading.transforms import ReduceView, MultiView, SimpleView


class Base_DataModule(pl.LightningDataModule):
    def __init__(self, config, mu, sig):
        super().__init__()

        paths = Path_Handler()
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
        # Batch all data together
        batch_size = self.config["batch_size"]
        n_workers = self.config["data"]["num_workers"]
        loader = DataLoader(self.data["train"], batch_size, shuffle=True, num_workers=n_workers)
        return loader

    def val_dataloader(self):
        n_workers = self.config["data"]["num_workers"]
        loader = DataLoader(self.data["val"], 1000, num_workers=n_workers)
        return loader

    def test_dataloader(self):
        n_workers = self.config["data"]["num_workers"]
        loader = DataLoader(self.data["test"], 1000, num_workers=n_workers)
        return loader

    def update_transforms(self, D_train):
        if not self.config["debug"]:
            mu, sig = compute_mu_sig_images(D_train)
            self.mu, self.sig = mu, sig

            # Define transforms with calculated values
            self.T_train.update_normalization(mu, sig)
            self.T_test.update_normalization(mu, sig)

        self.T_train.n_views = 2


class Base_DataModule_Eval(pl.LightningDataModule):
    """mu sig initially passed are for pre-normalization in pixel space, update_normalization updates normalizatio in representation space"""

    def __init__(self, encoder, config, mu, sig):
        super().__init__()
        self.config = config
        paths = Path_Handler()
        path_dict = paths._dict()
        self.path = path_dict[config["dataset"]]

        self.T_train = ReduceView(encoder, config, mu=mu, sig=sig)
        self.T_test = ReduceView(encoder, config, mu=mu, sig=sig)

        self.data = {}

    def prepare_data(self):
        return

    def train_dataloader(self):
        # Batch only labelled data
        n_workers = self.config["data"]["num_workers"]
        batch_size = self.config["linear"]["batch_size"]
        loader = DataLoader(self.data["train"], batch_size, shuffle=True, num_workers=n_workers)
        return loader

    def val_dataloader(self):
        n_workers = self.config["data"]["num_workers"]
        loader = DataLoader(self.data["val"], 1000, num_workers=n_workers)
        return loader

    def test_dataloader(self):
        n_workers = self.config["data"]["num_workers"]
        loader = DataLoader(self.data["test"], 1000, num_workers=n_workers)
        return loader

    def update_transforms(self, D_train):
        if not self.config["debug"]:
            mu, sig = compute_mu_sig_features(D_train)
            self.mu, self.sig = mu, sig

            # Define transforms with calculated values
            self.T_train.update_normalization(mu, sig)
            self.T_test.update_normalization(mu, sig)

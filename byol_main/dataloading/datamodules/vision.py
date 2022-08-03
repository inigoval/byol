import logging
import pytorch_lightning as pl

from torch.utils.data import DataLoader

from byol_main.dataloading.utils import compute_mu_sig_features, compute_mu_sig_images
from dataloading.utils import _get_imagenet_norms
from torchvision.datasets import STL10
from byol_main.paths import Path_Handler
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
            batch_size=self.config["data"]["pretrain_batch_size"],
            shuffle=True,
            num_workers=self.config["num_workers"],
            # num_workers=1,
            prefetch_factor=self.config["prefetch_factor"],
            pin_memory=self.config["pin_memory"],
            persistent_workers=self.config["persistent_workers"],
        )
        return loader

    def val_dataloader(self):
        return

    def test_dataloader(self):
        return

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


class STL10_DataModule(Base_DataModule):
    def __init__(self, config):
        norms = _get_imagenet_norms()
        super().__init__(config, **norms)

    def prepare_data(self):
        STL10(root=self.path, split="train+unlabeled", download=True)
        STL10(root=self.path, split="test", download=True)

    def setup(self):
        self.data["train"] = STL10(root=self.path, split="unlabeled", transform=self.T_train)

        # list of dictionaries
        self.data["eval"] = [
            {
                "name": "stl10_test",
                "train": STL10(root=self.path, split="train", transform=self.T_test),
                "val": STL10(root=self.path, split="test", transform=self.T_test),
                "test": STL10(root=self.path, split="test", transform=self.T_test),
            },
        ]

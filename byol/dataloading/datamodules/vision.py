import logging
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.datasets import STL10

from byol.dataloading.utils import compute_mu_sig_images
from byol.dataloading.utils import _get_imagenet_norms
from byol.paths import Path_Handler
from byol.dataloading.transforms import MultiView, SimpleView, MAEView, _train_view


class Base_DataModule(pl.LightningDataModule):
    def __init__(self, config, mu, sig):
        super().__init__()

        # override default paths via config if desired
        paths = Path_Handler(**config.get("paths_to_override", {}))
        path_dict = paths._dict()
        self.path = path_dict[config["dataset"]]

        self.config = config

        self.mu, self.sig = mu, sig

        self.T_train = _train_view(config)(config, mu=self.mu, sig=self.sig)

        self.T_test = SimpleView(config, mu=self.mu, sig=self.sig)

        self.data = {}

    def prepare_data(self):
        return

    def train_dataloader(self):
        loader = DataLoader(self.data["train"], **self.config["train_dataloader"])
        return loader

    def val_dataloader(self):
        loaders = [DataLoader(data, **self.config["val_dataloader"]) for _, data in self.data["val"]]
        return loaders

    def test_dataloader(self):
        loaders = [DataLoader(data, **self.config["val_dataloader"]) for _, data in self.data["test"]]
        return loaders

    def update_transforms(self, D_train):
        # if mu (and sig, implicitly) has been explicitly set, trust it is correct
        if self.mu != ((0,)):
            logging.info(
                "Skipping mu/sig calculation - mu, sig explicitly set to {}, {}".format(
                    self.mu, self.sig
                )
            )
        elif self.config["trainer"]["fast_dev_run"]:
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


class FineTuning_DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()

        # override default paths via config if desired
        paths = Path_Handler(**config.get("paths_to_override", {}))
        path_dict = paths._dict()
        self.path = path_dict[config["dataset"]]

        self.config = config

        self.mu, self.sig = config["data"]["mu"], config["data"]["sig"]

        self.data = {}

    def prepare_data(self):
        return

    def train_dataloader(self):
        loader = DataLoader(
            self.data["train"],
            batch_size=self.config["finetune"]["batch_size"],
            num_workers=8,
            prefetch_factor=20,
            shuffle=True,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.data["val"],
            batch_size=200,
            num_workers=8,
            prefetch_factor=20,
            shuffle=False,
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.data["test"],
            batch_size=200,
            num_workers=8,
            prefetch_factor=20,
            shuffle=False,
        )
        return loader


class STL10_DataModule(Base_DataModule):
    def __init__(self, config):
        norms = _get_imagenet_norms()
        super().__init__(config, **norms)

    def prepare_data(self):
        STL10(root=self.path, split="train+unlabeled", download=True)
        STL10(root=self.path, split="test", download=True)

    def setup(self):
        self.data["train"] = STL10(root=self.path, split="train+unlabeled", transform=self.T_train)

        # List of (name, train_dataset) tuples to evaluate linear layer
        self.data["val"] = [
            ("STL10_train", STL10(root=self.path, split="train", transform=self.T_test)),
            ("STL10_test", STL10(root=self.path, split="test", transform=self.T_test)),
        ]

        self.data["test"] = [
            ("STL10_test", STL10(root=self.path, split="test", transform=self.T_test)),
        ]

        # List of (name, train_dataset) tuples to train linear evaluation layer
        self.data["eval_train"] = [
            (
                "STl10_train",
                STL10(root=self.path, split="train", transform=self.T_test),
            ),
        ]


class Imagenette_DataModule(Base_DataModule):
    def __init__(self, config):
        norms = _get_imagenet_norms()
        super().__init__(config, **norms)

    def setup(self):
        self.data["train"] = ImageFolder(self.path / "train", transform=self.T_train)

        # List of (name, train_dataset) tuples to evaluate linear laye
        self.data["val"] = [
            ("imagenette_train", ImageFolder(self.path / "train", transform=self.T_test)),
            ("imagenette_val", ImageFolder(self.path / "val", transform=self.T_test)),
        ]

        self.data["test"] = [
            ("imagenette_val", ImageFolder(self.path / "val", transform=self.T_test)),
        ]

        # List of (name, train_dataset) tuples to train linear evaluation layer
        self.data["eval_train"] = [
            (
                "imagenette_train",
                ImageFolder(self.path / "train", transform=self.T_test),
                {"val": (0, 1), "test": (0,)},
            ),
        ]

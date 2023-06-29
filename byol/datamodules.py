import torch.nn as nn
import numpy as np
import torchvision.transforms as T
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from typing import Any, Dict, List, Tuple, Type, Optional
from torch.utils.data import Subset

from byol.utilities import rgz_cut, train_val_test_split
from byol.paths import Path_Handler
from byol.datasets import MBFRConfident, MBFRUncertain, RGZ108k, MBFRFull


class SimpleView(nn.Module):
    def __init__(self, config, mu=(0,), sig=(1,)):
        super().__init__()
        self.config = config

        augs = []
        # if config['dataset'] == 'gz2':  # is a tensor, needs to be a PIL to later call T.ToTensor
        #     augs.append(T.ToPILImage())

        if config["data"]["rotate"]:
            augs.append(T.RandomRotation(180))

        augs.append(T.Resize(config["data"]["input_height"]))

        if config["augmentations"]["center_crop_size"]:
            augs.append(T.CenterCrop(config["augmentations"]["center_crop_size"]))

        augs.append(T.ToTensor())
        self.view = T.Compose(augs)

        self.normalize = T.Normalize(mu, sig)
        # self.normalize = lambda x: x  # TODO temporarily disable normalisation (see also MultiView)

    def __call__(self, x):
        # Use rotation if training
        x = self.view(x)
        x = self.normalize(x)
        return x

    def update_normalization(self, mu, sig):
        self.normalize = T.Normalize(mu, sig)


class MultiView(nn.Module):
    def __init__(self, view, n_views=2):
        super().__init__()

        self.view = view
        self.n_views = n_views

    def __call__(self, x):
        if self.n_views > 1:
            views = []
            for _ in np.arange(self.n_views):
                view = self.view(x)
                views.append(view)
            return views
        else:
            return self.view(x)


def _blur_kernel(input_height):
    blur_kernel = int(input_height * 0.1)
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    return blur_kernel


class Base_DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers=0, prefetch_factor=20, pin_memory=False):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory

        self.data = {}

    def prepare_data(self):
        return

    def train_dataloader(self):
        loader = DataLoader(
            self.data["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
        )

        return loader

    def val_dataloader(self):
        loaders = [
            DataLoader(
                data,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                prefetch_factor=self.prefetch_factor,
                pin_memory=self.pin_memory,
            )
            for _, data in self.data["val"]
        ]
        return loaders

    def test_dataloader(self):
        loaders = [
            DataLoader(
                data,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                prefetch_factor=self.prefetch_factor,
                pin_memory=self.pin_memory,
            )
            for _, data in self.data["test"]
        ]
        return loaders


class RGZ_DataModule(Base_DataModule):
    def __init__(
        self,
        path,
        batch_size,
        center_crop,
        random_crop,
        s,
        p_blur,
        flip,
        rotation,
        cut_threshold=25,
        num_workers=0,
        prefetch_factor=20,
        pin_memory=False,
    ):
        super().__init__(batch_size, num_workers, prefetch_factor, pin_memory)

        self.path = path
        self.cut_threshold = cut_threshold

        self.mu = (0.008008896,)
        self.sig = (0.05303395,)

        # Train transforms
        augs = []
        if rotation:
            augs += [T.RandomRotation(180)]

        augs += [T.CenterCrop(center_crop), T.RandomResizedCrop(center_crop, scale=random_crop)]

        if flip:
            augs += [T.RandomHorizontalFlip(), T.RandomVerticalFlip()]

        augs += [
            T.RandomApply([T.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0)], p=0.8),
            T.ToTensor(),
            T.RandomApply([T.GaussianBlur(_blur_kernel(center_crop))], p=p_blur),
            T.Normalize(self.mu, self.sig),
        ]

        train_transform = T.Compose(augs)
        self.train_transform = MultiView(train_transform)

        # Test transforms
        self.test_transform = T.Compose(
            [
                T.CenterCrop(center_crop),
                T.ToTensor(),
                T.Normalize(self.mu, self.sig),
            ]
        )

    def prepare_data(self):
        MBFRFull(self.path, train=False, download=True)
        MBFRFull(self.path, train=True, download=True)
        RGZ108k(self.path, train=True, download=True)

    def setup(self, stage=None):
        # Compute mu and sigma value for data and update normalization constants
        # No longer needed
        # d_rgz = RGZ108k(self.path, train=True, transform=self.train_transform)
        # d_train = d_rgz
        # self.update_transforms(d_train)

        # Re-initialise dataset with new mu and sig values
        d_rgz = RGZ108k(
            self.path,
            train=True,
            transform=self.train_transform,
            remove_duplicates=True,
            cut_threshold=self.cut_threshold,
            mb_cut=True,
        )

        self.data["train"] = d_rgz

        self.data["val"] = [
            (
                "MB_conf_train",
                MBFRConfident(
                    self.path,
                    transform=self.test_transform,
                    aug_type="torchvision",
                    train=True,
                    test_size=None,
                ),
            ),
            (
                "MB_unc_train",
                MBFRUncertain(
                    self.path,
                    transform=self.test_transform,
                    aug_type="torchvision",
                    train=True,
                    test_size=None,
                ),
            ),
        ]

        self.data["test"] = [
            (
                "MB_conf_test",
                MBFRConfident(
                    self.path,
                    transform=self.test_transform,
                    aug_type="torchvision",
                    train=False,
                    test_size=None,
                ),
            ),
            (
                "MB_unc_test",
                MBFRUncertain(
                    self.path,
                    transform=self.test_transform,
                    aug_type="torchvision",
                    train=False,
                    test_size=None,
                ),
            ),
        ]

        # List of (name, train_dataset) tuples to train linear evaluation layer
        self.data["eval_train"] = [
            {
                "name": "MB_conf_train",
                "n_classes": 2,
                "data": MBFRConfident(
                    self.path,
                    transform=self.test_transform,
                    aug_type="torchvision",
                    train=True,
                    test_size=None,
                ),
            }
        ]


class FineTuning_DataModule(pl.LightningDataModule):
    def __init__(
        self,
        path,
        batch_size,
        num_workers=0,
        prefetch_factor=20,
        pin_memory=False,
    ):
        super().__init__()
        self.path = path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory

        self.data = {}

    def prepare_data(self):
        return

    def train_dataloader(self):
        loader = DataLoader(
            self.data["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
            shuffle=True,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.data["val"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
        return loader

    def test_dataloader(self):
        loaders = [
            DataLoader(
                data,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                prefetch_factor=self.prefetch_factor,
                pin_memory=self.pin_memory,
                shuffle=False,
            )
            for data in self.data["test"].values()
        ]
        return loaders


class RGZ_DataModule_Finetune(FineTuning_DataModule):
    def __init__(
        self,
        path,
        batch_size,
        center_crop,
        val_size=0.2,
        num_workers=0,
        prefetch_factor=20,
        pin_memory=False,
        seed=69,
    ):
        super().__init__(
            path,
            batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
        )

        self.mu = (0.008008896,)
        self.sig = (0.05303395,)

        self.center_crop = center_crop
        self.val_size = val_size
        self.seed = seed

        self.train_transform = T.Compose(
            [
                T.RandomRotation(180),
                T.CenterCrop(self.center_crop),
                T.RandomResizedCrop(self.center_crop, scale=(0.9, 1)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ToTensor(),
                T.Normalize(self.mu, self.sig),
            ]
        )

        self.test_transform = T.Compose(
            [
                T.CenterCrop(center_crop),
                T.ToTensor(),
                T.Normalize(self.mu, self.sig),
            ]
        )

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if self.val_size != 0:
            data = MBFRConfident(self.path, aug_type="torchvision", train=True)
            idx = np.arange(len(data))
            idx_train, idx_val = train_test_split(
                idx,
                test_size=self.val_size,
                stratify=data.full_targets,
                random_state=self.seed,
            )

            self.data["train"] = Subset(
                MBFRConfident(
                    self.path,
                    aug_type="torchvision",
                    train=True,
                    test_size=None,
                    transform=self.train_transform,
                ),
                idx_train,
            )

            self.data["val"] = Subset(
                MBFRConfident(
                    self.path,
                    aug_type="torchvision",
                    train=True,
                    test_size=None,
                    transform=self.test_transform,
                ),
                idx_val,
            )

        else:
            self.data["train"] = MBFRConfident(
                self.path,
                aug_type="torchvision",
                train=True,
                transform=self.train_transform,
            )
            self.data["val"] = MBFRConfident(
                self.path,
                aug_type="torchvision",
                train=True,
                transform=self.test_transform,
            )

        self.data["test"] = OrderedDict(
            {
                "MB_conf_test": MBFRConfident(
                    self.path,
                    aug_type="torchvision",
                    train=False,
                    test_size=None,
                    transform=self.test_transform,
                ),
                "MB_unc_test": MBFRUncertain(
                    self.path,
                    aug_type="torchvision",
                    train=False,
                    test_size=None,
                    transform=self.test_transform,
                ),
            },
        )


class RGZ_DataModule_Finetune_Regression(FineTuning_DataModule):
    def __init__(
        self,
        path,
        batch_size,
        center_crop,
        val_size=0.2,
        num_workers=0,
        prefetch_factor=20,
        pin_memory=False,
        seed=69,
    ):
        super().__init__(
            path,
            batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
        )

        self.mu = (0.008008896,)
        self.sig = (0.05303395,)

        self.center_crop = center_crop
        self.val_size = val_size
        self.seed = seed

        self.train_transform = T.Compose(
            [
                T.RandomRotation(180),
                T.CenterCrop(self.center_crop),
                T.RandomResizedCrop(self.center_crop, scale=(0.9, 1)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ToTensor(),
                T.Normalize(self.mu, self.sig),
            ]
        )

        self.test_transform = T.Compose(
            [
                T.CenterCrop(center_crop),
                T.ToTensor(),
                T.Normalize(self.mu, self.sig),
            ]
        )

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        rgz = RGZ108k(
            self.path,
            train=True,
            transform=self.train_transform,
            remove_duplicates=False,
            mb_cut=True,
            cut_threshold=25,
        )

        self.data["test"] = OrderedDict({})

        self.data["train"], self.data["val"], self.data["test"]["rgz"] = train_val_test_split(
            rgz,
            val_size=self.val_size,
            test_size=0.2,
            val_seed=self.seed,
            test_seed=69,
        )

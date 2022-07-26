import torch.utils.data as D
import numpy as np

from dataloading.base_dm import Base_DataModule_Eval, Base_DataModule, Base_DataModule_Supervised
from dataloading.utils import rgz_cut
from dataloading.datasets import RGZ20k
from astroaugmentations.datasets.MiraBest_F import MBFRFull, MBFRConfident, MBFRUncertain
from sklearn.model_selection import train_test_split


class RGZ_DataModule(Base_DataModule):
    def __init__(self, config):
        super().__init__(config, mu=(0,), sig=(1,))

    def prepare_data(self):
        MBFRFull(self.path, train=False, download=True)
        MBFRFull(self.path, train=True, download=True)
        RGZ20k(self.path, train=True, download=True)

    # def setup(self, stage=None):
    #     conf_test = self.config["data"]["conf_test"]

    #     self.T_train.n_views = 1

    #     D_train = self._train_set(self.T_train)

    #     self.update_transforms(D_train)

    #     # Re-initialise dataset with new mu and sig values
    #     self.data["train"] = self._train_set(self.T_train)

    #     # Initialise individual datasets with test transform (for evaluation)
    #     self.data["val"], self.data["test"] = self._val_test_set(self.T_test)

    #     self.data["labelled"] = self.data["val"]

    #     self.data["u"] = RGZ20k(self.path, train=True, transform=self.T_test)

    def setup(self, stage=None):
        self.T_train.n_views = 1

        D_train, _, _ = self.split_data()

        self.update_transforms(D_train)

        # Re-initialise dataset with new mu and sig values

        self.data["test"] = self._test_set()

        self.data["labelled"] = self.data["val"]

    def _test_set(self):
        d_conf = MBFRConfident(self.path, train=False, test_size=self.config["test_frac"])
        d_unc = MBFRUncertain(self.path, train=False, test_size=self.config["test_frac"])

        self.data["test_rgz"] = {"conf": d_conf, "unc": d_unc}

        return d_conf

    def split_data(self):
        """Split data in to train/val/test"""
        train_data, val_data, test_data = [], [], []

        D_conf, _ = self.train_test_split(
            MBFRConfident,
            self.config["data"]["test_frac"],
            aug_type="torchvision",
        )

        D_unc, D_unc_test = self.train_test_split(
            MBFRUncertain,
            self.config["data"]["test_frac"],
            aug_type="torchvision",
        )

        test_data.append(D_conf_test)
        test_data.append(D_unc_test)

        if D_conf is not None:
            D_conf_train, D_conf_val = self.train_val_split(D_conf, self.config["data"]["val_frac"])
            train_data.append(D_conf_train)
            print(D_conf_val)
            val_data = D_conf_val

        if D_unc is not None:
            D_unc_train, D_unc_val = self.train_val_split(D_unc, self.config["data"]["val_frac"])
            train_data.append(D_unc_test)
            # val_data.append(D_unc_val)

        # Concatenate datasets
        if self.config["data"]["rgz"]:
            D_rgz = RGZ20k(self.path, train=True, transform=self.T_train)
            D_rgz = rgz_cut(D_rgz, self.config["cut_threshold"], mb_cut=True)
            train_data.append(D_rgz)

        train_data = D.ConcatDataset(list(filter(None, train_data)))
        # val_data = list(filter(None, val_data))
        # test_data = list(filter(None, test_data))

        return train_data, val_data, test_data

    def train_test_split(self, dset, test_size, **kwargs):
        """Return (train_data, test_data) split from dset"""
        if test_size == 1:
            return None, dset(self.path, train=None, test_size=None, **kwargs)

        elif test_size == 0:
            return dset(self.path, train=None, test_size=None, **kwargs), None

        elif test_size == -1:
            return None, None

        else:
            return (
                dset(self.path, train=True, test_size=test_size, transform=self.T_train, **kwargs),
                dset(self.path, train=False, test_size=test_size, transform=self.T_test, **kwargs),
            )

    def train_val_split(self, data, val_size):
        """Return (train_data, test_data) split from dset"""

        if val_size == 1:
            return None, data

        elif val_size == 0:
            return data, None

        elif val_size == -1:
            return None, None

        else:
            idx, targets = np.arange(len(data.targets)), data.targets
            idx_train, idx_val = train_test_split(
                idx,
                train_size=self.config["data"]["val_size"],
                stratify=targets,
                random_state=self.config["seed"],
            )

            return D.Subset(data, idx_train), D.Subset(data, idx_val)

    def _train_set(self, transform):
        """Load MiraBest & RGZ datasets, cut MiraBest by angular size, remove duplicates from RGZ and concatenate the two"""

        data = []

        D_conf, _ = split_dataset(
            MBFRConfident,
            self.path,
            self.config["data"]["conf_test"],
            transform=transform,
            aug_type="torchvision",
        )

        if D_conf is not None:
            data.append(D_conf)

        D_unc, _ = split_dataset(
            MBFRUncertain,
            self.path,
            self.config["data"]["unc_test"],
            transform=transform,
            aug_type="torchvision",
        )

        if D_unc is not None:
            data.append(D_unc)

        # Concatenate datasets
        if self.config["data"]["rgz"]:
            D_rgz = RGZ20k(self.path, train=True, transform=transform)
            D_rgz = rgz_cut(D_rgz, self.config["cut_threshold"], mb_cut=True)
            data.append(D_rgz)

        return D.ConcatDataset(data)

    def _val_test_set(self, transform):
        data = []

        _, D_conf = split_dataset(
            MBFRConfident,
            self.path,
            self.config["data"]["conf_test"],
            transform=transform,
            aug_type="torchvision",
        )

        # if D_conf is not None:
        #     data.append(D_conf)

        data = D_conf

        idx, targets = np.arange(len(data.targets)), data.targets
        idx_val, idx_test = train_test_split(
            idx,
            train_size=self.config["data"]["val_size"],
            stratify=targets,
            random_state=self.config["seed"],
        )

        return D.Subset(data, idx_val), D.Subset(data, idx_test)

        # D_val, D_test = D.Subset(data, idx_val), D.Subset(data, idx_test)
        # return D.ConcatDataset([D_val, D_val, D_val]), D.ConcatDataset([D_test, D_test, D_test])


class RGZ_DataModule_Eval(Base_DataModule_Eval):
    def __init__(self, encoder, config):
        super().__init__(encoder, config)

    def setup(self, stage=None):

        # Calculate mean and std of data
        D_train = MBFRFull(self.path, train=True, transform=self.T_train, aug_type="torchvision")

        self.update_transforms(D_train)

        # Initialise individual datasets
        self.data["train"] = self._val_set(self.T_test)
        self.data["val"] = self._val_set(self.T_test)
        self.data["test"] = MBFRFull(
            self.path,
            train=False,
            transform=self.T_test,
            aug_type="torchvision",
        )

    def _val_set(self, transform):

        _, D_conf = split_dataset(
            MBFRConfident,
            self.path,
            self.config["data"]["conf_test"],
            transform=transform,
            aug_type="torchvision",
        )

        _, D_unc = split_dataset(
            MBFRUncertain,
            self.path,
            self.config["data"]["unc_test"],
            transform=transform,
            aug_type="torchvision",
        )

        data = []
        if D_conf is not None:
            data.append(D_conf)

        if D_unc is not None:
            data.append(D_unc)

        return D.ConcatDataset(data)


class RGZ_DataModule_Supervised(Base_DataModule_Supervised):
    def __init__(self, config):
        super().__init__(config, mu=(0,), sig=(1,))

    def setup(self, stage=None):

        self.T_train.n_views = 1
        D_train = self._train_set(self.T_train)

        self.update_transforms(D_train)

        # Re-initialise dataset with new mu and sig values self.data["train"] = self._train_set(self.T_train)

        # Initialise individual datasets with test transform (for evaluation)
        self.data["train"] = self._train_set(self.T_train)
        self.data["val"], self.data["test"] = self._val_test_set(self.T_test)
        self.data["labelled"] = D.ConcatDataset([self.data["train"], self.data["val"]])

    def _train_set(self, transform):
        """Load MiraBest & RGZ datasets, cut MiraBest by angular size, remove duplicates from RGZ and concatenate the two"""

        data = []

        D_conf, _ = split_dataset(
            MBFRConfident,
            self.path,
            self.config["data"]["conf_test"],
            transform=transform,
            aug_type="torchvision",
        )

        if D_conf is not None:
            data.append(D_conf)

        D_unc, _ = split_dataset(
            MBFRUncertain,
            self.path,
            self.config["data"]["unc_test"],
            transform=transform,
            aug_type="torchvision",
        )

        if D_unc is not None:
            data.append(D_unc)

        return D.ConcatDataset(data)

    def _val_test_set(self, transform):
        data = []

        _, D_conf = split_dataset(
            MBFRConfident,
            self.path,
            self.config["data"]["conf_test"],
            transform=transform,
            aug_type="torchvision",
        )

        if D_conf is not None:
            data.append(D_conf)

        #         _, D_unc = split_dataset(
        #             MBFRUncertain,
        #             self.path,
        #             self.config["data"]["unc_test"],
        #             transform=transform,
        #             aug_type="torchvision",
        #         )

        #         if D_unc is not None:
        #             data.append(D_unc)

        #         data = D.ConcatDataset(data)

        data = D_conf

        idx, targets = np.arange(len(data.targets)), data.targets
        idx_val, idx_test = train_test_split(
            idx,
            train_size=self.config["data"]["val_size"],
            stratify=targets,
            random_state=self.config["seed"],
        )

        return D.Subset(data, idx_val), D.Subset(data, idx_test)


def train_val_idx(dset, seed=None, val_frac=0.2):
    if seed is None:
        seed = np.random.randint(9999999)

    n = len(dset)
    idx = np.arange(n)
    labels = np.array(dset.targets)

    # Split into train/val #
    train_idx, val_idx = train_test_split(
        idx,
        test_size=val_frac,
        stratify=labels,
        random_state=seed,
    )

    return train_idx, val_idx  # pass data dict to dataloaders

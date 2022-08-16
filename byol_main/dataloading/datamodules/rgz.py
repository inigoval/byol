import torch.utils.data as D
import numpy as np

from dataloading.base_dm import Base_DataModule_Eval, Base_DataModule, Base_DataModule_Supervised
from dataloading.datasets import RGZ20k
from dataloading.utils import rgz_cut, size_cut, mb_cut
from astroaugmentations.datasets.MiraBest_F import MBFRFull, MBFRConfident, MBFRUncertain
from sklearn.model_selection import train_test_split


class RGZ_DataModule(Base_DataModule):
    def __init__(self, config):
        super().__init__(config, mu=(0,), sig=(1,))

    def prepare_data(self):
        MBFRFull(self.path, train=False, download=True)
        MBFRFull(self.path, train=True, download=True)
        RGZ20k(self.path, train=True, download=True)

    def setup(self, stage=None):

        self.T_train.n_views = 1
        d_train, _ = self._train_val_set()

        self.update_transforms(d_train)

        # Re-initialise dataset with new mu and sig values

        self.data["train"], self.data["val"] = self._train_val_set()

        self.data["test"] = self._test_set()

        self.data["labelled"] = self.data["val"]

    def _test_set(self):
        d_conf = MBFRConfident(
            self.path,
            train=False,
            test_size=self.config["data"]["test_frac"],
            transform=self.T_test,
            aug_type="torchvision",
        )

        d_unc = MBFRUncertain(
            self.path,
            train=False,
            test_size=self.config["data"]["test_frac"],
            transform=self.T_test,
            aug_type="torchvision",
        )

        # print(f"test: {len(d_conf)}")

        self.data["test_rgz"] = {"conf": d_conf, "unc": d_unc}

        return d_conf

    def _train_val_set(self):
        idx_train = {}
        idx_val = {}

        d_conf = MBFRConfident(
            self.path,
            train=True,
            test_size=self.config["data"]["test_frac"],
            transform=self.T_train,
            aug_type="torchvision",
        )

        idx_train["conf"], idx_val["conf"] = train_val_idx(
            d_conf,
            val_size=self.config["data"]["val_frac"],
            seed=self.config["seed"],
        )

        d_train = []

        if self.config["data"]["rgz"]:
            d_rgz = RGZ20k(self.path, train=True, transform=self.T_train)
            d_rgz = rgz_cut(d_rgz, self.config["cut_threshold"], mb_cut=True, remove_duplicates=True)
            # size_cut(d_rgz, self.config["cut_threshold"])
            # mb_cut(d_rgz)
            d_train.append(d_rgz)

        d_unc = MBFRUncertain(
            self.path,
            train=True,
            test_size=self.config["data"]["test_frac"],
            transform=self.T_train,
            aug_type="torchvision",
        )
        idx_train["unc"], idx_val["unc"] = train_val_idx(
            d_unc,
            val_size=self.config["data"]["val_frac"],
            seed=self.config["seed"],
        )

        d_val = MBFRConfident(
            self.path,
            train=True,
            test_size=self.config["data"]["test_frac"],
            transform=self.T_test,
            aug_type="torchvision",
        )
        d_val = D.Subset(d_val, idx_val["conf"])

        if idx_train["unc"] is not None:
            d_train.append(D.Subset(d_unc, idx_train["unc"]))

        if idx_train["conf"] is not None:
            d_train.append(D.Subset(d_conf, idx_train["conf"]))

        d_train = D.ConcatDataset(d_train)

        return d_train, d_val


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


def train_val_idx(dset, seed=None, val_size=0.2):
    if seed is None:
        seed = np.random.randint(9999999)

    n = len(dset)
    idx = np.arange(n)
    labels = np.array(dset.targets)

    if val_size == 1:
        return None, idx

    if val_size == 0:
        return idx, None

    else:
        # Split into train/val #
        train_idx, val_idx = train_test_split(
            idx,
            test_size=val_size,
            stratify=labels,
            random_state=seed,
        )

    return train_idx, val_idx  # pass data dict to dataloaders

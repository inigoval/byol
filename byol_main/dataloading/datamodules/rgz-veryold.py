import torch.utils.data as D

from dataloading.base_dm import Base_DataModule_Eval, Base_DataModule
from dataloading.utils import rgz_cut
from dataloading.datasets import RGZ20k
from astroaugmentations.datasets.MiraBest_F import MBFRFull, MBFRConfident, MBFRUncertain


class RGZ_DataModule(Base_DataModule):
    def __init__(self, config):
        super().__init__(config, mu=(0,), sig=(1,))

    def prepare_data(self):
        MBFRFull(self.path, train=False, download=True)
        MBFRFull(self.path, train=True, download=True)
        RGZ20k(self.path, train=True, download=True)

    def setup(self, stage=None):

        self.T_train.n_views = 1
        D_train = self._train_set(self.T_train)

        self.update_transforms(D_train)

        # Re-initialise dataset with new mu and sig values
        self.data["train"] = self._train_set(self.T_train)
        # Initialise individual datasets with test transform (for evaluation)
        # self.data["val"] = MBFRFull(self.path, train=False, transform=self.T_test)
        self.data["val"] = self._val_set(self.T_test)

        self.data["labelled"] = MBFRFull(
            self.path,
            train=True,
            transform=self.T_test,
            aug_type="torchvision",
        )
        self.data["u"] = RGZ20k(self.path, train=True, transform=self.T_test)

    def _train_set(self, transform):
        """Load MiraBest & RGZ datasets, cut MiraBest by angular size, remove duplicates from RGZ and concatenate the two"""
        D_rgz = RGZ20k(self.path, train=True, transform=transform)
        D_rgz = rgz_cut(D_rgz, self.config["cut_threshold"], mb_cut=True)
        D_conf, _ = split_dataset(
            MBFRConfident,
            self.path,
            self.config["data"]["conf_test"],
            transform=transform,
            aug_type="torchvision",
        )

        D_unc, _ = split_dataset(
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

        if self.config["data"]["rgz"]:
            data.append(D_rgz)

        return D.ConcatDataset(data)

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


def split_dataset(dset, path, test_size, **kwargs):
    """Split dataset to a given split"""
    if test_size == 1:
        return None, dset(path, train=False, test_size=test_size, **kwargs)

    elif test_size == 0:
        return dset(path, train=True, test_size=test_size, **kwargs), None

    else:
        return (
            dset(path, train=True, test_size=test_size, **kwargs),
            dset(path, train=False, test_size=test_size, **kwargs),
        )

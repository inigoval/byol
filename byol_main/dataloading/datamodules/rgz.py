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
        conf_test = self.config["data"]["conf_test"]

        self.T_train.n_views = 1
        D_train = self._train_set(self.T_train)

        self.update_transforms(D_train)

        # Re-initialise dataset with new mu and sig values
        self.data["train"] = self._train_set(self.T_train)
        # Initialise individual datasets with test transform (for evaluation)
        # self.data["val"] = MBFRFull(self.path, train=False, transform=self.T_test)
        self.data["val"] = self._val_set(self.T_test)
        self.data["test"] = MBFRFull(
            self.path,
            train=False,
            transform=self.T_test,
            aug_type="torchvision",
        )
        self.data["labelled"] = MBFRFull(
            self.path,
            train=True,
            transform=self.T_test,
            aug_type="torchvision",
        )
        self.data["u"] = RGZ20k(self.path, train=True, transform=self.T_test)

    def _train_set(self, transform):
        """Load MiraBest & RGZ datasets, cut MiraBest by angular size, remove duplicates from RGZ and concatenate the two"""
        D_conf = MBFRConfident(
            self.path,
            train=True,
            transform=transform,
            test_size=self.config["data"]["conf_test"],
            aug_type="torchvision",
        )
        D_unc = MBFRUncertain(
            self.path,
            train=True,
            transform=transform,
            test_size=self.config["data"]["unc_test"],
            aug_type="torchvision",
        )

        D_rgz = RGZ20k(self.path, train=True, transform=transform)
        D_rgz = rgz_cut(D_rgz, self.config["cut_threshold"], mb_cut=True)

        # Concatenate datasets
        data = [D_conf, D_unc]
        if self.config["data"]["rgz"]:
            data.append(D_rgz)

        return D.ConcatDataset(data)

    def _val_set(self, transform):
        D_conf = MBFRConfident(
            self.path,
            train=False,
            transform=transform,
            test_size=self.config["data"]["conf_test"],
            aug_type="torchvision",
        )

        D_unc = MBFRUncertain(
            self.path,
            train=False,
            transform=transform,
            test_size=self.config["data"]["unc_test"],
            aug_type="torchvision",
        )

        return D_conf


class RGZ_DataModule_Eval(Base_DataModule_Eval):
    def __init__(self, encoder, config):
        super().__init__(encoder, config)

    def setup(self, stage=None):

        # Calculate mean and std of data
        D_train = MBFRFull(self.path, train=True, transform=self.T_train, aug_type="torchvision")

        self.update_transforms(D_train)

        # Initialise individual datasets
        self.data["train"] = MBFRFull(
            self.path,
            train=True,
            transform=self.T_train,
            aug_type="torchvision",
        )
        self.data["val"] = self._val_set(self.T_test)
        self.data["test"] = MBFRFull(
            self.path,
            train=False,
            transform=self.T_test,
            aug_type="torchvision",
        )
        self.data["mb"] = MBFRFull(
            self.path,
            train=True,
            transform=self.T_test,
            aug_type="torchvision",
        )

    def _train_set(self, transform):
        """Load MiraBest & RGZ datasets, cut MiraBest by angular size, remove duplicates from RGZ and concatenate the two"""
        D_conf = MBFRConfident(
            self.path,
            train=True,
            transform=transform,
            test_size=1 - self.config["data"]["conf_train"],
            aug_type="torchvision",
        )

        # Concatenate datasets
        # data = [D_conf, D_unc]

        return D_conf

    def _val_set(self, transform):
        D_conf = MBFRConfident(
            self.path,
            train=False,
            transform=transform,
            test_size=self.config["data"]["conf_test"],
            aug_type="torchvision",
        )

        D_unc = MBFRUncertain(
            self.path,
            train=False,
            transform=transform,
            test_size=self.config["data"]["unc_test"],
            aug_type="torchvision",
        )

        return D_conf

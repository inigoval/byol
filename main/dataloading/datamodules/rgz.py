import torch.utils.data as D

from dataloading.base_dm import Base_DataModule_Eval, Base_DataModule
from torchvision.datasets import ImageFolder
from dataloading.utils import rgz_cut
from dataloading.datasets import MB_nohybrids, RGZ20k


class RGZ_DataModule(Base_DataModule):
    def __init__(self, config):
        super().__init__(config, mu=(0,), sig=(1,))

    def prepare_data(self):
        MB_nohybrids(self.path, train=False, download=True)
        MB_nohybrids(self.path, train=True, download=True)
        RGZ20k(self.path, train=True, download=True)

    def setup(self):
        self.T_train.n_views = 1
        D_train = self._train_set(self.T_train)

        self.update_transforms(D_train)

        # Re-initialise dataset with new mu and sig values
        self.data["train"] = self._train_set(self.T_train)
        # Initialise individual datasets with test transform (for evaluation)
        self.data["val"] = MB_nohybrids(self.path, train=False, transform=self.T_test)
        self.data["test"] = MB_nohybrids(self.path, train=False, transform=self.T_test)
        self.data["l"] = MB_nohybrids(self.path, train=True, transform=self.T_test)
        self.data["u"] = RGZ20k(self.path, train=True, transform=self.T_test)

    def _train_set(self, transform):
        """Load MiraBest & RGZ datasets, cut MiraBest by angular size, remove duplicates from RGZ and concatenate the two"""
        D_rgz = RGZ20k(self.path, train=True, transform=transform)
        D_rgz = rgz_cut(D_rgz, self.config["cut_threshold"], mb_cut=True)
        D_mb = MB_nohybrids(self.path, train=True, transform=transform)

        # Concatenate datasets
        return D.ConcatDataset([D_rgz, D_mb])


class RGZ_DataModule_Eval(Base_DataModule_Eval):
    def __init__(self, encoder, config):
        super().__init__(encoder, config)

    def setup(self, stage=None):
        # D_train = self.cut_and_cat()

        # Calculate mean and std of data
        D_train = MB_nohybrids(self.path, train=True, transform=self.T_train)

        self.update_transforms(D_train)

        # Initialise individual datasets
        self.data["train"] = MB_nohybrids(self.path, train=True, transform=self.T_train)
        self.data["val"] = MB_nohybrids(self.path, train=False, transform=self.T_test)
        self.data["test"] = MB_nohybrids(self.path, train=False, transform=self.T_test)
        self.data["mb"] = MB_nohybrids(self.path, train=True, transform=self.T_test)
        self.data["rgz"] = RGZ20k(self.path, train=True, transform=self.T_test)

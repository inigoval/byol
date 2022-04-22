# uses dataset from pytorch-galaxy-datasets/galaxy_zoo_2.py
# does not use the datamodule, as this code has different requirements (e.g. normalisation, augmentation choice, etc)

import numpy as np
# https://github.com/mwalmsley/pytorch-galaxy-datasets
from pytorch_galaxy_datasets.galaxy_zoo_2 import GZ2Dataset
from sklearn.model_selection import train_test_split

from dataloading.base_dm import Base_DataModule_Eval, Base_DataModule


def split_catalog(catalog, train_fraction=.7, val_fraction=.1, test_fraction=.2):
    assert np.isclose(train_fraction + val_fraction + test_fraction, 1.)
    
    train_catalog, hidden_catalog = train_test_split(catalog, train_size=train_fraction)
    val_catalog, test_catalog = train_test_split(
        hidden_catalog, train_size=val_fraction/(val_fraction+test_fraction))
    return train_catalog, val_catalog, test_catalog


# config arg not used
class GZ2_DataModule(Base_DataModule):  # not the same as in pytorch-galaxy-datasets
    def __init__(self, config):
        super().__init__(config, mu=(0,), sig=(1,))

    def prepare_data(self):
        GZ2Dataset(self.path, label_cols=['label'], download=True)

    def setup(self):
        self.T_train.n_views = 1

        full_catalog = GZ2Dataset(self.path).catalog
        train_catalog, val_catalog, test_catalog = split_catalog(full_catalog)

        D_train = GZ2Dataset(self.path, catalog=train_catalog, transform=self.T_train)
        self.update_transforms(D_train)

        # Re-initialise dataset with new mu and sig values
        self.data["train"] = GZ2Dataset(
            self.path, catalog=train_catalog, download=True, transform=self.T_train
        )
        # Initialise individual datasets with test transform (for evaluation)
        self.data["val"] = GZ2Dataset(self.path, catalog=val_catalog, transform=self.T_test)
        self.data["test"] = GZ2Dataset(self.path, catalog=test_catalog, transform=self.T_test)
        self.data["l"] = GZ2Dataset(self.path, catalog=train_catalog,  transform=self.T_test)


# config arg not used
class GZ2_DataModule_Eval(Base_DataModule_Eval):
    def __init__(self, encoder, config):
        super().__init__(encoder, config)

    def setup(self):

        full_catalog = GZ2Dataset(self.path).catalog
        train_catalog, val_catalog, test_catalog = split_catalog(full_catalog)

        # Initialise individual datasets with identity transform (for evaluation)
        D_train = GZ2Dataset(self.path, catalog=train_catalog, download=True, transform=self.T_train)

        self.update_transforms(D_train)

        self.data["train"] = GZ2Dataset(
            self.path, catalog=train_catalog, download=True, transform=self.T_train
        )
        self.data["val"] = GZ2Dataset(self.path, catalog=val_catalog, transform=self.T_test)
        self.data["test"] = GZ2Dataset(self.path, catalog=test_catalog, transform=self.T_test)
        self.data["l"] = GZ2Dataset(self.path,catalog=train_catalog,  transform=self.T_test)


if __name__ == '__main__':

    GZ2_DataModule(config=None)[0]

    GZ2_DataModule_Eval(config=None)

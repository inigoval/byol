# uses dataset from pytorch-galaxy-datasets/galaxy_zoo_2.py
# does not use the datamodule, as this code has different requirements (e.g. normalisation, augmentation choice, etc)

import numpy as np
# https://github.com/mwalmsley/pytorch-galaxy-datasets
from pytorch_galaxy_datasets.galaxy_zoo_2 import GZ2Dataset
from sklearn.model_selection import train_test_split

from byol_main.dataloading.base_dm import Base_DataModule_Eval, Base_DataModule


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
        GZ2Dataset(self.path, download=True)

    def setup(self, stage=None):  # stage by ptl convention
        self.T_train.n_views = 1

        full_catalog = GZ2Dataset(self.path, download=True).catalog
        full_catalog = full_catalog.query('label >= 0').sample(10000)  # -1 indicates cannot be assigned a label
        train_catalog, val_catalog, test_catalog = split_catalog(full_catalog)

        D_train = GZ2Dataset(self.path, label_cols=['label'], catalog=train_catalog, transform=self.T_train)
        self.update_transforms(D_train)

        # Re-initialise dataset with new mu and sig values
        self.data["train"] = GZ2Dataset(
            self.path, catalog=train_catalog, label_cols=['label'], download=True, transform=self.T_train
        )
        # Initialise individual datasets with test transform (for evaluation)
        self.data["val"] = GZ2Dataset(self.path, label_cols=['label'], catalog=val_catalog, transform=self.T_test)
        self.data["test"] = GZ2Dataset(self.path, label_cols=['label'], catalog=test_catalog, transform=self.T_test)
        self.data["l"] = GZ2Dataset(self.path, label_cols=['label'], catalog=val_catalog,  transform=self.T_test)  # will be unpacked into feature_bank, target_bank, for knn eval
        # TODO temporarily switched to val catalog as it's smaller and so easier to run KNN on


# config arg not used
# TODO nasty duplication with the above - unsure why there's two of these
class GZ2_DataModule_Eval(Base_DataModule_Eval):
    def __init__(self, encoder, config):
        super().__init__(encoder, config)

    def setup(self):

        full_catalog = GZ2Dataset(self.path).catalog
        full_catalog = full_catalog.query('label >= 0').sample(10000)  # -1 indicates cannot be assigned a label
        train_catalog, val_catalog, test_catalog = split_catalog(full_catalog)
        train_catalog, val_catalog, test_catalog = split_catalog(full_catalog)

        # Initialise individual datasets with identity transform (for evaluation)
        D_train = GZ2Dataset(self.path, catalog=train_catalog, download=True, transform=self.T_train)

        self.update_transforms(D_train)

        self.data["train"] = GZ2Dataset(
            self.path, catalog=train_catalog, download=True, transform=self.T_train
        )
        self.data["val"] = GZ2Dataset(self.path, catalog=val_catalog, transform=self.T_test)
        self.data["test"] = GZ2Dataset(self.path, catalog=test_catalog, transform=self.T_test)
        self.data["l"] = GZ2Dataset(self.path,catalog=val_catalog,  transform=self.T_test)


if __name__ == '__main__':

    import yaml

    with open('config/byol/gz2.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config['dataset'] = 'gz2'
    config['debug'] = True
    config['num_workers'] = 1
    # config = {
    #     'batch_size': 32,
    #     'num_workers': 1,
    #     'debug': True,
    #     'dataset': 'gz2'
    # }
    print(config)

    datamodule = GZ2_DataModule(config=config)
    # GZ2_DataModule_Eval(config=config)
    datamodule.setup()

    for (images, labels) in datamodule.train_dataloader():
        print(images[0].shape, labels.shape)  # [0] as list of views
        break

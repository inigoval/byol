# uses dataset from pytorch-galaxy-datasets/galaxy_zoo_2.py
# does not use the datamodule, as this code has different requirements (e.g. normalisation, augmentation choice, etc)
import logging

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
        full_catalog = full_catalog.query('label >= 0') # -1 indicates cannot be assigned a label
        if self.config['debug']:
            full_catalog = full_catalog.sample(20000)  
        train_catalog, val_catalog, test_catalog = split_catalog(full_catalog)

        logging.info('Loading 5k subset of full dataset to adjust mu, sigma')  # subset as kinda slow as single-threaded here, possibly
        D_train = GZ2Dataset(self.path, label_cols=['label'], catalog=train_catalog.sample(5000), transform=self.T_train)
        self.update_transforms(D_train)

        # Re-initialise dataset with new mu and sig values
        self.data["train"] = GZ2Dataset(
            self.path, catalog=train_catalog, label_cols=['label'], download=True, transform=self.T_train
        )
        # Initialise individual datasets with test transform (for evaluation)
        self.data["val"] = GZ2Dataset(self.path, label_cols=['label'], catalog=val_catalog, transform=self.T_test)
        self.data["test"] = GZ2Dataset(self.path, label_cols=['label'], catalog=test_catalog, transform=self.T_test)
        self.data["labelled"] = GZ2Dataset(self.path, label_cols=['label'], catalog=train_catalog.sample(10000),  transform=self.T_test)  # will be unpacked into feature_bank, target_bank, for knn eval
        # TODO temporarily switched to val catalog as it's smaller and so easier to run KNN on


# config arg not used
# TODO nasty duplication with the above - unsure why there's two of these
class GZ2_DataModule_Eval(Base_DataModule_Eval):
    def __init__(self, encoder, config):
        super().__init__(encoder, config)

    def setup(self, stage=None):

        full_catalog = GZ2Dataset(self.path).catalog
        full_catalog = full_catalog.query('label >= 0')
        if self.config['debug']:
            full_catalog = full_catalog.sample(20000)  
        train_catalog, val_catalog, test_catalog = split_catalog(full_catalog)


        logging.info('Loading 5k subset of full dataset to adjust mu, sigma')  # subset as kinda slow as single-threaded here, possibly
        D_train = GZ2Dataset(self.path, label_cols=['label'], catalog=train_catalog.sample(5000), transform=self.T_train)
        self.update_transforms(D_train)


        # Initialise individual datasets with identity transform (for evaluation)
        self.data["train"] = GZ2Dataset(
            self.path, catalog=train_catalog, label_cols=['label'], download=True, transform=self.T_train
        )
        self.data["val"] = GZ2Dataset(self.path, catalog=val_catalog, label_cols=['label'], transform=self.T_test)
        self.data["test"] = GZ2Dataset(self.path, catalog=test_catalog, label_cols=['label'], transform=self.T_test)
        self.data["labelled"] = GZ2Dataset(self.path, catalog=train_catalog.sample(10000), label_cols=['label'], transform=self.T_test)


if __name__ == '__main__':

    import yaml
    import torch

    with open('config/byol/gz2.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config['dataset'] = 'gz2'
    config['debug'] = True
    config['num_workers'] = 1
    config['data'] = {'mu': 0, 'sig': 1, 'rotate': True, 'input_height': 64} # needed for _Eval
    # print(config)

    for datamodule in [GZ2_DataModule(config=config), GZ2_DataModule_Eval(config=config, encoder=lambda x: torch.from_numpy(np.random.rand(len(x), 512)))]:

        datamodule.setup()

        for (images, labels) in datamodule.train_dataloader():
            print(images[0].shape, labels.shape)  # [0] as list of views
            assert labels.min() >= 0

        for (images, labels) in datamodule.val_dataloader():
            print(images[0].shape, labels.shape)  # [0] as list of views
            assert labels.min() >= 0
    

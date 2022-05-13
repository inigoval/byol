# uses dataset from pytorch-galaxy-datasets/galaxy_zoo_2.py
# does not use the datamodule, as this code has different requirements (e.g. normalisation, augmentation choice, etc)
import logging

import numpy as np
# https://github.com/mwalmsley/pytorch-galaxy-datasets
from pytorch_galaxy_datasets import galaxy_dataset
from pytorch_galaxy_datasets.prepared_datasets import decals_dr5_setup

from foundation.datasets import decals_dr5

from byol_main.dataloading.base_dm import Base_DataModule_Eval, Base_DataModule


class Galaxy_DataModule(Base_DataModule):

    def __init__(self, config, mu=(0,), sig=(1,)):
        super().__init__(config, mu, sig)

    # TODO could refactor up further
    def adjust_mu_and_std_from_subset(self, train_catalog, label_cols, size):
        # subset as kinda slow as single-threaded here, possibly
        logging.info('Loading 5k subset of train dataset to adjust mu, sigma') 
        D_train = galaxy_dataset.GalaxyDataset(train_catalog.sample(size), label_cols=label_cols, transform=self.T_train)
        self.update_transforms(D_train)


    def create_transformed_datasets_from_catalogs(self, label_cols, train_catalog, val_catalog, test_catalog):
        # Re-initialise dataset with new mu and sig values
        self.data["train"] = galaxy_dataset.GalaxyDataset(catalog=train_catalog, label_cols=label_cols, transform=self.T_train)
        # Initialise individual datasets with test transform (for evaluation)
        # these must always be classification problems with a int label, for knn to make sense
        # val used for knn input, searched within feature/target bank
        self.data["val"] = galaxy_dataset.GalaxyDataset(label_cols=['label'], catalog=val_catalog, transform=self.T_test)
        # test not currently used
        self.data["test"] = galaxy_dataset.GalaxyDataset(label_cols=['label'], catalog=test_catalog, transform=self.T_test)
        # labelled unpacked into feature_bank, target_bank, for knn eval
        self.data["labelled"] = galaxy_dataset.GalaxyDataset(label_cols=['label'], catalog=train_catalog.sample(10000),  transform=self.T_test) 

    def prepare_data(self):
        raise NotImplementedError
 

    def setup(self, stage=None):
        raise NotImplementedError


class Decals_DataModule(Galaxy_DataModule):
    def __init__(self, config):
        super().__init__(config, mu=(0,), sig=(1,))  # TODO specify explicitly

    
    def prepare_data(self):
        # will actually just download both anyway, but for completeness
        decals_dr5_setup(self.path, train=True, download=True)
        decals_dr5_setup(self.path, train=False, download=True)


    def setup(self, stage=None):

        if self.config['labels'] == 'classification':
            catalog_creation_func = decals_dr5.decals_smooth_vs_featured
        elif self.config['labels'] == 'dirichlet':
            catalog_creation_func = decals_dr5.decals_dirichlet
        else:
            raise ValueError(self.config['labels'])
    
        label_cols, (train_catalog, val_catalog, test_catalog) = catalog_creation_func(self.path, self.config['debug'])

        self.adjust_mu_and_std_from_subset(train_catalog, label_cols, size=5000)

        self.create_transformed_datasets_from_catalogs(label_cols, train_catalog, val_catalog, test_catalog)



# config arg not used
class Decals_DataModule_Eval(Base_DataModule_Eval):
    def __init__(self, encoder, config):
        super().__init__(encoder, config)

    def setup(self, stage=None):

        # assumes superclass set self.config = config
        temporary_datamodule = Decals_DataModule(self.config)
        # eval must always be a classification problem
        self.config = self.config.copy()  # avoid mutate-by-ref for other objects
        self.config['labels'] = 'classification'
        temporary_datamodule.setup()

        # now re-use exactly the same data from temporary datamodule (now we will actually use the labels)
        self.data['train'] = temporary_datamodule.data['train']
        self.data['val'] = temporary_datamodule.data['val']
        self.data['test'] = temporary_datamodule.data['test']
        self.data['labelled'] = temporary_datamodule.data['labelled']

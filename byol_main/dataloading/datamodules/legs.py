# uses dataset from pytorch-galaxy-datasets/galaxy_zoo_2.py
# does not use the datamodule, as this code has different requirements (e.g. normalisation, augmentation choice, etc)
import logging

import numpy as np
# https://github.com/mwalmsley/pytorch-galaxy-datasets
from pytorch_galaxy_datasets import galaxy_dataset
from pytorch_galaxy_datasets.prepared_datasets import legs_setup
from sklearn.model_selection import train_test_split

from byol_main.dataloading.base_dm import Base_DataModule_Eval, Base_DataModule

# pretty much a duplicate of gz2.py

def add_smooth_featured_labels(df):
    df = df[df['smooth-or-featured-dr5_total-votes'] > 20]  # should be somewhat reliable, and classified in dr5 (say)
    df = df[df['smooth-or-featured-dr5_artifact_fraction'] < 0.3]  # remove major artifacts
    df['label'] = (df['smooth-or-featured-dr5_smooth'] > df['smooth-or-featured-dr5_featured-or-disk']).astype(int)
    return df

# config arg used by super() only for now, but could use to modify behaviour
class Legs_DataModule(Base_DataModule):  # not the same as in pytorch-galaxy-datasets
    def __init__(self, config):
        super().__init__(config, mu=(0,), sig=(1,))

    def prepare_data(self):
        # will actually just download both anyway, but for completeness
        legs_setup(train=True, download=True)
        legs_setup(train=False, download=True)

    def setup(self, stage=None):  # stage by ptl convention
        self.T_train.n_views = 1

        train_and_val_catalog, _ = legs_setup(split='train', download=True)
        test_catalog, _ = legs_setup(split='test', download=True)
        unlabelled_catalog = legs_setup(split='unlabelled', download=True)

        # only has regression labels. Let's make a smooth/featured class (and drop artifacts) to have simple knn target (under 'label')
        train_and_val_catalog = add_smooth_featured_labels(train_and_val_catalog)
        test_catalog = add_smooth_featured_labels(test_catalog)
        unlabelled_catalog['label'] = -1  # should not be used, but will still be accessed

        if self.config['debug']:
            train_catalog = train_and_val_catalog.sample(20000)  
            test_catalog = test_catalog.sample(2000)  
        
        train_catalog, val_catalog = train_test_split(train_and_val_catalog, train_size=0.8)

        # subset as kinda slow as single-threaded here, possibly
        logging.info('Loading 5k subset of train dataset to adjust mu, sigma') 
        D_train = galaxy_dataset.GalaxyDataset(train_catalog.sample(5000), label_cols=['label'], transform=self.T_train)
        self.update_transforms(D_train)

        # Re-initialise dataset with new mu and sig values
        self.data["train"] = galaxy_dataset.GalaxyDataset(catalog=unlabelled_catalog, label_cols=['label'], transform=self.T_train)  # train on all unlabelled data (don't need train catalog itself here)
        # Initialise individual datasets with test transform (for evaluation)
        self.data["val"] = galaxy_dataset.GalaxyDataset(label_cols=['label'], catalog=val_catalog, transform=self.T_test)  # use any labelled data NOT in 'labelled' as feature bank
        self.data["test"] = galaxy_dataset.GalaxyDataset(label_cols=['label'], catalog=test_catalog, transform=self.T_test)  # not used
        self.data["labelled"] = galaxy_dataset.GalaxyDataset(label_cols=['label'], catalog=train_catalog.sample(10000),  transform=self.T_test)  # will be unpacked into feature_bank, target_bank, for knn eval


# used for the linear eval at the end
# really, I would like different datamodule for self-sup training, knn validation, and linear eval
# easiest way is likely replacing self.data['val'] above, and ignoring linear eval at the end
class Legs_DataModule_Eval(Base_DataModule_Eval):
    def __init__(self, encoder, config):
        super().__init__(encoder, config)

    def setup(self, stage=None):

        # assumes superclass set self.config = config
        temporary_datamodule = Legs_DataModule(self.config)
        temporary_datamodule.setup()

        # now re-use exactly the same data from temporary datamodule (now we will actually use the labels)
        self.data['train'] = temporary_datamodule.data['train']
        self.data['val'] = temporary_datamodule.data['val']
        self.data['test'] = temporary_datamodule.data['test']
        self.data['labelled'] = temporary_datamodule.data['labelled']

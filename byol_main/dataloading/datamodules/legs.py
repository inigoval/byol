# uses dataset from pytorch-galaxy-datasets/galaxy_zoo_2.py
# does not use the datamodule, as this code has different requirements (e.g. normalisation, augmentation choice, etc)
import logging

import pandas as pd
# https://github.com/mwalmsley/pytorch-galaxy-datasets

from pytorch_galaxy_datasets import galaxy_dataset
from foundation.datasets import legs

from byol_main.dataloading.datamodules import generic_galaxy


class Legs_DataModule(generic_galaxy.Galaxy_DataModule):
    def __init__(self, config):
        super().__init__(
            config,
            mu=((0.1207394003868103, 0.11860499531030655, 0.11299177259206772)),
            sig=(0.21977204084396362, 0.21977204084396362, 0.21977204084396362)
        )

    def prepare_data(self):
        pass

    def setup(self, stage=None):

        if self.config['catalog_to_use'] == 'smooth_vs_featured':
            catalog_creation_func = legs.legs_smooth_vs_featured
        elif self.config['catalog_to_use'] == 'labelled_dirichlet_with_rings':
            # currently only gives labelled data, will extend
            catalog_creation_func = legs.legs_labelled_dirichlet_with_rings
        elif self.config['catalog_to_use'] == 'all_dirichlet_with_rings':
            catalog_creation_func = legs.legs_all_dirichlet_with_rings
        else:
            raise ValueError(self.config['catalog_to_use'])
    
        label_cols, (train_catalog, val_catalog, test_catalog, unlabelled_catalog) = catalog_creation_func(
            self.path, self.config['debug'], download=True)

        self.adjust_mu_and_std_from_subset(train_catalog, label_cols, size=5000)

        # not using self.create_transformed_datasets_from_catalogs here, custom as unlabelled
        # might generalise later

        # train on all train+unlabelled data (labels not used)
        self.data["train"] = galaxy_dataset.GalaxyDataset(catalog=pd.concat(
            [unlabelled_catalog, train_catalog], axis=0), label_cols=label_cols, transform=self.T_train)

        # Initialise individual datasets with test transform (for evaluation)
        # use any labelled data NOT in 'labelled' as feature bank
        # 'val' is for knn, must be classification e.g. rings
        # missing labels are encoded as -1
        self.data["val"] = galaxy_dataset.GalaxyDataset(
            label_cols=['label'], catalog=val_catalog.query('label >= 0'), transform=self.T_test)
        # 'val_supervised' is for supervised head (if present, otherwise ignored)
        self.data['val_supervised'] = galaxy_dataset.GalaxyDataset(
            label_cols=label_cols, catalog=val_catalog, transform=self.T_test)
        # 'test' is not used
        self.data["test"] = galaxy_dataset.GalaxyDataset(
            label_cols=['label'], catalog=test_catalog, transform=self.T_test)

        # only used for knn feature bank (and so has no effect other than val metric)
        self.data["labelled"] = galaxy_dataset.GalaxyDataset(
            label_cols=['label'], catalog=test_catalog.sample(10000),  transform=self.T_test)  # TODO temp


if __name__ == '__main__':

    import yaml
    import torch

    with open('config/byol/legs.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config['dataset'] = 'legs'
    config['debug'] = False
    config['num_workers'] = 1
    config['data'] = {'mu': 0, 'sig': 1, 'rotate': True, 'input_height': 64,
                      'precrop_size_ratio': 1.3, 'p_blur': 0., 'val_batch_size': 16}  # needed for _Eval
    config['p_blur'] = 0.  # TODO shouldn't this be under config['data']?
    # print(config)

    for datamodule in [Legs_DataModule(config=config)]:

        datamodule.setup()

        for (images, labels) in datamodule.train_dataloader():
            print(images[0].shape, labels.shape)  # [0] as list of views
            assert labels.min() >= 0
            break

        for (images, labels) in datamodule.val_dataloader():
            print(images[0].shape, labels.shape)  # [0] as list of views
            assert labels.min() >= 0
            break

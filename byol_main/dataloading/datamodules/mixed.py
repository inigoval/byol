# uses dataset from pytorch-galaxy-datasets/galaxy_zoo_2.py
# does not use the datamodule, as this code has different requirements (e.g. normalisation, augmentation choice, etc)
import logging
import numpy as np

import pandas as pd
# https://github.com/mwalmsley/pytorch-galaxy-datasets

from pytorch_galaxy_datasets import galaxy_dataset
from foundation.datasets import mixed, tidal

from byol_main.dataloading.datamodules import generic_galaxy


class Mixed_DataModule(generic_galaxy.Galaxy_DataModule):
    def __init__(self, config):
        super().__init__(
            config,
            mu=((0)),
            sig=((1))
        )

    def prepare_data(self):
        pass

    def setup(self, stage=None):

        label_cols, (train_catalog, val_catalog, test_catalog, unlabelled_catalog) = mixed.everything_all_dirichlet_with_rings(
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

        # 'val_supervised' is for supervised head (if present, otherwise ignored)
        assert not np.any(val_catalog[label_cols].isna())  # all vote counts should be 0 or counts
        self.data['val_supervised'] = galaxy_dataset.GalaxyDataset(
            label_cols=label_cols, catalog=val_catalog, transform=self.T_test)
        # 'test' is not used
        self.data["test"] = galaxy_dataset.GalaxyDataset(
            label_cols=['ring_label'], catalog=test_catalog, transform=self.T_test)

        if self.config['val_dataset'] == 'rings':
            
            ring_val_filtered = val_catalog.query('ring_label >= 0')
            ring_knn_targets =  galaxy_dataset.GalaxyDataset(
                label_cols=['ring_label'], catalog=ring_val_filtered.sample(min(len(ring_val_filtered), 10000)), transform=self.T_test)

            # using test here is a slight cheat as I wouldn't really know the labels yet, but equally I am training on the train set already
            # kinda imperfect either way
            # maybe I should use train after all - TODO ponder
            # (changing now will mess with current experiments)
            ring_test_filtered = test_catalog.query('ring_label >= 0')
            ring_knn_bank = galaxy_dataset.GalaxyDataset(
                label_cols=['ring_label'], catalog=ring_test_filtered.sample(min(len(ring_test_filtered), 10000)),  transform=self.T_test
            )

            self.data["val_knn"] = ring_knn_targets
            self.data["labelled"] = ring_knn_bank

        elif self.config['val_dataset'] == 'tidal':

            _, (tidal_train_catalog, tidal_val_catalog, _, _) = tidal.get_tidal_catalogs(
                self.path, self.config['debug'], download=True
            )
            tidal_knn_targets =  galaxy_dataset.GalaxyDataset(
                label_cols=['tidal_label'], catalog=tidal_val_catalog.query('tidal_label >= 0'), transform=self.T_test)

            # I can use "train" catalog here as never shown in contrastive/supervised manner
            tidal_knn_bank = galaxy_dataset.GalaxyDataset(
                label_cols=['tidal_label'], catalog=tidal_train_catalog,  transform=self.T_test
            )

            self.data["val_knn"] = tidal_knn_targets
            self.data["labelled"] = tidal_knn_bank

        else:

            raise ValueError('{} not recognised'.format(self.data["val_knn"]))

        # only used for knn feature bank (and so has no effect other than val metric)
        # also needs to be filtered to avoid missing labels
    
        # dataloader index is assumed to line up with val_knn dataloader index





if __name__ == '__main__':

    import yaml
    import torch

    logging.basicConfig(level=logging.INFO)

    torch.set_num_threads(24)
    logging.info('Threads: {}'.format(torch.get_num_threads()))

    with open('/share/nas2/walml/repos/byol/config/byol/legs.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config['dataset'] = 'mixed'
    config['debug'] = False
    config['num_workers'] = 20
    config['data'] = {'mu': 0, 'sig': 1, 'rotate': True, 'input_height': config['data']['input_height'],
                      'precrop_size_ratio': 1.3, 'p_blur': 0., 'val_batch_size': 512}  # needed for _Eval
    config['p_blur'] = 0.  # TODO shouldn't this be under config['data']?
    # print(config)
    config['val_dataset'] = 'rings'
    config['pin_memory'] = True
    config['persistent_workers'] = False
    config['type'] = 'byol_supervised'

    for datamodule in [Mixed_DataModule(config=config)]:

        datamodule.setup()

        logging.info('Checking image shapes - {}'.format(config['data']['input_height']))

        # for dataloader_idx, dataloader in enumerate(datamodule.val_dataloader()):
        #     logging.info('Val dataloader {}'.format(dataloader_idx))
        #     for images, _ in dataloader:
        #         if not (images.shape[2], images.shape[3]) == (config['data']['input_height'], config['data']['input_height']):
        #             raise ValueError(images.shape)
        
        # logging.info('All val images are correct shape')

        # for ((view_a, view_b), labels) in datamodule.train_dataloader():
        #     for view in [view_a, view_b]:
        #         if not (view.shape[2], view.shape[3]) == (config['data']['input_height'], config['data']['input_height']):
        #             raise ValueError(view.shape)
        # logging.info('All train images are correct shape')


        for ((view_a, view_b), labels) in datamodule.data['labelled']:

            for view in [view_a, view_b]:
                if not (view.shape[2], view.shape[3]) == (config['data']['input_height'], config['data']['input_height']):
                    raise ValueError(view.shape)

        logging.info('All labelled images are correct shape')

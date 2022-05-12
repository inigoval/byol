# uses dataset from pytorch-galaxy-datasets/galaxy_zoo_2.py
# does not use the datamodule, as this code has different requirements (e.g. normalisation, augmentation choice, etc)
import logging

import numpy as np
# https://github.com/mwalmsley/pytorch-galaxy-datasets
from pytorch_galaxy_datasets import galaxy_dataset
from pytorch_galaxy_datasets.prepared_datasets import gz2_setup
from sklearn.model_selection import train_test_split

from foundation.datasets import gz2, dataset_utils

from byol_main.dataloading.base_dm import Base_DataModule_Eval, Base_DataModule

# config arg used by super() only for now, but could use to modify behaviour
class GZ2_DataModule(Base_DataModule):  # not the same as in pytorch-galaxy-datasets
    def __init__(self, config):
        super().__init__(
            config, 
            # recalculate
            # mu=(0,), 
            # sig=(1,))
            # standard smooth/featured
            mu=(0.05309092253446579, 0.04963503032922745, 0.03850013017654419),
            sig=(0.17045791447162628, 0.17045791447162628, 0.17045791447162628)
        )

    def prepare_data(self):
        # will just download both anyway
        gz2_setup(self.path, train=True, download=True)
        # gz2_setup(self.path, train=False, download=True)

    def setup(self, stage=None):  # stage by ptl convention
        self.T_train.n_views = 1

        # TODO control with config once there are other options?
        train_and_val_catalog, test_catalog = gz2.gz2_valid_labels(self.path, self.config['debug'])  
        
        train_catalog, val_catalog = train_test_split(train_and_val_catalog, train_size=0.8)

        dataset_utils.check_dummy_metrics(val_catalog['label'])

        # subset as kinda slow as single-threaded here, possibly
        # will skip if explicitly set in __init__ above (hardcoded from prev. calculation)
        logging.info('Loading 5k subset of train dataset to adjust mu, sigma') 
        D_train = galaxy_dataset.GalaxyDataset(train_catalog.sample(5000), label_cols=['label'], transform=self.T_train)
        self.update_transforms(D_train)

        # Re-initialise dataset with new mu and sig values
        self.data["train"] = galaxy_dataset.GalaxyDataset(catalog=train_catalog, label_cols=['label'], transform=self.T_train)
        # Initialise individual datasets with test transform (for evaluation)
        self.data["val"] = galaxy_dataset.GalaxyDataset(label_cols=['label'], catalog=val_catalog, transform=self.T_test)
        self.data["test"] = galaxy_dataset.GalaxyDataset(label_cols=['label'], catalog=test_catalog, transform=self.T_test)
        self.data["labelled"] = galaxy_dataset.GalaxyDataset(label_cols=['label'], catalog=train_catalog.sample(10000),  transform=self.T_test) 


# config arg not used
class GZ2_DataModule_Eval(Base_DataModule_Eval):
    def __init__(self, encoder, config):
        super().__init__(encoder, config)

    def setup(self, stage=None):

        # assumes superclass set self.config = config
        temporary_datamodule = GZ2_DataModule(self.config)
        temporary_datamodule.setup()

        # now re-use exactly the same data from temporary datamodule (now we will actually use the labels)
        self.data['train'] = temporary_datamodule.data['train']
        self.data['val'] = temporary_datamodule.data['val']
        self.data['test'] = temporary_datamodule.data['test']
        self.data['labelled'] = temporary_datamodule.data['labelled']


if __name__ == '__main__':

    import yaml
    import torch

    with open('config/byol/gz2.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config['dataset'] = 'gz2'
    config['debug'] = True
    config['num_workers'] = 1
    config['data'] = {'mu': 0, 'sig': 1, 'rotate': True, 'input_height': 64, 'precrop_size_ratio': 1.3, 'p_blur': 0., 'val_batch_size': 16} # needed for _Eval
    config['p_blur'] = 0.  # TODO shouldn't this be under config['data']?
    # print(config)

    for datamodule in [GZ2_DataModule(config=config), GZ2_DataModule_Eval(config=config, encoder=lambda x: torch.from_numpy(np.random.rand(len(x), 512)))]:

        datamodule.setup()

        for (images, labels) in datamodule.train_dataloader():
            print(images[0].shape, labels.shape)  # [0] as list of views
            assert labels.min() >= 0
            break

        for (images, labels) in datamodule.val_dataloader():
            print(images[0].shape, labels.shape)  # [0] as list of views
            assert labels.min() >= 0
            break

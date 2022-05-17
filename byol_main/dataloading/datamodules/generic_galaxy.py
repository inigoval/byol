from distutils.command.config import config
import logging
from typing import List

import pandas as pd

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from pytorch_galaxy_datasets import galaxy_dataset
from foundation.datasets import dataset_utils
from byol_main.dataloading.base_dm import Base_DataModule, Base_DataModule_Eval
from byol_main.dataloading.transforms import ReduceView


class Galaxy_DataModule(Base_DataModule):

    def __init__(self, config, mu=(0,), sig=(1,)):
        super().__init__(config, mu, sig)

    # TODO could refactor up further
    def adjust_mu_and_std_from_subset(self, train_catalog, label_cols, size):
        # subset as kinda slow as single-threaded here, possibly
        logging.info('Loading 5k subset of train dataset to adjust mu, sigma')
        D_train = galaxy_dataset.GalaxyDataset(
            train_catalog.sample(size), label_cols=label_cols, transform=self.T_train)
        self.update_transforms(D_train)

    def create_transformed_datasets_from_catalogs(
        self,
        training_label_cols: List,
        train_catalog: pd.DataFrame,
        val_catalog: pd.DataFrame,
        test_catalog: pd.DataFrame,
        unlabelled_catalog=pd.DataFrame()  # optional, not all datasets have unlabelled data
        ):
        """
        Assign keys of self.data (e.g. self.data['train']) to GalaxyDataset's created from the catalogs above.
        These are then used for training BYOL ('train') or for knn evaluation ('val', 'labelled')

        All catalogs are assumed to have label column(s) even if e.g. -1 everywhere.
        
        train_catalog should have training_label_cols (e.g. the GZ vote counts, or simply ['label']).
        Expected to be either classification target or vote counts.
        If no supervised head, these are ignored.

        val and labelled catalogs should have a single 'label' column with integer classification targets
        these are used for knn validation
        (ideally, labelled catalog should exclude val catalog to avoid cheating TODO)

        Args:
            training_label_cols (list): Catalog columns to use as targets during training. Only used if BYOL includes supervised head
            train_catalog (pd.DataFrame): _description_
            val_catalog (pd.DataFrame): _description_
            test_catalog (pd.DataFrame): _description_
            unlabelled_catalog (pd.DataFrame): optional (default empty dataframe). If passed, added to train_catalog for constrastive training. Ignored by supervised head.
        """
        # Re-initialise dataset with new mu and sig values
        self.data["train"] = galaxy_dataset.GalaxyDataset(
            catalog=pd.concat([train_catalog, unlabelled_catalog]), label_cols=training_label_cols, transform=self.T_train)
        # Initialise individual datasets with test transform (for evaluation)
        # these must always be classification problems with a int label, for knn to make sense
        # val used for knn input, searched within feature/target bank
        self.data["val"] = galaxy_dataset.GalaxyDataset(
            label_cols=['label'], catalog=val_catalog, transform=self.T_test)
        # test not currently used
        self.data["test"] = galaxy_dataset.GalaxyDataset(
            label_cols=['label'], catalog=test_catalog, transform=self.T_test)
        # labelled unpacked into feature_bank, target_bank, for knn eval
        self.data["labelled"] = galaxy_dataset.GalaxyDataset(
            label_cols=['label'], catalog=train_catalog.sample(10000),  transform=self.T_test)

        dataset_utils.check_dummy_metrics(val_catalog['label'])
        # logging.info('Catalog sizes: train/val={}, test={}, unlabelled={}'.format(
        #     len(train_and_val_catalog), len(test_catalog), len(unlabelled_catalog)))
        # logging.info('Class balance: {}'.format(
        #     train_and_val_catalog['label'].value_counts(normalize=True)))

        


    def prepare_data(self):
        raise NotImplementedError

    def setup(self, stage=None):
        raise NotImplementedError



class GalaxyInferenceDataModule(pl.LightningDataModule):

    def __init__(self, encoder, catalog, config):
        super().__init__()
        # self.mu, self.sig = mu, sig

        # config must include mu, sig
        # used by ReduceView.pre_normalize

        # train flag controls if rotations/flips will be applied
        self.T_test = ReduceView(encoder, config, train=False)

        self.id_strs = list(catalog['id_str'].values)

        self.data = {
            'predict': galaxy_dataset.GalaxyDataset(
            label_cols=['label'], catalog=catalog, transform=self.T_test)
        }  

    def predict_dataloader(self):
        n_workers = self.config["num_workers"]
        batch_size = self.config["linear"]["batch_size"]
        loader = DataLoader(
            self.data["predict"],
            batch_size,
            shuffle=False,  # crucial
            num_workers=n_workers,
            prefetch_factor=self.config['prefetch_factor'],
        )
        return loader 



class Galaxy_DataModule_Eval(Base_DataModule_Eval):
    def __init__(self, encoder, config):
        super().__init__(encoder, config)

    def setup(self, stage=None):
        raise NotImplementedError  # I never use the linear eval anyway

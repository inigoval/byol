import logging

from pytorch_galaxy_datasets import galaxy_dataset

from foundation.datasets import dataset_utils

from byol_main.dataloading.base_dm import Base_DataModule, Base_DataModule_Eval


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

    def create_transformed_datasets_from_catalogs(self, label_cols, train_catalog, val_catalog, test_catalog):
        # Re-initialise dataset with new mu and sig values
        self.data["train"] = galaxy_dataset.GalaxyDataset(
            catalog=train_catalog, label_cols=label_cols, transform=self.T_train)
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


class Galaxy_DataModule_Eval(Base_DataModule_Eval):
    def __init__(self, encoder, config):
        super().__init__(encoder, config)

    def setup(self, stage=None):
        raise NotImplementedError  # I never use the linear eval anyway

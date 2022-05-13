# uses dataset from pytorch-galaxy-datasets/galaxy_zoo_2.py
# does not use the datamodule, as this code has different requirements (e.g. normalisation, augmentation choice, etc)

# https://github.com/mwalmsley/pytorch-galaxy-datasets

from pytorch_galaxy_datasets.prepared_datasets import decals_dr5_setup

from foundation.datasets import decals_dr5

from byol_main.dataloading.base_dm import Base_DataModule_Eval
from byol_main.dataloading.datamodules import generic_galaxy

class Decals_DataModule(generic_galaxy.Galaxy_DataModule):
    def __init__(self, config):
        super().__init__(config, mu=(0,), sig=(1,))  # TODO specify explicitly

    
    def prepare_data(self):
        # will actually just download both anyway, but for completeness
        # decals_dr5_setup(self.path, train=True, download=True)
        # decals_dr5_setup(self.path, train=False, download=True)
        pass  # setup() will download anyway, and only ever runs on one node


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



# # config arg not used
# class Decals_DataModule_Eval(Base_DataModule_Eval):
#     def __init__(self, encoder, config):
#         super().__init__(encoder, config)

#     def setup(self, stage=None):

#         # assumes superclass set self.config = config
#         temporary_datamodule = Decals_DataModule(self.config)
#         # eval must always be a classification problem
#         self.config = self.config.copy()  # avoid mutate-by-ref for other objects
#         self.config['labels'] = 'classification'
#         temporary_datamodule.setup()

#         # now re-use exactly the same data from temporary datamodule (now we will actually use the labels)
#         self.data['train'] = temporary_datamodule.data['train']
#         self.data['val'] = temporary_datamodule.data['val']
#         self.data['test'] = temporary_datamodule.data['test']
#         self.data['labelled'] = temporary_datamodule.data['labelled']

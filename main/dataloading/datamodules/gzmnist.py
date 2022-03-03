from dataloading.base_dm import Base_DataModule_Eval, Base_DataModule
from galaxy_mnist import GalaxyMNIST


class GalaxyMNIST_DataModule(Base_DataModule):
    def __init__(self, config):
        super().__init__(config, mu=(0,), sig=(1,))

    def prepare_data(self):
        GalaxyMNIST(self.path, train=False, download=True)
        GalaxyMNIST(self.path, train=True, download=True)

    def setup(self):
        self.T_train.n_views = 1
        D_train = GalaxyMNIST(self.path, train=True, download=True, transform=self.T_train)

        self.update_transforms(D_train)

        # Re-initialise dataset with new mu and sig values
        self.data["train"] = GalaxyMNIST(
            self.path, train=True, download=True, transform=self.T_train
        )
        # Initialise individual datasets with test transform (for evaluation)
        self.data["val"] = GalaxyMNIST(self.path, train=False, transform=self.T_test)
        self.data["test"] = GalaxyMNIST(self.path, train=False, transform=self.T_test)
        self.data["l"] = GalaxyMNIST(self.path, train=True, transform=self.T_test)


class GalaxyMNIST_DataModule_Eval(Base_DataModule_Eval):
    def __init__(self, encoder, config):
        super().__init__(encoder, config)

    def setup(self):
        # Initialise individual datasets with identity transform (for evaluation)
        D_train = GalaxyMNIST(self.path, train=True, download=True, transform=self.T_train)

        self.update_transforms(D_train)

        self.data["train"] = GalaxyMNIST(
            self.path, train=True, download=True, transform=self.T_train
        )
        self.data["val"] = GalaxyMNIST(self.path, train=False, transform=self.T_test)
        self.data["test"] = GalaxyMNIST(self.path, train=False, transform=self.T_test)
        self.data["l"] = GalaxyMNIST(self.path, train=True, transform=self.T_test)

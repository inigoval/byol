from dataloading.base_dm import Base_DataModule_Eval, Base_DataModule
from galaxy_mnist import GalaxyMNIST


class GalaxyMNIST_DataModule(Base_DataModule):
    def __init__(self, config):
        super().__init__(config, mu=(0, 0, 0), sig=(1, 1, 1))

    def prepare_data(self):
        GalaxyMNIST(self.path, train=False, download=True)
        GalaxyMNIST(self.path, train=True, download=True)

    def setup(self):
        D_train = GalaxyMNIST(self.path, train=True, download=True)
        self.update_transforms(D_train)

        self.data["train"] = GalaxyMNIST(self.path, train=True, transform=self.T_train)
        self.data["l"] = GalaxyMNIST(self.path, train=True, transform=self.T_train)
        self.data["val"] = GalaxyMNIST(self.path, train=False, transform=self.T_train)
        self.data["test"] = GalaxyMNIST(self.path, train=False, transform=self.T_train)


class GalaxyMNIST_DataModule_Eval(Base_DataModule_Eval):
    def __init__(self, encoder, config):
        super().__init__(encoder, config, **norms)

    def setup(self):
        # Initialise individual datasets with identity transform (for evaluation)
        D_train = ImageFolder(self.path / "train", transform=self.T_train)
        self.update_transforms(D_train)
        self.data["train"] = ImageFolder(self.path / "train", transform=self.T_train)
        self.data["val"] = ImageFolder(self.path / "val", transform=self.T_test)

from dataloading.base_dm import Base_DataModule_Eval, Base_DataModule
from torchvision.datasets import ImageFolder
from dataloading.utils import _get_imagenet_norms


class Imagenette_DataModule(Base_DataModule):
    def __init__(self, config):
        norms = _get_imagenet_norms()
        super().__init__(config, **norms)

    def setup(self):
        self.data["train"] = ImageFolder(self.path / "train", transform=self.T_train)
        self.data["l"] = ImageFolder(self.path / "train", transform=self.T_test)

        self.data["val"] = ImageFolder(self.path / "val", transform=self.T_test)
        self.data["test"] = ImageFolder(self.path / "val", transform=self.T_test)


class Imagenette_DataModule_Eval(Base_DataModule_Eval):
    def __init__(self, encoder, config):
        norms = _get_imagenet_norms()
        super().__init__(encoder, config, **norms)

    def setup(self):
        # Initialise individual datasets with identity transform (for evaluation)
        D_train = ImageFolder(self.path / "train", transform=self.T_train)
        self.update_transforms(D_train)
        self.data["train"] = ImageFolder(self.path / "train", transform=self.T_train)
        self.data["val"] = ImageFolder(self.path / "val", transform=self.T_test)

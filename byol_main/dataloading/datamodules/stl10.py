from dataloading.base_dm import Base_DataModule_Eval, Base_DataModule
from dataloading.utils import _get_imagenet_norms
from torchvision.datasets import STL10


class STL10_DataModule(Base_DataModule):
    def __init__(self, config):
        norms = _get_imagenet_norms()
        super().__init__(config, **norms)

    def prepare_data(self):
        STL10(root=self.path, split="train+unlabeled", download=True)
        STL10(root=self.path, split="test", download=True)

    def setup(self):
        path = self.path
        T_train = self.T_train
        T_test = self.T_test

        self.data["train"] = STL10(root=path, split="train+unlabeled", transform=T_train)
        self.data["u"] = STL10(root=path, split="unlabeled", transform=T_test)
        self.data["labelled"] = STL10(root=path, split="train", transform=T_test)
        self.data["val"] = STL10(root=path, split="test", transform=T_test)
        self.data["test"] = STL10(root=path, split="test", transform=T_test)


class STL10_DataModule_Eval(Base_DataModule_Eval):
    def __init__(self, encoder, config):
        super().__init__(encoder, config)

    def setup(self):
        # Initialise individual datasets with identity transform (for evaluation)

        path = self.path
        T_train = self.T_train
        T_test = self.T_test

        D_train = STL10(root=path, split="train+unlabeled", transform=T_train)

        self.update_transforms(D_train)

        self.data["train"] = STL10(root=path, split="train", transform=T_train)
        self.data["u"] = STL10(root=path, split="unlabeled", transform=T_test)
        self.data["val"] = STL10(root=path, split="test", transform=T_test)
        self.data["test"] = STL10(root=path, split="test", transform=T_test)

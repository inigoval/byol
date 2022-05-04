from dataloading.base_dm import Base_DataModule_Eval, Base_DataModule
from torchvision.datasets import CIFAR10
from dataloading.utils import _get_cifar10_norms


class CIFAR10_DataModule(Base_DataModule):
    def __init__(self, config):
        norms = _get_cifar10_norms()
        super().__init__(config, **norms)

    def prepare_data(self):
        CIFAR10(self.path, train=True, download=True)
        CIFAR10(self.path, train=False, download=True)

    def setup(self):
        self.data["train"] = CIFAR10(self.path, train=True, transform=self.T_train)
        self.data["labelled"] = CIFAR10(self.path, train=True, transform=self.T_test)
        self.data["val"] = CIFAR10(self.path, train=False, transform=self.T_test)
        self.data["test"] = CIFAR10(self.path, train=False, transform=self.T_test)


class CIFAR10_DataModule_Eval(Base_DataModule_Eval):
    def __init__(self, encoder, config):
        norms = _get_cifar10_norms()
        super().__init__(encoder, config, **norms)

    def setup(self):
        # Initialise individual datasets with identity transform (for evaluation)
        D_train = CIFAR10(self.path, train=True, transform=self.T_train)
        self.update_transforms(D_train)
        self.data["train"] = CIFAR10(self.path, train=True, transform=self.T_train)
        self.data["val"] = CIFAR10(self.path, train=False, transform=self.T_test)
        self.data["test"] = CIFAR10(self.path, train=False, transform=self.T_test)

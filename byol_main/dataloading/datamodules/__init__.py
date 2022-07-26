from .imagenette import Imagenette_DataModule_Eval, Imagenette_DataModule
from .gzmnist import GalaxyMNIST_DataModule, GalaxyMNIST_DataModule_Eval
from .gz2 import GZ2_DataModule, GZ2_DataModule_Eval
from .rgz import RGZ_DataModule, RGZ_DataModule_Eval, RGZ_DataModule_Supervised
from .decals_dr5 import Decals_DataModule
from .legs import Legs_DataModule
from .mixed import Mixed_DataModule
from .cifar10 import CIFAR10_DataModule_Eval, CIFAR10_DataModule
from .generic_galaxy import Galaxy_DataModule, Galaxy_DataModule_Eval

# Load data and record hyperparameters #
datasets = {
    "imagenette": {
        "pretrain": Imagenette_DataModule,
        "linear": Imagenette_DataModule_Eval,
    },
    "gzmnist": {
        "pretrain": GalaxyMNIST_DataModule,
        "linear": Galaxy_DataModule_Eval,
    },
    "gz2": {
        "pretrain": GZ2_DataModule,
        "linear": Galaxy_DataModule_Eval,
    },
    "decals_dr5": {
        "pretrain": Decals_DataModule,
        "linear": Galaxy_DataModule_Eval,
    },
    "legs": {
        "pretrain": Legs_DataModule,
        "linear": Galaxy_DataModule_Eval,
    },
    "legs_and_rings": {
        "pretrain": Legs_DataModule,
        "linear": Galaxy_DataModule_Eval,  # not used
    },
    "mixed": {
        "pretrain": Mixed_DataModule,
        "linear": Galaxy_DataModule_Eval,  # not used
    },
    "rgz": {
        "pretrain": RGZ_DataModule,
        "linear": RGZ_DataModule_Eval,
    },
    # "stl10": {
    #     "pretrain": STL10_DataModule,
    #     "linear": STL10_DataModule_Eval,
    # },
    "cifar10": {
        "pretrain": CIFAR10_DataModule,
        "linear": CIFAR10_DataModule_Eval,
    },
}
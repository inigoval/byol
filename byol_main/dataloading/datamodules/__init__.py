# from .imagenette import Imagenette_DataModule_Eval, Imagenette_DataModule
# from .cifar10 import CIFAR10_DataModule_Eval, CIFAR10_DataModule
# from .stl10 import STL10_DataModule

# # from .gz2 import GZ2_DataModule, GZ2_DataModule_Eval
# from .rgz import RGZ_DataModule, RGZ_DataModule_Eval, RGZ_DataModule_Supervised
# from .gzmnist import GalaxyMNIST_DataModule, GalaxyMNIST_DataModule_Eval

from .vision import STL10_DataModule, Imagenette_DataModule

# from .decals_dr5 import Decals_DataModule
# from .legs import Legs_DataModule
# from .mixed import Mixed_DataModule
# from .generic_galaxy import Galaxy_DataModule, Galaxy_DataModule_Eval

datasets = {"stl10": STL10_DataModule, "imagenette": Imagenette_DataModule}

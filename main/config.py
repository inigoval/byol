import yaml
from paths import Path_Handler

# Define paths
paths = Path_Handler()
path_dict = paths._dict()


def load_config():
    """Helper function to load yaml config file, convert to python dictionary and return."""
    path = path_dict["root"] / "config.yml"
    with open(path, "r") as ymlconfig:
        config = yaml.load(ymlconfig, Loader=yaml.FullLoader)
    return config


def update_config(config):
    """Update config with values requiring initialisation of config"""

    # Learning rate scaling from BYOL
    # config["lr"] = config["lr"] * config["batch_size"] / 256

    # Adjust parameters for different data-sets
    if config["dataset"] == "imagenette":
        config["data"]["color_channels"] = 3
        config["data"]["classes"] = 10
        config["data"]["input_height"] = 128
        config["data"]["center_crop_size"] = 128

    if config["dataset"] == "rgz":
        config["data"]["color_channels"] = 1
        config["data"]["classes"] = 2
        config["data"]["input_height"] = config["center_crop_size"]
        config["data"]["rotate"] = True

    # Adjust parameters for different data-sets
    if config["dataset"] == "stl10":
        config["data"]["color_channels"] = 3
        config["data"]["classes"] = 10
        config["data"]["input_height"] = 96
        config["data"]["center_crop_size"] = 128

    if config["dataset"] == "cifar10":
        config["data"]["color_channels"] = 3
        config["data"]["classes"] = 10
        config["data"]["input_height"] = 32
        config["data"]["center_crop_size"] = 32

    if config["dataset"] == "gzmnist":
        config["data"]["color_channels"] = 3
        config["data"]["classes"] = 4
        config["data"]["input_height"] = 64

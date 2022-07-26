import yaml
from byol_main.paths import Path_Handler

# Define paths
paths = Path_Handler()
path_dict = paths._dict()


def load_config():
    """Helper function to load yaml config file, convert to python dictionary and return."""

    # load global config
    global_path = path_dict["config"] / "global.yml"
    with open(global_path, "r") as ymlconfig:
        config = yaml.load(ymlconfig, Loader=yaml.FullLoader)

    dataset = config["dataset"]
    type = config["type"]
    path = path_dict["config"] / type / f"{dataset}.yml"

    # load data-set specific config
    with open(path, "r") as ymlconfig:
        dataset_config = yaml.load(ymlconfig, Loader=yaml.FullLoader)

    # if loading a benchmark, use load the specific config
    preset = dataset_config["preset"]
    if preset:
        path = path_dict["config"] / type / f"{dataset}-{preset}.yml"
        with open(path, "r") as ymlconfig:
            dataset_config = yaml.load(ymlconfig, Loader=yaml.FullLoader)

    # combine global with data-set specific config. dataset config has priority
    config.update(dataset_config)

    return config


def update_config(config):
    """Update config with values requiring initialisation of config"""

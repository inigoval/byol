import yaml

from torch.optim import Adam, SGD

from byol.paths import Path_Handler

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
    path = path_dict["config"] / f"{dataset}.yml"

    # load data-set specific config
    with open(path, "r") as ymlconfig:
        dataset_config = yaml.load(ymlconfig, Loader=yaml.FullLoader)

    # if loading a benchmark, use load the specific config
    preset = dataset_config["preset"]
    if preset != "none":
        path = path_dict["config"] / f"{dataset}-{preset}.yml"
        with open(path, "r") as ymlconfig:
            dataset_config = yaml.load(ymlconfig, Loader=yaml.FullLoader)

    # combine global with data-set specific config. dataset config has priority
    config.update(dataset_config)

    return config


def update_config(config):
    """Update config with values requiring initialisation of config"""

    # Create unpackable dictionary for logistic regression model
    optimizers = {"adam": Adam, "sgd": SGD}

    # Generic dataloading settings
    dataloading = {
        "num_workers": config["dataloading"]["num_workers"],
        "pin_memory": config["dataloading"]["pin_memory"],
        "prefetch_factor": config["dataloading"]["prefetch_factor"],
        "persistent_workers": config["dataloading"]["persistent_workers"],
    }

    # Create unpackable dictionary for training dataloaders
    config["train_dataloader"] = {
        "shuffle": False,
        "batch_size": config["data"]["batch_size"],
        **dataloading,
    }

    # Create unpackable dictionary for validation dataloaders
    config["val_dataloader"] = {
        "shuffle": False,
        "batch_size": config["dataloading"]["val_batch_size"],
        **dataloading,
    }

    config["model"]["optimizer"]["batch_size"] = config["data"]["batch_size"]

    # Set finetuning config to values from rest of config
    config["model"]["architecture"]["n_c"] = config["data"]["color_channels"]
    config["finetune"]["n_classes"] = config["data"]["classes"]
    config["finetune"]["dim"] = config["model"]["architecture"]["features"]


def load_config_finetune():
    """Helper function to load yaml config file, convert to python dictionary and return."""

    path = path_dict["config"] / "finetune.yml"

    # load data-set specific config
    with open(path, "r") as ymlconfig:
        config = yaml.load(ymlconfig, Loader=yaml.FullLoader)

    if config["finetune"]["preset"] == "optimal":
        path = path_dict["config"] / "finetune_optimal.yml"
        with open(path, "r") as ymlconfig:
            config = yaml.load(ymlconfig, Loader=yaml.FullLoader)

    return config

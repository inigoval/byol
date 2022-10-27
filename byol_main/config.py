import yaml

from torch.optim import Adam, SGD

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
    if preset != "none":
        path = path_dict["config"] / type / f"{dataset}-{preset}.yml"
        with open(path, "r") as ymlconfig:
            dataset_config = yaml.load(ymlconfig, Loader=yaml.FullLoader)

    # combine global with data-set specific config. dataset config has priority
    config.update(dataset_config)

    return config


def update_config(config):
    """Update config with values requiring initialisation of config"""

    # Create unpackable dictionary for logistic regression model
    optimizers = {"adam": Adam, "sgd": SGD}
    config["logreg"] = {
        # "input_dim": config["model"]["features"],
        "num_classes": config["data"]["classes"],
        "learning_rate": config["linear"]["lr"],
        "optimizer": optimizers[config["linear"]["opt"]],
    }

    if optimizers[config["linear"]["opt"]] == SGD:
        config["logreg"]["momentum"] = config["linear"]["momentum"]
        config["logreg"]["nesterov"] = config["linear"]["nesterov"]
        config["logreg"]["weight_decay"] = config["linear"]["weight_decay"]

    # Generic dataloading settings
    dataloading = {
        "num_workers": config["dataloading"]["num_workers"],
        "pin_memory": config["dataloading"]["pin_memory"],
        "prefetch_factor": config["dataloading"]["prefetch_factor"],
        "persistent_workers": config["dataloading"]["persistent_workers"],
    }

    # Create unpackable dictionary for logreg training dataloader
    config["logreg_dataloader"] = {
        "batch_size": config["linear"]["batch_size"],
        "shuffle": True,
        **dataloading,
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

    config["finetune"]["n_classes"] = config["data"]["classes"]

    if config["type"] == "mae":
        config["finetune"]["dim"] = config["model"]["vit"]["dim"]
    elif config["type"] == "byol":
        config["finetune"]["dim"] = config["model"]["features"]

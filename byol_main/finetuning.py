import wandb
import pytorch_lightning as pl
import logging

from pathlib import Path

from dataloading.datamodules import finetune_datasets
from paths import Path_Handler
from .config import load_config, update_config
from finetune import finetune
from byol import byol

RUN_ID = Path("")


def main():

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    path_dict = Path_Handler()._dict()

    model = byol.load_from_checkpoint("model.ckpt")
    # model = byol.load_from_checkpoint(path_dict['files'] / RUN_ID / "model.ckpt"])

    ## Load up config from model ##
    config = model.config

    # Load finetuning and trainer config from config file
    config["finetune"] = load_config()["finetune"]
    config["trainer"] = load_config()["trainer"]

    wandb.init(project=f"{config['project_name']}_finetune")

    path_dict = Path_Handler()._dict()

    logger = pl.loggers.WandbLogger(
        project=config["project_name"],
        save_dir=path_dict["files"] / str(wandb.run.id),
        reinit=True,
        config=config,
    )

    ## Run pretraining ##
    for seed in range(config["finetune"]["iterations"]):
        config["finetune"]["seed"] = seed
        finetune_datamodule = finetune_datasets[config["dataset"]](config)
        finetune(config, model.encoder, finetune_datamodule, logger)

    logger.experiment.finish()


if __name__ == "__main__":

    main()

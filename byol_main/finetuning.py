import wandb
import pytorch_lightning as pl
import logging

from pathlib import Path

from dataloading.datamodules import finetune_datasets
from paths import Path_Handler
from config import load_config, update_config, load_config_finetune
from finetune.finetune import run_finetuning
from finetune.dataloading import finetune_datasets
from byol import BYOL


def main():

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    # Load paths
    path_dict = Path_Handler()._dict()

    # Load up finetuning config
    config_finetune = load_config_finetune()

    ## Run pretraining ##
    for seed in range(config_finetune["finetune"]["iterations"]):

        if config_finetune["finetune"]["run_id"].lower() is not "none":
            experiment_dir = path_dict["files"] / config_finetune["finetune"]["run_id"] / "checkpoints"
            model = BYOL.load_from_checkpoint(experiment_dir / "last.ckpt")
        else:
            model = BYOL.load_from_checkpoint("model.ckpt")

        ## Load up config from model to save correct hparams for easy logging ##
        config = model.config
        config.update(config_finetune)
        config["finetune"]["dim"] = model.encoder.dim
        project_name = f"{config['project_name']}_finetune"

        config["finetune"]["seed"] = seed
        pl.seed_everything(seed)

        # Initiate wandb logging
        wandb.init(project=project_name, config=config)

        logger = pl.loggers.WandbLogger(
            project=project_name,
            save_dir=path_dict["files"] / str(wandb.run.id),
            reinit=True,
            config=config,
        )

        finetune_datamodule = finetune_datasets[config["dataset"]](config)
        run_finetuning(config, model.encoder, finetune_datamodule, logger)
        logger.experiment.finish()


if __name__ == "__main__":

    main()

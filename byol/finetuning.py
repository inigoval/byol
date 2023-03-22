import wandb
import pytorch_lightning as pl
import logging

from pathlib import Path

from paths import Path_Handler
from finetune.main import run_finetuning
from finetune.dataloading import finetune_datasets
from config import load_config, update_config, load_config_finetune
from models import BYOL
from architectures.models import MLP


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    # Load paths
    path_dict = Path_Handler()._dict()

    # Load up finetuning config
    config_finetune = load_config_finetune()

    ## Run finetuning ##
    for seed in range(config_finetune["finetune"]["iterations"]):
        # for seed in range(1, 10):

        if config_finetune["finetune"]["run_id"].lower() != "none":
            experiment_dir = path_dict["files"] / config_finetune["finetune"]["run_id"] / "checkpoints"
            model = BYOL.load_from_checkpoint(experiment_dir / "last.ckpt")
        else:
            model = BYOL.load_from_checkpoint("model.ckpt")

        ## Load up config from model to save correct hparams for easy logging ##
        config = model.config
        config.update(config_finetune)
        config["finetune"]["dim"] = model.encoder.dim
        # project_name = f"{config['project_name']}_finetune"
        # project_name = "BYOL_LDecay_finetune"
        # project_name = "BYOL_LabelVolume_finetune"
        # project_name = "BYOL_nlayers_finetune"
        project_name = "BYOL_laptoptest_finetune"

        config["finetune"]["seed"] = seed
        pl.seed_everything(seed)

        # Initiate wandb logging
        wandb.init(project=project_name, config=config)

        logger = pl.loggers.WandbLogger(
            project=project_name,
            save_dir=path_dict["files"] / "finetune" / str(wandb.run.id),
            reinit=True,
            config=config,
        )

        finetune_datamodule = finetune_datasets[config["dataset"]](config)
        run_finetuning(config, model.encoder, finetune_datamodule, logger)
        logger.experiment.finish()
        wandb.finish()


if __name__ == "__main__":
    main()

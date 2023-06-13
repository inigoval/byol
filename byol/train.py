import wandb
import pytorch_lightning as pl
import logging
import torch

from pytorch_lightning.callbacks import LearningRateMonitor

from models import BYOL
from config import load_config, update_config
from datamodules import RGZ_DataModule
from paths import Path_Handler, create_path
from finetuning import run_finetuning
from datamodules import RGZ_DataModule, RGZ_DataModule_Finetune

# TODO put elsewhere
# https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch-lightning/Profile_PyTorch_Code.ipynb#scrollTo=qRoUXZdtJIUD


def run_contrastive_pretraining(config, datamodule, wandb_logger):
    pl.seed_everything(config["seed"])

    # Save model for test evaluation
    # TODO might be better to use val/supervised_loss when available
    loss_to_monitor = "train/loss"

    checkpoint_mode = {
        "min_loss": {"mode": "min", "monitor": loss_to_monitor},
        "last": {"monitor": None},
    }
    ## Creates experiment path if it doesn't exist already ##
    experiment_dir = config["files"] / config["run_id"]
    create_path(experiment_dir)

    ## Initialise checkpoint ##
    pretrain_checkpoint = pl.callbacks.ModelCheckpoint(
        # **checkpoint_mode[config["evaluation"]["checkpoint_mode"]],
        monitor=None,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        auto_insert_metric_name=False,
        verbose=True,
        dirpath=experiment_dir / "checkpoints",
        save_last=True,
        # e.g. byol/files/(run_id)/checkpoints/12-344-18.134.ckpt.
        # filename="{epoch}-{step}-{loss_to_monitor:.4f}",  # filename may not work here TODO
        filename="model",
        save_weights_only=True,
    )
    logging.info(f"checkpoint monitoring: {checkpoint_mode[config['evaluation']['checkpoint_mode']]}")

    ## Initialise callbacks ##
    callbacks = [pretrain_checkpoint]

    # add learning rate monitor, only supported with a logger
    if wandb_logger is not None:
        # change to step, may be slow
        callbacks += [LearningRateMonitor(logging_interval="step")]

    # if config['profiler'] == 'kineto':
    # callbacks += [profiler_callback]

    logging.info(f"Threads: {torch.get_num_threads()}")

    ## Initialise pytorch lightning trainer ##
    pre_trainer = pl.Trainer(
        **config["trainer"],
        max_epochs=config["model"]["n_epochs"],
        check_val_every_n_epoch=config["evaluation"]["check_val_every_n_epoch"],
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=200,
        # max_steps = 200  # TODO temp
    )

    # Initialise model #
    model = BYOL(config)

    # Train model #
    pre_trainer.fit(model, datamodule)
    pre_trainer.test(model, dataloaders=datamodule)

    return pretrain_checkpoint, model


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    ## Load up config from yml files ##
    config = load_config()
    update_config(config)

    wandb.init(project=config["project_name"])
    config["run_id"] = str(wandb.run.id)

    paths = Path_Handler()._dict()

    wandb_logger = pl.loggers.WandbLogger(
        project=config["project_name"],
        # and will then add e.g. run-20220513_122412-l5ikqywp automatically
        save_dir=paths["files"] / config["run_id"],
        # log_model="True",
        # reinit=True,
        config=config,
    )

    config["files"] = paths["files"]

    datamodule = RGZ_DataModule(
        path=paths["rgz"],
        batch_size=config["data"]["batch_size"],
        center_crop=config["augmentations"]["center_crop"],
        random_crop=config["augmentations"]["random_crop"],
        s=config["augmentations"]["s"],
        p_blur=config["augmentations"]["p_blur"],
        flip=config["augmentations"]["flip"],
        rotation=config["augmentations"]["rotation"],
        cut_threshold=config["data"]["cut_threshold"],
        prefetch_factor=config["dataloading"]["prefetch_factor"],
        num_workers=config["dataloading"]["num_workers"],
    )

    ## Run pretraining ##
    pretrain_checkpoint, model = run_contrastive_pretraining(config, datamodule, wandb_logger)

    wandb.save(pretrain_checkpoint.best_model_path)
    # wadnb.save()
    wandb_logger.experiment.finish()


if __name__ == "__main__":
    main()

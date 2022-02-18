import wandb
import pytorch_lightning as pl
import logging

from paths import Path_Handler
from dataloading.datamodules import imagenette_DataModule, imagenette_DataModule_eval
from dataloading.datamodules import mb_DataModule, mb_DataModule_eval
from dataloading.datamodules import stl10_DataModule, stl10_DataModule_eval
from dataloading.datamodules import cifar10_DataModule, cifar10_DataModule_eval

from byol_lightly import byol
from evaluation import linear_net, Feature_Bank
from config import load_config, update_config
from utilities import freeze_model, log_examples

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    config = load_config()
    update_config(config)

    pl.seed_everything(config["seed"])

    paths = Path_Handler()
    path_dict = paths._dict()

    # Save model with best accuracy for test evaluation, model will be saved in wandb and also #
    pretrain_checkpoint = pl.callbacks.ModelCheckpoint(
        monitor="train/loss",
        mode="min",
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        auto_insert_metric_name=False,
        verbose=True,
        dirpath="wandb/",
        filename="{train/loss:.3f}",
        save_weights_only=True,
    )

    # Initialise wandb logger, change this if you want to use a different logger #
    wandb_logger = pl.loggers.WandbLogger(
        project=config["project_name"],
        save_dir=path_dict["files"],
        reinit=True,
        config=config,
    )

    # Load data and record hyperparameters #
    datasets = {
        "imagenette": {
            "pretrain": imagenette_DataModule,
            "linear": imagenette_DataModule_eval,
        },
        "rgz": {
            "pretrain": mb_DataModule,
            "linear": mb_DataModule_eval,
        },
        "stl10": {
            "pretrain": stl10_DataModule,
            "linear": stl10_DataModule_eval,
        },
        "cifar10": {
            "pretrain": cifar10_DataModule,
            "linear": cifar10_DataModule_eval,
        },
    }

    pretrain_data = datasets[config["dataset"]]["pretrain"](config)
    pretrain_data.prepare_data()
    pretrain_data.setup()

    # Record mean and standard deviation used in normalisation for inference #
    config["data"]["mu"] = pretrain_data.mu
    config["data"]["sig"] = pretrain_data.sig
    config["data"]["n_steps"] = len(pretrain_data.train_dataloader())

    log_examples(wandb_logger, pretrain_data.data["train"])

    # List of callbacks
    callbacks = [pretrain_checkpoint, Feature_Bank()]

    trainer_settings = {
        "slurm": {"gpus": 1, "num_nodes": 1},
        "gpu": {"devices": 1, "accelerator": "gpu"},
    }

    pre_trainer = pl.Trainer(
        # gpus=1,
        **trainer_settings[config["compute"]],
        fast_dev_run=config["debug"],
        max_epochs=config["model"]["n_epochs"],
        logger=wandb_logger,
        deterministic=True,
        callbacks=callbacks,
        #    check_val_every_n_epoch=3,
        #    log_every_n_steps=10,
    )

    # Initialise model #
    model = byol(config)
    config["model"]["output_dim"] = config["model"]["features"]

    # Train model #
    pre_trainer.fit(model, pretrain_data)

    # Run test loop #
    # pre_trainer.test(ckpt_path="best")

    # Save model in wandb #
    if not config["debug"]:
        wandb.save(pretrain_checkpoint.best_model_path)

    ##################################################
    ########## LINEAR EVALUATION PROTOCOL ############
    ##################################################

    # Extract and load best encoder from pretraining
    if config["debug"] is True:
        encoder = model.backbone
    else:
        best_model_path = pretrain_checkpoint.best_model_path
        pretrained_model = byol.load_from_checkpoint(best_model_path)
        encoder = model.backbone

    # Freeze encoder weights
    freeze_model(encoder)

    # Switch data-loader to linear evaluation mode
    eval_data = datasets[config["dataset"]]["linear"](encoder, config)
    eval_data.prepare_data()
    eval_data.setup()

    if not config["debug"]:
        config["linear"]["mu"] = eval_data.mu
        config["linear"]["sig"] = eval_data.sig

    linear_checkpoint = pl.callbacks.ModelCheckpoint(
        monitor="linear_eval/val_acc",
        mode="max",
        every_n_epochs=1,
        verbose=True,
    )

    linear_trainer = pl.Trainer(
        **trainer_settings[config["compute"]],
        fast_dev_run=config["debug"],
        max_epochs=config["linear"]["n_epochs"],
        logger=wandb_logger,
        deterministic=True,
    )

    linear_model = linear_net(config)
    linear_trainer.fit(linear_model, eval_data)
    # linear_trainer.test(linear_model, dataloaders=eval_data, ckpt_path="best")

    wandb_logger.experiment.finish()

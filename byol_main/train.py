import wandb
import pytorch_lightning as pl
import logging

from pytorch_lightning.callbacks import LearningRateMonitor

from paths import Path_Handler
from byol_main.dataloading.datamodules import Imagenette_DataModule, Imagenette_DataModule_Eval
from byol_main.dataloading.datamodules import GalaxyMNIST_DataModule, GalaxyMNIST_DataModule_Eval
from byol_main.dataloading.datamodules import GZ2_DataModule, GZ2_DataModule_Eval
from byol_main.dataloading.datamodules import RGZ_DataModule, RGZ_DataModule_Eval
from byol_main.dataloading.datamodules import CIFAR10_DataModule, CIFAR10_DataModule_Eval
from byol_main.byol import BYOL, Update_M
from byol_main.nnclr import NNCLR
from byol_main.evaluation import linear_net, Feature_Bank
from byol_main.config import load_config, update_config
from byol_main.utilities import freeze_model, log_examples

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    config = load_config()
    # update_config(config)

    pl.seed_everything(config["seed"])

    paths = Path_Handler()
    path_dict = paths._dict()

    # Save model for test evaluation#
    checkpoint_mode = {
        "min_loss": {"mode": "min", "monitor": "train/loss"},
        "last": {"monitor": None},
    }
    pretrain_checkpoint = pl.callbacks.ModelCheckpoint(
        **checkpoint_mode[config["checkpoint_mode"]],
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
            "pretrain": Imagenette_DataModule,
            "linear": Imagenette_DataModule_Eval,
        },
        "gzmnist": {
            "pretrain": GalaxyMNIST_DataModule,
            "linear": GalaxyMNIST_DataModule_Eval,
        },
        "gz2": {
            "pretrain": GZ2_DataModule,
            "linear": GZ2_DataModule_Eval,
        },
        "rgz": {
            "pretrain": RGZ_DataModule,
            "linear": RGZ_DataModule_Eval,
        },
        # "stl10": {
        #     "pretrain": STL10_DataModule,
        #     "linear": STL10_DataModule_Eval,
        # },
        "cifar10": {
            "pretrain": CIFAR10_DataModule,
            "linear": CIFAR10_DataModule_Eval,
        },
    }

    pretrain_data = datasets[config["dataset"]]["pretrain"](config)
    # pretrain_data.prepare_data()
    # pretrain_data.setup()

    # Record mean and standard deviation used in normalisation for inference #
    # config["data"]["mu"] = pretrain_data.mu
    # config["data"]["sig"] = pretrain_data.sig
    # config["data"]["n_steps"] = len(pretrain_data.train_dataloader())

    # List of callbacks
    callbacks = [
        pretrain_checkpoint,
        LearningRateMonitor(),
    ]

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
        precision=config["precision"]
        #    check_val_every_n_epoch=3,
        #    log_every_n_steps=10,
    )

    # Initialise model #
    models = {"byol": BYOL, "nnclr": NNCLR}
    model = models[config["type"]](config)

    config["model"]["output_dim"] = config["model"]["features"]

    # Train model #
    pre_trainer.fit(model, pretrain_data)

    log_examples(wandb_logger, pretrain_data.data["train"])

    print(model.config["data"]["mu"])

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
        pretrained_model = BYOL.load_from_checkpoint(best_model_path)
        encoder = model.backbone

    encoder = model.backbone
    # Freeze encoder weights
    freeze_model(encoder)
    encoder.eval()

    logging.info('Training complete - switching to eval mode')

    # Switch data-loader to linear evaluation mode
    eval_data = datasets[config["dataset"]]["linear"](encoder, config)
    # eval_data.prepare_data()
    # eval_data.setup()

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
        # always full precision, never distributed. May need a batch size adjustment.
    )

    linear_model = linear_net(config)
    linear_trainer.fit(linear_model, eval_data)
    # linear_trainer.test(linear_model, dataloaders=eval_data, ckpt_path="best")

    wandb_logger.experiment.finish()

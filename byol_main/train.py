import wandb
import pytorch_lightning as pl
import logging

from pytorch_lightning.callbacks import LearningRateMonitor

from paths import Path_Handler
from byol_main.dataloading.datamodules import Imagenette_DataModule, Imagenette_DataModule_Eval
from byol_main.dataloading.datamodules import GalaxyMNIST_DataModule, GalaxyMNIST_DataModule_Eval
from byol_main.dataloading.datamodules import GZ2_DataModule, GZ2_DataModule_Eval
from byol_main.dataloading import RGZ_DataModule, RGZ_DataModule_Eval, RGZ_DataModule_Supervised
from byol_main.dataloading.datamodules import CIFAR10_DataModule, CIFAR10_DataModule_Eval
from byol_main.byol import BYOL, Update_M
from byol_main.nnclr import NNCLR
from byol_main.evaluation import linear_net, Feature_Bank, Epoch_Averaged_Test, Count_Similarity
from byol_main.config import load_config, update_config
from byol_main.utilities import freeze_model, log_examples

from byol_rr import BYOL_RR, count_masks, Count_Masks
from byol_pretext import BYOL_Pretext
from supervised import Supervised

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    config = load_config()
    update_config(config)

    for i in range(config["n_iterations"]):
        config["seed"] += 1

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
            # dirpath="wandb/",
            filename="{train/loss:.3f}",
            save_weights_only=True,
            # save_top_k=config["topk"],
        )

        # Initialise wandb logger, change this if you want to use a different logger #
        wandb_logger = pl.loggers.WandbLogger(
            project=config["project_name"],
            save_dir=path_dict["files"],
            # reinit=True,
            config=config,
        )

        # Load data and record hyperparameters #
        if config["type"] == "supervised":
            datasets = {
                # "imagenette": {
                #     "pretrain": Imagenette_DataModule,
                #     "linear": Imagenette_DataModule_Eval,
                # },
                # "gzmnist": {
                #     "pretrain": GalaxyMNIST_DataModule,
                #     "linear": GalaxyMNIST_DataModule_Eval,
                # },
                # "gz2": {
                #     "pretrain": GZ2_DataModule,
                #     "linear": GZ2_DataModule_Eval,
                # },
                "rgz": {
                    "pretrain": RGZ_DataModule_Supervised,
                    "linear": RGZ_DataModule_Eval,
                }
                # "stl10": {
                #     "pretrain": STL10_DataModule,
                #     "linear": STL10_DataModule_Eval,
                # },
                # "cifar10": {
                #     "pretrain": CIFAR10_DataModule,
                #     "linear": CIFAR10_DataModule_Eval,
                # },
            }
        else:
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

        # Record mean and standard deviation used in normalisation for inference #
        # config["data"]["mu"] = pretrain_data.mu
        # config["data"]["sig"] = pretrain_data.sig
        # config["data"]["n_steps"] = len(pretrain_data.train_dataloader())

        # List of callbacks
        callbacks = [
            pretrain_checkpoint,
            LearningRateMonitor(),
            Epoch_Averaged_Test(),
            # Count_Similarity(),
        ]

        if config["type"] == "byol_rr":
            callbacks.append(Count_Masks())

        if config["early_stopping"]:
            callbacks.append(pl.callbacks.EarlyStopping(monitor="train/loss", mode="min", patience=4))

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
        models = {
            "byol_pretext": BYOL_Pretext,
            "byol_rr": BYOL_RR,
            "byol": BYOL,
            "supervised": Supervised,
            "nnclr": NNCLR,
        }
        _model = models[config["type"]]

        model = _model(config)

        config["model"]["output_dim"] = config["model"]["features"]

        # Train model #
        pre_trainer.fit(model, pretrain_data)
        pre_trainer.test(model, dataloaders=pretrain_data)

        # log_examples(wandb_logger, pretrain_data.data["train"])

        # Run test loop #
        # knn_acc = 0
        # y_masked = []
        # for ckpt_path in pre_trainer.checkpoint_callback.best_k_models:
        #     knn_acc += pre_trainer.test(ckpt_path=ckpt_path)
        #     model = _model.load_from_checkpoint(ckpt_path)
        #     # y_masked.append(count_masks(model, pretrain_data.train_dataloader()))

        # y_masked = torch.cat(y_masked)

        # Save model in wandb #
        if not config["debug"]:
            wandb.save(pretrain_checkpoint.best_model_path)

        ##################################################
        ############### EVAL #############################
        ##################################################

        ##################################################
        ########## LINEAR EVALUATION PROTOCOL ############
        ##################################################

        if config["linear_eval"]:

            # Extract and load best encoder from pretraining
            if config["debug"] is False:
                best_model_path = pretrain_checkpoint.best_model_path
                model = _model.load_from_checkpoint(best_model_path)

            encoder = model.backbone

            # Freeze encoder weights
            freeze_model(encoder)
            encoder.eval()

            logging.info("Training complete - switching to eval mode")

            # Load eval data
            eval_data = datasets[config["dataset"]]["linear"](encoder, config)

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

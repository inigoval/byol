import wandb
import pytorch_lightning as pl
import logging
import pytorch_lightning as pl
import torch
import torchmetrics as tm
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from pathlib import Path
from einops import rearrange
from typing import Any, Dict, List, Tuple, Type, Union
from torch import Tensor
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from byol.paths import Path_Handler
from byol.config import load_config_regression
from byol.models import BYOL
from byol.datamodules import RGZ_DataModule_Finetune_Regression
from byol.resnet import ResNet, BasicBlock
from byol.paths import Path_Handler, create_path


class LinearRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.batchnorm = nn.BatchNorm1d(input_dim)
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.batchnorm(x)
        x = self.linear(x)
        x = F.relu(x)
        return x


class FineTuneRegression(pl.LightningModule):
    """
    Parent class for self-supervised LightningModules to perform linear evaluation with multiple
    data-sets.
    """

    def __init__(
        self,
        encoder: nn.Module,
        dim: int,
        n_out,
        n_epochs=100,
        n_layers=0,
        batch_size=1024,
        lr_decay=0.75,
        seed=69,
        config=None,
    ):
        super().__init__()

        self.encoder = encoder
        self.dim = dim
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.n_out = n_out
        self.lr_decay = lr_decay
        self.n_epochs = n_epochs
        self.seed = seed
        self.layers = []

        self.config = config

        self.head = LinearRegression(input_dim=dim, output_dim=n_out)

        # Set finetuning layers for easy access
        if self.n_layers:
            layers = self.encoder.finetuning_layers
            assert self.n_layers <= len(
                layers
            ), f"Network only has {len(layers)} layers, {self.n_layers} specified for finetuning"

            self.layers = layers[::-1][:n_layers]

        # self.save_hyperparameters(ignore=["encoder", "head", "layers"])
        self.save_hyperparameters()

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = rearrange(x, "b c h w -> b (c h w)")
        x = self.head(x)
        return x

    def on_fit_start(self):
        # Log size of data-sets #

        self.val_loss = tm.MeanAbsoluteError().to(self.device)

        self.test_loss = nn.ModuleList(
            [tm.MeanAbsoluteError().to(self.device)] * len(self.trainer.datamodule.data["test"])
        )

        logging_params = {f"n_{key}": len(value) for key, value in self.trainer.datamodule.data.items()}
        self.logger.log_hyperparams(logging_params)

        # Make sure network that isn't being finetuned is frozen
        # probably unnecessary but best to be sure
        set_grads(self.encoder, False)
        if self.n_layers:
            for layer in self.layers:
                set_grads(layer, True)

    def training_step(self, batch, batch_idx):
        # Load data and targets
        x, y = batch
        y = y["size"]
        assert not torch.isnan(x).any()
        preds = self.forward(x).squeeze()
        # print("preds: ", preds)
        # print("targets: ", y)
        loss = F.mse_loss(preds, y.float(), reduction="sum")
        # loss = F.l1_loss(preds, y)
        # self.log(
        #     "finetuning/train_mean_absolute_error", F.l1_loss(preds, y), on_step=False, on_epoch=True
        # )

        self.log("finetuning/train_loss", loss, on_step=False, on_epoch=True)
        # print(loss.item())
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y = y["size"]
        preds = self.forward(x).squeeze()
        self.val_loss(preds, y)
        self.log("finetuning/val_loss", self.val_loss, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y = y["size"]
        name = list(self.trainer.datamodule.data["test"].keys())[dataloader_idx]

        preds = self.forward(x).squeeze()
        self.test_loss[dataloader_idx](preds, y)
        self.log(
            f"finetuning/test/{name}_loss",
            self.test_loss[dataloader_idx],
            on_step=False,
            on_epoch=True,
            add_dataloader_idx=False,
        )

    def configure_optimizers(self):
        if not self.n_layers:
            # Scale base lr=0.1
            lr = 0.1 * self.batch_size / 256
            params = self.head.parameters()
            return torch.optim.SGD(params, momentum=0.9, lr=lr)
        else:
            lr = 0.001 * self.batch_size / 256
            params = [{"params": self.head.parameters(), "lr": lr}]
            # layers.reverse()

            # Append parameters of layers for finetuning along with decayed learning rate
            for i, layer in enumerate(self.layers):
                params.append({"params": layer.parameters(), "lr": lr * (self.lr_decay**i)})

            # Initialize AdamW optimizer with cosine decay learning rate
            opt = torch.optim.AdamW(params, weight_decay=0.05, betas=(0.9, 0.999))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.n_epochs)
            return [opt], [scheduler]


def run_finetuning(config, encoder, datamodule, logger):
    paths = Path_Handler()._dict()

    checkpoint = ModelCheckpoint(
        monitor=None,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        auto_insert_metric_name=False,
        verbose=True,
        dirpath=paths["files"] / "finetuning" / config["run_id"],
        filename="model",  # filename may not work here TODO
        save_weights_only=True,
        # save_top_k=3,
    )

    callbacks = []

    early_stop_callback = EarlyStopping(
        monitor="finetuning/train_loss", min_delta=0.00, patience=3, verbose=True, mode="min"
    )

    if config["finetune"]["early_stopping"]:
        callbacks.append(early_stop_callback)

    ## Initialise pytorch lightning trainer ##
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        max_epochs=config["finetune"]["n_epochs"],
        **config["trainer"],
    )

    # Initialize linear head
    # if config["finetune"]["head"] == "linear":
    #     # head = LogisticRegression(input_dim=encoder.dim, output_dim=config["finetune"]["n_out"])
    #     head = "linear"

    # elif config["finetune"]["head"] == "mlp":
    #     head = MLPHead(
    #         input_dim=encoder.dim,
    #         depth=config["finetune"]["depth"],
    #         width=config["finetune"]["width"],
    #         output_dim=config["finetune"]["n_out"],
    #     )
    # else:
    #     raise ValueError("Head must be either linear or mlp")

    model = FineTuneRegression(
        encoder,
        dim=encoder.dim,
        n_out=config["finetune"]["n_out"],
        n_epochs=config["finetune"]["n_epochs"],
        n_layers=config["finetune"]["n_layers"],
        batch_size=config["finetune"]["batch_size"],
        lr_decay=config["finetune"]["lr_decay"],
        # lr_decay=0,
        seed=config["seed"],
        config=config,
    )
    trainer.fit(model, datamodule)

    trainer.test(model, dataloaders=datamodule)

    return checkpoint, model


def set_grads(module, value: bool):
    for params in module.parameters():
        params.requires_grad = value


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    # Load paths
    paths = Path_Handler()._dict()

    # Load up finetuning config
    config_finetune = load_config_regression()

    ## Run finetuning ##
    for seed in range(config_finetune["finetune"]["iterations"]):
        # for seed in range(1, 10):

        if config_finetune["finetune"]["run_id"].lower() != "none":
            experiment_dir = paths["files"] / config_finetune["finetune"]["run_id"]
            model = BYOL.load_from_checkpoint(experiment_dir / "last.ckpt")
        else:
            model = BYOL.load_from_checkpoint("byol.ckpt")

        encoder = model.encoder
        # encoder = ResNet(block=BasicBlock, layers=[2, 2, 2, 2], n_c=1, downscale=True, features=512)
        ## Load up config from model to save correct hparams for easy logging ##
        config = model.config
        config.update(config_finetune)
        config["finetune"]["dim"] = model.encoder.dim
        project_name = "BYOL_finetune_regression"

        config["finetune"]["seed"] = seed
        pl.seed_everything(seed)

        # Initiate wandb logging
        wandb.init(project=project_name, config=config)

        config["run_id"] = str(wandb.run.id)

        logger = pl.loggers.WandbLogger(
            project=project_name,
            save_dir=paths["files"] / "finetuning" / str(wandb.run.id),
            reinit=True,
            config=config,
        )

        finetune_datamodule = RGZ_DataModule_Finetune_Regression(
            paths["rgz"],
            batch_size=config["finetune"]["batch_size"],
            center_crop=config["augmentations"]["center_crop"],
            num_workers=config["dataloading"]["num_workers"],
            prefetch_factor=config["dataloading"]["prefetch_factor"],
            pin_memory=config["dataloading"]["pin_memory"],
        )

        create_path(paths["files"] / "finetuning")
        create_path(paths["files"] / "finetuning" / config["run_id"])

        checkpoint, model = run_finetuning(config, encoder, finetune_datamodule, logger)
        logger.experiment.finish()
        wandb.finish()


if __name__ == "__main__":
    main()

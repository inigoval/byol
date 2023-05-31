import wandb
import pytorch_lightning as pl
import logging

from pathlib import Path

from paths import Path_Handler
from finetune.dataloading import finetune_datasets
from config import load_config, update_config, load_config_finetune
from models import BYOL


import pytorch_lightning as pl
import torch
import torchmetrics as tm
import torch.nn.functional as F
import torch.nn as nn
from typing import Union

from einops import rearrange
from typing import Any, Dict, List, Tuple, Type

from torch import Tensor

# from models import LogisticRegression
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.batchnorm = nn.BatchNorm1d(input_dim)
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.batchnorm(x)
        x = self.linear(x)
        return x


class FineTune(pl.LightningModule):
    """
    Parent class for self-supervised LightningModules to perform linear evaluation with multiple
    data-sets.
    """

    def __init__(
        self,
        encoder: nn.Module,
        head,
        dim: int,
        n_classes,
        n_epochs=100,
        n_layers=0,
        batch_size=1024,
        lr_decay=0.75,
        seed=69,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["encoder", "head"])

        self.n_layers = n_layers
        self.batch_size = batch_size
        self.encoder = encoder
        self.lr_decay = lr_decay
        self.n_epochs = n_epochs
        self.seed = seed
        self.head = head
        self.n_classes = n_classes
        self.layers = []

        # Set head
        if head == "linear":
            self.head = LogisticRegression(input_dim=dim, output_dim=n_classes)
            self.head_type = "linear"
        elif isinstance(head, nn.Module):
            self.head = head
            self.head_type = "custom"
        else:
            raise ValueError("Head must be either 'linear' or a PyTorch Module")

        # Set finetuning layers for easy access
        if self.n_layers:
            layers = self.encoder.finetuning_layers
            assert self.n_layers <= len(
                layers
            ), f"Network only has {len(layers)} layers, {self.n_layers} specified for finetuning"

            self.layers = layers[::-1][:n_layers]

        self.train_acc = tm.Accuracy(
            task="multiclass", average="micro", threshold=0, num_classes=self.n_classes
        ).to(self.device)

        self.val_acc = tm.Accuracy(
            task="multiclass", average="micro", threshold=0, num_classes=self.n_classes
        ).to(self.device)

        self.test_acc = tm.Accuracy(
            task="multiclass", average="micro", threshold=0, num_classes=self.n_classes
        ).to(self.device)

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = rearrange(x, "b c h w -> b (c h w)")
        x = self.head(x)
        return x

    def on_fit_start(self):
        # Log size of data-sets #

        self.train_acc = tm.Accuracy(
            task="multiclass", average="micro", threshold=0, num_classes=self.n_classes
        ).to(self.device)
        self.val_acc = tm.Accuracy(
            task="multiclass", average="micro", threshold=0, num_classes=self.n_classes
        ).to(self.device)

        self.test_acc = nn.ModuleList(
            [
                tm.Accuracy(
                    task="multiclass", average="micro", threshold=0, num_classes=self.n_classes
                ).to(self.device)
            ]
            * len(self.trainer.datamodule.data["test"])
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
        logits = self.forward(x)
        y_pred = logits.softmax(dim=-1)
        loss = F.cross_entropy(y_pred, y, label_smoothing=0.1 if self.n_layers else 0)
        self.log("finetuning/train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        preds = self.forward(x)
        self.val_acc(preds, y)
        self.log("finetuning/val_acc", self.val_acc, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        name = list(self.trainer.datamodule.data["test"].keys())[dataloader_idx]

        preds = self.forward(x)
        self.test_acc[dataloader_idx](preds, y)
        self.log(
            f"finetuning/test/{name}_acc", self.test_acc[dataloader_idx], on_step=False, on_epoch=True
        )

    def configure_optimizers(self):
        if not self.n_layers and self.head_type == "linear":
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


class MLPHead(nn.Module):
    """
    Fully connected head with a single hidden layer. Batchnorm applied as first layer so that
    feature space of encoder doesn't need to be normalized.
    """

    def __init__(self, input_dim, depth, width, output_dim):
        super(MLPHead, self).__init__()

        self.input_layer = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, width),
            nn.GELU(),
        )

        self.hidden_layers = nn.ModuleList()
        for i in range(depth):
            self.hidden_layers.append(
                nn.Sequential(
                    nn.Linear(width, width),
                    nn.GELU(),
                )
            )

        self.output_layer = nn.Sequential(
            nn.Linear(width, output_dim),
        )

    def forward(self, x):
        x = self.input_layer(x)

        for layer in self.hidden_layers:
            x = layer(x)

        x = self.output_layer(x)

        return x


def run_finetuning(config, encoder, datamodule, logger):
    checkpoint = ModelCheckpoint(
        monitor=None,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        auto_insert_metric_name=False,
        verbose=True,
        # dirpath=config["files"] / config["run_id"] / "finetuning",
        # e.g. byol/files/(run_id)/checkpoints/12-344-18.134.ckpt.
        filename="{epoch}",  # filename may not work here TODO
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
    if config["finetune"]["head"] == "linear":
        # head = LogisticRegression(input_dim=encoder.dim, output_dim=config["finetune"]["n_classes"])
        head = "linear"

    elif config["finetune"]["head"] == "mlp":
        head = MLPHead(
            input_dim=encoder.dim,
            depth=config["finetune"]["depth"],
            width=config["finetune"]["width"],
            output_dim=config["finetune"]["n_classes"],
        )
    else:
        raise ValueError("Head must be either linear or mlp")

    model = FineTune(
        encoder,
        head,
        dim=encoder.dim,
        n_classes=config["finetune"]["n_classes"],
        n_epochs=config["finetune"]["n_epochs"],
        n_layers=config["finetune"]["n_layers"],
        batch_size=config["finetune"]["batch_size"],
        lr_decay=config["finetune"]["lr_decay"],
        seed=config["seed"],
        head_type=config["finetune"]["head"],
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
        # project_name = "BYOL_laptoptest_finetune"
        # project_name = "BYOL_debugging"
        project_name = "BYOL_finetune_mlp"

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

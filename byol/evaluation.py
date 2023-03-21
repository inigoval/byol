import pytorch_lightning as pl
import torch
import torchmetrics as tm
import torch.nn.functional as F
import sklearn
import logging
from einops import rearrange

from sklearn.preprocessing import StandardScaler
from typing import Any, Dict, List, Tuple, Type
from torch import Tensor

from byol.utilities import log_examples, embed_dataset
from byol.networks.models import LogisticRegression


class Lightning_Eval(pl.LightningModule):
    """
    Parent class for self-supervised LightningModules to perform linear evaluation with multiple
    data-sets.
    """

    def __init__(self, config):
        super().__init__()

    def on_fit_start(self):
        self.config["data"]["mu"] = self.trainer.datamodule.mu
        self.config["data"]["sig"] = self.trainer.datamodule.sig

        # Log size of data-sets #
        logging_params = {"n_train": len(self.trainer.datamodule.data["train"])}

        for name, data in (
            self.trainer.datamodule.data["val"]
            + self.trainer.datamodule.data["test"]
            + [(d["name"], d["data"]) for d in self.trainer.datamodule.data["eval_train"]]
        ):
            logging_params[f"n_{name}"] = len(data)

        self.logger.log_hyperparams(logging_params)

        if not self.config["trainer"]["fast_dev_run"]:
            log_examples(self.logger, self.trainer.datamodule.data["train"])

    def on_validation_start(self):
        ## List of evaluation classes and dataloader_idx to use ##
        self.train_list = []
        self.val_list = [name for (name, _) in self.trainer.datamodule.data["val"]]
        self.test_list = [name for (name, _) in self.trainer.datamodule.data["test"]]

        ## Prepare for linear evaluation ##
        # Cycle through validation data-sets
        for d in self.trainer.datamodule.data["eval_train"]:
            # Initialise linear eval data-set and run setup with training data
            lin_eval = Linear_Eval(self, d, self.val_list, self.test_list)
            lin_eval.setup(self, d["data"])

            # Add to list of evaluations
            self.train_list.append(lin_eval)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        val_name = self.val_list[dataloader_idx]

        # Run validation step for filtered data-sets
        for val in self.train_list:
            val.step(self, x, y, val_name, stage="val")

    def on_validation_epoch_end(self):
        # Complete validation for all data-sets
        for val in self.train_list:
            val.end(self, stage="val")

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        test_name = self.test_list[dataloader_idx]

        # Run validation step for filtered data-sets
        for val in self.train_list:
            val.step(self, x, y, test_name, stage="test")

    def on_test_epoch_end(self):
        # Complete validation for all data-sets
        for val in self.train_list:
            val.end(self, stage="test")


class Data_Eval:
    """
    Parent class for evaluation classes.
    """

    def __init__(self, train_data_dict, val_list, test_list):
        self.train_data_name = train_data_dict["name"]
        self.n_classes = train_data_dict["n_classes"]
        self.val_list = val_list
        self.test_list = test_list

    def setup(self):
        return

    def step(self, pl_module, x, y):
        return

    def end(self, pl_module, stage):
        return


class Linear_Eval(Data_Eval):
    """
    Callback to perform linear evaluation at the end of each epoch.

    Attributes:
        data: Data dictionary containing 'train', 'val', 'test' and 'name' keys.
        clf: Linear classification model.

    """

    def __init__(self, pl_module, train_data_dict, val_list, test_list):
        """
        Args:
            data: Data dictionary containing 'train', 'val', 'test' and 'name' keys.
        """
        super().__init__(train_data_dict, val_list, test_list)

        pl_module.lin_acc = {
            "val": {
                val: tm.Accuracy(
                    average="micro", threshold=0, task="multiclass", num_classes=self.n_classes
                )
                for val in self.val_list
            },
            "test": {
                test: tm.Accuracy(
                    average="micro", threshold=0, task="multiclass", num_classes=self.n_classes
                )
                for test in self.test_list
            },
        }

    def setup(self, pl_module, data):
        with torch.no_grad():
            model = sklearn.linear_model.LogisticRegression(penalty="none")
            X, y = embed_dataset(pl_module.encoder, data)
            X, y = X.detach().cpu().numpy(), y.detach().cpu().numpy()
            self.scaler = StandardScaler()
            self.scaler.fit(X)
            X = self.scaler.transform(X)
            model.fit(X, y)
            self.model = model

        # Could shorten this with recursion but might be less clear
        for acc in pl_module.lin_acc["val"].values():
            acc.reset()
        for acc in pl_module.lin_acc["test"].values():
            acc.reset()

    def step(self, pl_module, X, y, val_name, stage):
        X = pl_module(X).squeeze()
        X, y = X.detach().cpu().numpy(), y.detach().cpu()
        X = self.scaler.transform(X)
        preds = self.model.predict(X)

        pl_module.lin_acc[stage][val_name].update(torch.tensor(preds), y)

    def end(self, pl_module, stage):
        for val_name, acc in pl_module.lin_acc[stage].items():
            # Grab evaluation data-set name directly from dataloader
            pl_module.log(f"{stage}/{self.train_data_name}/linear_acc/{val_name}", acc.compute())


class FineTune(pl.LightningModule):
    """
    Parent class for self-supervised LightningModules to perform linear evaluation with multiple
    data-sets.
    """

    def __init__(
        self,
        encoder,
        dim,
        n_classes,
        n_epochs=100,
        n_layers=0,
        batch_size=1024,
        lr_decay=0.75,
        seed=0,
        **kwargs,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.freeze = True if n_layers == 0 else False
        self.batch_size = batch_size
        self.encoder = encoder
        self.lr_decay = lr_decay
        self.head = LogisticRegression(input_dim=dim, output_dim=n_classes)
        self.n_epochs = n_epochs
        self.seed = seed

        self.val_acc = tm.Accuracy(average="micro", threshold=0)
        self.test_acc = tm.Accuracy(average="micro", threshold=0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = rearrange(x, "b c h w -> b (c h w)")
        x = self.head(x)
        return x

    def on_fit_start(self):
        # Log size of data-sets #
        logging_params = {key: len(value) for key, value in self.trainer.datamodule.data.items()}
        self.logger.log_hyperparams(logging_params)

    def training_step(self, batch, batch_idx):
        # Load data and targets
        x, y = batch
        logits = self.forward(x)
        y_pred = logits.softmax(dim=-1)
        loss = F.cross_entropy(y_pred, y, label_smoothing=0.1 if self.freeze else 0)
        self.log("finetune/train_loss_{self.seed}", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        preds = self.forward(x)
        self.val_acc(preds, y)
        self.log("finetuning/val_acc_{self.seed}", self.val_acc, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        preds = self.forward(x)
        self.test_acc(preds, y)
        self.log("finetuning/test_acc_{self.seed}", self.test_acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        if self.freeze:
            # Scale base lr=0.1
            lr = 0.1 * self.batch_size / 256
            params = self.head.parameters()
            return torch.optim.SGD(params, momentum=0.9, lr=lr)
        else:
            lr = 0.001 * self.batch_size / 256
            params = [{"params": self.head.parameters(), "lr": lr}]
            layers = self.encoder.finetuning_layers[::-1]
            # layers.reverse()
            assert self.n_layers <= len(
                layers
            ), f"Network only has {len(layers)} layers, {self.n_layers} specified for finetuning"
            for i, layer in enumerate(layers[: self.n_layers]):
                params.append({"params": layer.parameters(), "lr": lr * (self.lr_decay**i)})

            opt = torch.optim.AdamW(params, weight_decay=0.05, betas=(0.9, 0.999))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.n_epochs)
            return [opt], [scheduler]


def finetune(config, encoder, datamodule, logger):
    checkpoint = pl.callbacks.ModelCheckpoint(
        monitor=None,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        auto_insert_metric_name=False,
        verbose=True,
        dirpath=config["files"] / config["run_id"] / "finetuning",
        # e.g. byol/files/(run_id)/checkpoints/12-344-18.134.ckpt.
        filename="{epoch}",  # filename may not work here TODO
        save_weights_only=True,
        # save_top_k=3,
    )

    trainer_settings = {
        "slurm": {"gpus": 1, "num_nodes": 1},
        "gpu": {"devices": 1, "accelerator": "gpu"},
    }
    config["trainer_settings"] = trainer_settings[config["compute"]]

    ## Initialise pytorch lightning trainer ##
    trainer = pl.Trainer(
        **trainer_settings[config["compute"]],
        fast_dev_run=config["trainer"]["fast_dev_run"],
        max_epochs=config["finetune"]["n_epochs"],
        logger=logger,
        deterministic=True,
        callbacks=[checkpoint],
        precision=config["precision"],
    )

    model = FineTune(encoder, **config["finetune"])

    trainer.fit(model, datamodule)

    trainer.test(model)

    return checkpoint, model

import pytorch_lightning as pl
import torch
import torchmetrics as tm
import torch.nn.functional as F
from sklearn import linear_model
import logging
from einops import rearrange

from sklearn.preprocessing import StandardScaler
from typing import Any, Dict, List, Tuple, Type
from torch import Tensor

from byol.utilities import log_examples, embed_dataset


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.batchnorm = nn.BatchNorm1d(input_dim)
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.batchnorm(x)
        x = self.linear(x)
        return x


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
            model = linear_model.LogisticRegression(penalty="none")
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

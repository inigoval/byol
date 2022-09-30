import pytorch_lightning as pl
import torch
import torchmetrics as tm
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import sklearn

from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from torch.nn.functional import softmax
from typing import Any, Dict, List, Tuple, Type
from torch import Tensor
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torchvision.transforms import Normalize

from torchvision.transforms import Normalize
from utilities import (
    log_examples,
    embed_dataset,
    freeze_model,
    unfreeze_model,
    compute_encoded_mu_sig,
    check_unique_list,
)


class Lightning_Eval(pl.LightningModule):
    """
    Parent class for self-supervised LightningModules to perform linear evaluation with multiple
    data-sets.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

    def on_fit_start(self):
        self.config["data"]["mu"] = self.trainer.datamodule.mu
        self.config["data"]["sig"] = self.trainer.datamodule.sig

        # Log size of data-sets #
        logging_params = {"n_train": len(self.trainer.datamodule.data["train"])}

        for name, data in self.trainer.datamodule.data["val"]:
            logging_params[f"n_{name}"] = len(data)

        for name, data in self.trainer.datamodule.data["test"]:
            logging_params[f"n_{name}"] = len(data)

        for name, data, _ in self.trainer.datamodule.data["eval_train"]:
            logging_params[f"n_{name}"] = len(data)

        # for name, data in self.trainer.datamodule.data["test"]:
        #     logging_params[f"{name}_n"] = len(data)

        self.logger.log_hyperparams(logging_params)
        # self.log("train/mu", self.trainer.datamodule.mu)
        # self.log("train/sig", self.trainer.datamodule.sig)

        # logger = self.logger.experiment

        if not self.config["debug"]:
            log_examples(self.logger, self.trainer.datamodule.data["train"])

    def on_validation_start(self):

        ## List of evaluation classes and dataloader_idx to use ##
        self.eval_list = []

        ## Prepare for linear evaluation ##
        # Cycle through validation data-sets
        for idx, (name, data, dataloader_idx) in enumerate(self.trainer.datamodule.data["eval_train"]):
            if self.config["evaluation"]["linear_eval"]:
                # Initialise linear eval data-set and run setup with training data
                lin_eval = Linear_Eval(name, dataloader_idx)
                lin_eval.setup(self, data)

                # Add to list of evaluations
                self.eval_list.append(lin_eval)

            if self.config["evaluation"]["ridge_eval"]:
                # Initialise linear eval data-set and run setup with training data
                ridge_eval = Ridge_Eval(name, dataloader_idx)
                ridge_eval.setup(self, data)

                # Add to list of evaluations
                self.eval_list.append(ridge_eval)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        # Only evaluate on given data if evaluation model has given dataloader_idx
        eval_list_filtered = [
            val for val in self.eval_list if dataloader_idx in val.dataloader_idx["val"]
        ]

        # Run validation step for filtered data-sets
        for val in eval_list_filtered:
            val.step(self, x, y, dataloader_idx, stage="val")

    def on_validation_epoch_end(self):
        # Complete validation for all data-sets
        for val in self.eval_list:
            val.end(self, stage="val")

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch

        eval_list_filtered = [
            val for val in self.eval_list if dataloader_idx in val.dataloader_idx["test"]
        ]

        # Run validation step for filtered data-sets
        for val in eval_list_filtered:
            val.step(self, x, y, dataloader_idx, stage="test")

    def on_test_epoch_end(self):
        # Complete validation for all data-sets
        for val in self.eval_list:
            val.end(self, stage="test")


class Data_Eval:
    """
    Parent class for evaluation classes.
    """

    def __init__(self, train_data_name, dataloader_idx):

        self.train_data_name = train_data_name
        self.dataloader_idx = dataloader_idx

    def setup(self):
        return

    def step(self, pl_module, x, y):
        return

    def end(self, pl_module, stage):
        return


class Ridge_Eval(Data_Eval):
    """
    Callback to perform linear evaluation at the end of each epoch.

    Attributes:
        data: Data dictionary containing 'train', 'val', 'test' and 'name' keys.
        clf: Linear classification model.

    """

    def __init__(self, train_data_name, dataloader_idx):
        """
        Args:
            data: Data dictionary containing 'train', 'val', 'test' and 'name' keys.
        """
        super().__init__(train_data_name, dataloader_idx)
        # for key, idx in dataloader_idx.items():
        #     self.acc[key] = [tm.Accuracy(average='micro', threshold=0)] * len(idx)

        assert check_unique_list(dataloader_idx["val"])
        assert check_unique_list(dataloader_idx["test"])

        self.acc = {
            "val": {idx: tm.Accuracy(average="micro", threshold=0) for idx in dataloader_idx["val"]},
            "test": {idx: tm.Accuracy(average="micro", threshold=0) for idx in dataloader_idx["test"]},
        }

        # self.acc = {
        #     "val": [(tm.Accuracy(average="micro", threshold=0), idx) for idx in dataloader_idx["val"]],
        #     "test": [(tm.Accuracy(average="micro", threshold=0), idx) for idx in dataloader_idx["test"]],
        # }

    def setup(self, pl_module, data):
        with torch.no_grad():
            model = RidgeClassifier()
            X, y = embed_dataset(pl_module.backbone, data)
            X, y = X.detach().cpu().numpy(), y.detach().cpu().numpy()
            self.scaler = StandardScaler()
            self.scaler.fit(X)
            X = self.scaler.transform(X)
            model.fit(X, y)
            self.model = model

        # Could shorten this with recursion but might be less clear
        for acc in self.acc["val"].values():
            acc.reset()
        for acc in self.acc["test"].values():
            acc.reset()

    def step(self, pl_module, X, y, dataloader_idx, stage):
        X = pl_module(X).squeeze()
        X, y = X.detach().cpu().numpy(), y.detach().cpu()
        X = self.scaler.transform(X)
        preds = self.model.predict(X)

        self.acc[stage][dataloader_idx].update(torch.tensor(preds), y)

    def end(self, pl_module, stage):

        for dataloader_idx, acc in self.acc[stage].items():
            # Grab evaluation data-set name directly from dataloader
            eval_name, _ = pl_module.trainer.datamodule.data[stage][dataloader_idx]
            pl_module.log(f"{stage}/{self.train_data_name}/ridge_acc/{eval_name}", acc.compute())


class Linear_Eval(Data_Eval):
    """
    Callback to perform linear evaluation at the end of each epoch.

    Attributes:
        data: Data dictionary containing 'train', 'val', 'test' and 'name' keys.
        clf: Linear classification model.

    """

    def __init__(self, train_data_name, dataloader_idx):
        """
        Args:
            data: Data dictionary containing 'train', 'val', 'test' and 'name' keys.
        """
        super().__init__(train_data_name, dataloader_idx)
        # for key, idx in dataloader_idx.items():
        #     self.acc[key] = [tm.Accuracy(average='micro', threshold=0)] * len(idx)

        assert check_unique_list(dataloader_idx["val"])
        assert check_unique_list(dataloader_idx["test"])

        self.acc = {
            "val": {idx: tm.Accuracy(average="micro", threshold=0) for idx in dataloader_idx["val"]},
            "test": {idx: tm.Accuracy(average="micro", threshold=0) for idx in dataloader_idx["test"]},
        }

        # self.acc = {
        #     "val": [(tm.Accuracy(average="micro", threshold=0), idx) for idx in dataloader_idx["val"]],
        #     "test": [(tm.Accuracy(average="micro", threshold=0), idx) for idx in dataloader_idx["test"]],
        # }

    def setup(self, pl_module, data):
        with torch.no_grad():
            model = sklearn.linear_model.LogisticRegression(penalty="l2", C=1)
            X, y = embed_dataset(pl_module.backbone, data)
            X, y = X.detach().cpu().numpy(), y.detach().cpu().numpy()
            self.scaler = StandardScaler()
            self.scaler.fit(X)
            X = self.scaler.transform(X)
            model.fit(X, y)
            self.model = model

        # Could shorten this with recursion but might be less clear
        for acc in self.acc["val"].values():
            acc.reset()
        for acc in self.acc["test"].values():
            acc.reset()

    def step(self, pl_module, X, y, dataloader_idx, stage):
        X = pl_module(X).squeeze()
        X, y = X.detach().cpu().numpy(), y.detach().cpu()
        X = self.scaler.transform(X)
        preds = self.model.predict(X)

        self.acc[stage][dataloader_idx].update(torch.tensor(preds), y)

    def end(self, pl_module, stage):

        for dataloader_idx, acc in self.acc[stage].items():
            # Grab evaluation data-set name directly from dataloader
            eval_name, _ = pl_module.trainer.datamodule.data[stage][dataloader_idx]
            pl_module.log(f"{stage}/{self.train_data_name}/linear_acc/{eval_name}", acc.compute())


class Linear_Eval_PL(Data_Eval):
    """
    Callback to perform linear evaluation at the end of each epoch.

    Attributes:
        data: Data dictionary containing 'train', 'val', 'test' and 'name' keys.
        clf: Linear classification model.

    """

    def __init__(self, dataset_name, dataloader_idx):
        """
        Args:
            data: Data dictionary containing 'train', 'val', 'test' and 'name' keys.
        """
        super().__init__(dataset_name, dataloader_idx)

        assert check_unique_list(dataloader_idx["val"])
        assert check_unique_list(dataloader_idx["test"])

        self.acc = {
            "val": {idx: tm.Accuracy(average="micro", threshold=0) for idx in dataloader_idx["val"]},
            "test": {idx: tm.Accuracy(average="micro", threshold=0) for idx in dataloader_idx["test"]},
        }

        self.normalize = Normalize(0, 1)

    def setup(self, pl_module, data):
        ## Train linear classifier ##

        # Enable gradients to train linear model but freeze encoder
        torch.set_grad_enabled(True)
        freeze_model(pl_module.backbone)
        pl_module.backbone.eval()

        # Calculate mean and std in feature space and save normalization constants
        mean, std = compute_encoded_mu_sig(data, pl_module)
        self.normalize = Normalize(mean, std)

        # Define data-loader and initialize logistic regression model
        train_dataloader = DataLoader(data, **pl_module.config["logreg_dataloader"])
        model = LogisticRegression(**pl_module.config["logreg"]).to(pl_module.device)

        for epoch in np.arange(pl_module.config["linear"]["n_epochs"]):
            for data in train_dataloader:
                X, y = data
                X, y = X.to(pl_module.device), y.to(pl_module.device)
                X = pl_module(X)
                X = self.normalize(X)
                model.training_step(X, y)

        # Disable gradients to train linear model
        torch.set_grad_enabled(False)

        # Save model
        model.eval()
        self.model = model
        unfreeze_model(pl_module.backbone)

        # Could shorten this with recursion but might be less clear
        for acc in self.acc["val"].values():
            acc.reset()
        for acc in self.acc["test"].values():
            acc.reset()

    def step(self, pl_module, X, y, dataloader_idx, stage):
        X = pl_module.backbone(X)
        X = self.normalize(X)
        preds = self.model(X).detach().cpu()

        self.acc[stage][dataloader_idx].update(preds, y.detach().cpu())
        # self.acc.update(preds, y.detach().cpu())

    def end(self, pl_module, stage):
        for dataloader_idx, acc in self.acc[stage].items():
            # Grab evaluation data-set name directly from dataloader
            eval_name, _ = pl_module.trainer.datamodule.data[stage][dataloader_idx]
            pl_module.log(f"{stage}/{self.train_data_name}/linear_acc/{eval_name}", acc.compute())

        # pl_module.log(f"{stage}/{self.name}/linear_acc", self.acc.compute())


class LogisticRegression(nn.Module):
    """Logistic regression model."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        bias: bool = True,
        learning_rate: float = 1e-4,
        optimizer: Type[Optimizer] = Adam,
        # l1_strength: float = 0.0,
        # l2_strength: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            input_dim: number of dimensions of the input (at least 1)
            num_classes: number of class labels (binary: 2, multi-class: >2)
            bias: specifies if a constant or intercept should be fitted (equivalent to fit_intercept in sklearn)
            learning_rate: learning_rate for the optimizer
            optimizer: the optimizer to use (default: ``Adam``)
            l1_strength: L1 regularization strength (default: ``0.0``)
            l2_strength: L2 regularization strength (default: ``0.0``)
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.bias = bias
        self.learning_rate = learning_rate
        # self.l1_strength = l1_strength
        # self.l2_strength = l2_strength

        self.linear = nn.Linear(in_features=self.input_dim, out_features=self.num_classes, bias=bias)
        self.optimizer = optimizer(self.linear.parameters(), lr=self.learning_rate)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x.squeeze())
        y_hat = softmax(x, dim=-1)
        return y_hat

    def training_step(self, x: Tensor, y: Tensor):
        # Encode & flatten
        # x = self.encoder(x).squeeze()

        y_hat = self.linear(x.squeeze())

        # PyTorch cross_entropy function combines log_softmax and nll_loss in single function
        loss = F.cross_entropy(y_hat, y, reduction="sum")

        # # L1 regularizer
        # if self.l1_strength > 0:
        #     l1_reg = self.linear.weight.abs().sum()
        #     loss += self.l1_strength * l1_reg

        # # L2 regularizer
        # if self.l2_strength > 0:
        #     l2_reg = self.linear.weight.pow(2).sum()
        #     loss += self.l2_strength * l2_reg

        # Normalize loss by number of samples
        loss /= x.size(0)

        # Backpropagate and step optimizer
        loss.backward()
        self.optimizer.step()

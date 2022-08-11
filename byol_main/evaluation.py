import logging

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as tmF
import torchmetrics as tm
import wandb
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
from pytorch_lightning.callbacks import Callback
from torch.utils.data import DataLoader
from statistics import mean
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import RidgeClassifier

from dataloading.utils import dset2tens
from paths import Path_Handler
from networks.models import MLPHead, LogisticRegression
from utilities import log_examples, fig2img, embed_dataset


def knn_predict(
    feature: torch.Tensor,
    feature_bank: torch.Tensor,
    target_bank: torch.Tensor,
    num_classes: int,
    knn_k: int = 200,
    knn_t: float = 0.1,
    leave_first_out=False,
) -> torch.Tensor:
    """Code copied from https://github.com/lightly-ai/lightly/blob/master/lightly/utils/benchmarking.py

    Run kNN predictions on features based on a feature bank
    This method is commonly used to monitor performance of self-supervised
    learning methods.
    The default parameters are the ones
    used in https://arxiv.org/pdf/1805.01978v1.pdf.
    Args:
        feature:
            Tensor of shape [N, D] for which you want predictions
        feature_bank:
            Tensor of a database of features used for kNN, of shape [D, N] where N is len(l datamodule)
        target_bank:
            Labels for the features in our feature_bank, of shape ()
        num_classes:
            Number of classes (e.g. `10` for CIFAR-10)
        knn_k:
            Number of k neighbors used for kNN
        knn_t:
            Temperature parameter to reweights similarities for kNN
    Returns:
        A tensor containing the kNN predictions
    Examples:
        >>> images, targets, _ = batch
        >>> feature = backbone(images).squeeze()
        >>> # we recommend to normalize the features
        >>> feature = F.normalize(feature, dim=1)
        >>> pred_labels = knn_predict(
        >>>     feature,
        >>>     feature_bank,
        >>>     target_bank,
        >>>     num_classes=10,
        >>> )
    """

    assert target_bank.min() >= 0

    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(
        feature.squeeze(), feature_bank.squeeze()
    )  # (B, D) matrix. mult (D, N) gives (B, N) (as feature dim got summed over to got cos sim)

    # [B, K]
    # this will be slow if feature_bank is large (e.g. 100k datapoints)
    if leave_first_out is True:
        sim_weight, sim_idx = sim_matrix.topk(k=knn_k + 1, dim=-1)
        sim_weight, sim_idx = sim_weight[:, 1:], sim_idx[:, 1:]
    elif leave_first_out is False:
        sim_weight, sim_idx = sim_matrix.topk(k=knn_k, dim=-1)

    # [B, K]
    # target_bank is (1, N) (due to .t() in init)
    # feature.size(0) is the validation batch size
    # expand copies target_bank to (val_batch, N)
    # gather than indexes the N dimension to place the right labels (of the top k features), making sim_labels (val_batch) with values of the correct labels
    # if these aren't true, will get index error when trying to index target_bank
    assert sim_idx.min() >= 0
    assert sim_idx.max() < target_bank.size(0)
    sim_labels = torch.gather(target_bank.expand(feature.size(0), -1), dim=-1, index=sim_idx)
    # we do a reweighting of the similarities
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_y = torch.zeros(feature.size(0) * knn_k, num_classes)
    one_hot_y = one_hot_y.type_as(target_bank)

    # [B*K, C]
    one_hot_y = one_hot_y.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)

    # weighted score ---> [B, C]
    pred_scores = torch.sum(
        one_hot_y.view(feature.size(0), -1, num_classes) * sim_weight.unsqueeze(dim=-1),
        dim=1,
    )
    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels


def knn_weight(
    feature: torch.Tensor,
    feature_bank: torch.Tensor,
    knn_k: int = 5,
    knn_t: float = 0.1,
) -> torch.Tensor:
    """Code modified from https://github.com/lightly-ai/lightly/blob/master/lightly/utils/benchmarking.py

    Run kNN predictions on features based on a feature bank
    This method is commonly used to monitor performance of self-supervised
    learning methods.
    The default parameters are the ones
    used in https://arxiv.org/pdf/1805.01978v1.pdf.
    Args:
        feature:
            Tensor of shape [N, D] for which you want predictions
        feature_bank:
            Tensor of a database of features used for kNN, of shape [D, N] where N is len(l datamodule)
        target_bank:
            Labels for the features in our feature_bank, of shape ()
        num_classes:
            Number of classes (e.g. `10` for CIFAR-10)
        knn_k:
            Number of k neighbors used for kNN
        knn_t:
            Temperature parameter to reweights similarities for kNN
    Returns:
        A tensor containing the kNN predictions
    Examples:
        >>> images, targets, _ = batch
        >>> feature = backbone(images).squeeze()
        >>> # we recommend to normalize the features
        >>> feature = F.normalize(feature, dim=1)
        >>> pred_labels = knn_predict(
        >>>     feature,
        >>>     feature_bank,
        >>> )
    """

    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    # (B, D) matrix. mult (D, N) gives (B, N) (as feature dim got summed over to got cos sim)
    sim_matrix = torch.mm(feature.squeeze(), feature_bank.squeeze())

    # [B, K]
    sim_weight, sim_idx = sim_matrix.topk(k=knn_k, dim=-1)
    # this will be slow if feature_bank is large (e.g. 100k datapoints)
    sim_weight = sim_weight.squeeze()

    # we do a reweighting of the similarities
    # sim_weight = (sim_weight / knn_t).exp().squeeze().type_as(feature_bank)
    sim_weight = (sim_weight / knn_t).exp()

    return torch.mean(sim_weight, dim=-1, keepdim=False)


class Lightning_Eval(pl.LightningModule):
    # for many knn eval datasets
    def __init__(self, config):
        super().__init__()
        self.config = config

    def on_fit_start(self):
        self.config["data"]["mu"] = self.trainer.datamodule.mu
        self.config["data"]["sig"] = self.trainer.datamodule.sig

        ## Log size of data-sets ##
        logging_params = {"n_train": len(self.trainer.datamodule.data["train"])}

        for data in self.trainer.datamodule.data["val"]:
            logging_params[f"{data['name']}_n"] = len(data["val"])

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
        self.val_list = []

        ## Prepare for linear evaluation ##
        # Cycle through validation data-sets
        for idx, data in enumerate(self.trainer.datamodule.data["val"]):
            if self.config["evaluation"]["linear_eval"]:
                # Initialise linear eval data-set and run setup with training data
                lin_eval = Linear_Eval(data["name"], idx)
                lin_eval.setup(self, data["train"])

                # Add to list of evaluations
                self.val_list.append(lin_eval)

            # if self.config["knn_eval"]:
            #     self.val_list.append((KNN_Eval(name, idx)))

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        # Loop through validation data-sets
        for idx, data in enumerate(self.val_list):
            if dataloader_idx == idx:
                # Filter out validation sets that require different data-loader
                val_list_filtered = [val for val in self.val_list if val.dataloader_idx == idx]

                # Run validation step for filtered data-sets
                for val in val_list_filtered:
                    val.step(self, x, y)

    def on_validation_epoch_end(self):
        # Complete validation for all data-sets
        for val in self.val_list:
            val.end(self, stage="val")

    def test_step(self, batch, batch_idx):
        return


class Data_Eval:
    """
    Parent class for evaluation classes.
    """

    def __init__(self, dataset_name, dataloader_idx):

        self.name = dataset_name
        self.dataloader_idx = dataloader_idx

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

    def __init__(self, dataset_name, dataloader_idx):
        """
        Args:
            data: Data dictionary containing 'train', 'val', 'test' and 'name' keys.
        """
        super().__init__(dataset_name, dataloader_idx)
        self.acc = tm.Accuracy(average="micro", threshold=0)

    def setup(self, pl_module, data):

        with torch.no_grad():
            model = RidgeClassifier(normalize=True)
            X_train, y_train = embed_dataset(pl_module.backbone, data)
            X_train, y_train = X_train.detach().cpu().numpy(), y_train.detach().cpu().numpy()
            model.fit(X_train, y_train)
            self.model = model

        self.acc.reset()

    def step(self, pl_module, X, y):
        X = pl_module(X).squeeze()
        X, y = X.detach().cpu().numpy(), y.detach().cpu()
        preds = self.model.predict(X)
        self.acc.update(torch.tensor(preds), y)

    def end(self, pl_module, stage):
        pl_module.log(f"{stage}/{self.name}/linear_acc", self.acc.compute())

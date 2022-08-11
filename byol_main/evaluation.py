import logging
import pytorch_lightning as pl
import torch
import torchmetrics as tm

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier

from utilities import log_examples, embed_dataset


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
            #     if self.config["evaluation"]["linear_eval"]:
            #         # Initialise linear eval data-set and run setup with training data
            #         lin_eval = Linear_Eval(data["name"], idx)
            #         lin_eval.setup(self, data["train"])

            # Add to list of evaluations
            # self.val_list.append(lin_eval)

            if self.config["evaluation"]["ridge_eval"]:
                # Initialise linear eval data-set and run setup with training data
                ridge_eval = Ridge_Eval(data["name"], idx)
                ridge_eval.setup(self, data["train"])

                # Add to list of evaluations
                self.val_list.append(ridge_eval)

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


class Ridge_Eval(Data_Eval):
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
            model = RidgeClassifier()
            X, y = embed_dataset(pl_module.backbone, data)
            X, y = X.detach().cpu().numpy(), y.detach().cpu().numpy()
            self.scaler = StandardScaler()
            self.scaler.fit(X)
            X = self.scaler.transform(X)
            model.fit(X, y)
            self.model = model

        self.acc.reset()

    def step(self, pl_module, X, y):
        X = pl_module(X).squeeze()
        X, y = X.detach().cpu().numpy(), y.detach().cpu()
        X = self.scaler.transform(X)
        preds = self.model.predict(X)
        self.acc.update(torch.tensor(preds), y)

    def end(self, pl_module, stage):
        pl_module.log(f"{stage}/{self.name}/linear_acc", self.acc.compute())

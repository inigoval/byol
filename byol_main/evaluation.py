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

from dataloading.utils import dset2tens
from paths import Path_Handler
from networks.models import MLPHead, LogisticRegression
from utilities import log_examples, fig2img


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
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.knn_acc_val = tm.Accuracy(average="micro", threshold=0)
        self.knn_acc_test = {
            "conf": tm.Accuracy(average="micro", threshold=0),
            "unc": tm.Accuracy(average="micro", threshold=0),
        }

    def on_train_start(self):
        self.config["data"]["mu"] = self.trainer.datamodule.mu
        self.config["data"]["sig"] = self.trainer.datamodule.sig

        data = self.trainer.datamodule.data
        self.logger.log_hyperparams(
            {
                "n_val": len(data["val"]),
                "n_test": len(data["test"]),
                "n_train": len(data["train"]),
            }
        )
        # self.log("train/mu", self.trainer.datamodule.mu)
        # self.log("train/sig", self.trainer.datamodule.sig)

        # logger = self.logger.experiment

        if not self.config["debug"]:
            log_examples(self.logger, self.trainer.datamodule.data["train"])

    def on_validation_start(self):
        with torch.no_grad():
            data_bank = self.trainer.datamodule.data[
                "labelled"
            ]  # will get split into feature_bank and target_bank
            data_bank_loader = DataLoader(
                data_bank, 200
            )  # 200 is the batch size used for the unpacking below
            feature_bank = []
            target_bank = []
            for data in data_bank_loader:
                # Load data and move to correct device
                (x, y) = data  # supervised-style batch of (images, labels), with batch size from above
                x = x.type_as(self.dummy_param)
                y = y.type_as(self.dummy_param).long()

                # Encode data and normalize features (for kNN)
                feature = self.forward(x).squeeze()  # (batch, features)  e.g. (200, 512)
                feature = F.normalize(feature, dim=1)
                # tensor of all features, within which to find nearest-neighbours. Each is (B, N_features), N_features being the output dim of self.forward e.g. BYOL
                feature_bank.append(feature)
                target_bank.append(y)  # tensor with labels of those features

            # Save full feature bank for validation epoch
            # (features, len(l datamodule)) due to the .t() transpose
            self.feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
            # either (label_dim, len) or just (len) depending on if labels should have singleton label_dim dimension
            self.target_bank = torch.cat(target_bank, dim=0).t().contiguous()
            assert all(self.target_bank >= 0)

    def validation_step(self, batch, batch_idx):
        if hasattr(self, "feature_bank") and hasattr(self, "target_bank"):
            # Load batch
            x, y = batch
            # Extract + normalize features
            feature = self.forward(x).squeeze()
            feature = F.normalize(feature, dim=1)

            # Load feature bank and labels
            feature_bank = self.feature_bank.type_as(x)
            target_bank = self.target_bank.type_as(y)

            pred_labels = knn_predict(
                feature,  # feature to search for
                feature_bank,  # feature bank to identify NN within
                target_bank,  # labels of those features in feature_bank, same index
                self.config["data"]["classes"],
                knn_k=self.config["knn"]["neighbors"],
                knn_t=self.config["knn"]["temperature"],
                leave_first_out=self.config["knn"]["leave_first_out"],
            )

            top1 = pred_labels[:, 0]

            # Compute accuracy
            # assert top1.min() >= 0
            self.knn_acc_val.update(top1, y)

    def validation_epoch_end(self, outputs):
        if hasattr(self, "feature_bank") and hasattr(self, "target_bank"):
            self.log("val/kNN_acc", self.knn_acc_val.compute() * 100)
            self.knn_acc_val.reset()

    def test_step(self, batch, batch_idx):
        return

    #     def test_step(self, batch, batch_idx):
    #         if hasattr(self, "feature_bank") and hasattr(self, "target_bank"):
    #             x, y = batch
    #             # Extract + normalize features
    #             feature = self.forward(x).squeeze()
    #             feature = F.normalize(feature, dim=1)

    #             # Load feature bank and labels
    #             feature_bank = self.feature_bank.type_as(x)
    #             target_bank = self.target_bank.type_as(y)

    #             pred_labels = knn_predict(
    #                 feature,  # feature to search for
    #                 feature_bank,  # feature bank to identify NN within
    #                 target_bank,  # labels of those features in feature_bank, same index
    #                 self.config["data"]["classes"],
    #                 knn_k=self.config["knn"]["neighbors"],
    #                 knn_t=self.config["knn"]["temperature"],
    #                 leave_first_out=self.config["knn"]["leave_first_out"],
    #             )

    #             top1 = pred_labels[:, 0]

    #             # Compute accuracy
    #             # assert top1.min() >= 0
    #             self.knn_acc_test.update(top1, y)

    # def test_epoch_end(self, outputs):
    #     if hasattr(self, "feature_bank") and hasattr(self, "target_bank"):
    #         self.log("test/kNN_acc", self.knn_acc_test.compute() * 100)
    #         self.knn_acc_test.reset()
    #         # return self.knn_acc_test.compute() * 100


class Feature_Bank(Callback):
    """Code adapted from https://github.com/lightly-ai/lightly/blob/master/lightly/utils/benchmarking.py

    Calculates a feature bank for validation"""

    def __init__(self):
        super().__init__()

    def on_validation_epoch_start(self, trainer, pl_module):
        with torch.no_grad():
            encoder = pl_module.backbone

            data_bank = pl_module.trainer.datamodule.data["labelled"]
            data_bank_loader = DataLoader(data_bank, 200)
            feature_bank = []
            target_bank = []
            for data in data_bank_loader:
                # Load data and move to correct device
                x, y = data
                x = x.type_as(pl_module.dummy_param)
                y = y.type_as(pl_module.dummy_param).long()

                # Encode data and normalize features (for kNN)
                feature = encoder(x).squeeze()
                feature = F.normalize(feature, dim=1)
                feature_bank.append(feature)
                target_bank.append(y)

            # Save full feature bank for validation epoch
            pl_module.feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
            pl_module.target_bank = torch.cat(target_bank, dim=0).t().contiguous()


class Epoch_Averaged_Test(Callback):
    """Code adapted from https://github.com/lightly-ai/lightly/blob/master/lightly/utils/benchmarking.py

    Calculates a feature bank for validation"""

    def __init__(self):
        super().__init__()
        self.acc = {"conf": [], "unc": []}
        self.knn_acc = {
            "conf": tm.Accuracy(average="micro", threshold=0),
            "unc": tm.Accuracy(average="micro", threshold=0),
        }

    def on_validation_epoch_end(self, trainer, pl_module):
        config = pl_module.config
        if (
            hasattr(pl_module, "feature_bank")
            and hasattr(pl_module, "target_bank")
            and pl_module.current_epoch >= config["model"]["n_epochs"] - config["topk"] - 1
        ):

            for label, data in pl_module.trainer.datamodule.data["test_rgz"].items():
                loader = DataLoader(
                    data,
                    200,
                    num_workers=config["num_workers"],
                    prefetch_factor=20,
                    persistent_workers=config["persistent_workers"],
                )

                for batch in loader:
                    x, y = batch

                    x = x.type_as(pl_module.dummy_param)
                    y = y.type_as(pl_module.dummy_param).long()

                    # Extract + normalize features
                    feature = pl_module.forward(x).squeeze()
                    feature = F.normalize(feature, dim=1)

                    # Load feature bank and labels
                    feature_bank = pl_module.feature_bank.type_as(x)
                    target_bank = pl_module.target_bank.type_as(y)

                    pred_labels = knn_predict(
                        feature,  # feature to search for
                        feature_bank,  # feature bank to identify NN within
                        target_bank,  # labels of those features in feature_bank, same index
                        pl_module.config["data"]["classes"],
                        knn_k=pl_module.config["knn"]["neighbors"],
                        knn_t=pl_module.config["knn"]["temperature"],
                        leave_first_out=False,
                    )

                    top1 = pred_labels[:, 0].cpu().detach()

                    # Compute accuracy
                    # assert top1.min() >= 0
                    self.knn_acc[label].update(top1, y.cpu().detach())

                acc = self.knn_acc[label].compute()
                pl_module.log(f"test/kNN_acc_{label}", acc)
                self.acc[label].append(acc.item())

    def on_test_epoch_start(self, trainer, pl_module):
        if (
            hasattr(pl_module, "feature_bank")
            and hasattr(pl_module, "target_bank")
            and not pl_module.config["debug"]
        ):
            for key, value in self.acc.items():
                pl_module.log(f"test/kNN_acc_agg_{key}", mean(value))
            # pl_module.log("test/kNN_acc_agg", pl_module.knn_acc_test.compute())


class linear_net(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        n_classes = self.config["data"]["classes"]
        output_dim = self.config["model"]["output_dim"]
        self.logreg = LogisticRegression(output_dim, n_classes)
        self.ce_loss = torch.nn.CrossEntropyLoss()

        self.train_acc = tm.Accuracy(average="micro", top_k=1, threshold=0)
        self.val_acc = tm.Accuracy(average="micro", top_k=1, threshold=0)

        self.dummy_param = nn.Parameter(torch.empty(0))

        paths = Path_Handler()
        self.paths = paths.dict

    def forward(self, x):
        """Return prediction"""
        return self.logreg(x)

    def training_step(self, batch, batch_idx):
        # Load data and targets
        x, y = batch
        x = x.view(x.shape[0], -1)
        logits = self.forward(x)
        y_pred = logits.softmax(dim=-1)
        loss = self.ce_loss(logits, y)
        self.log("linear_eval/train_loss", loss, on_step=False, on_epoch=True)

        # predictions = torch.argmax(logits, dim=1).int()
        # acc = tmF.accuracy(y_pred, y)
        # self.log("linear_eval/train_acc", acc, on_step=False, on_epoch=True)

        self.train_acc(y_pred, y)
        self.log("linear_eval/train_acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.type_as(self.dummy_param)

        x = x.view(x.shape[0], -1)
        logits = self.forward(x)
        # y_pred = logits.softmax(dim=-1)

        loss = self.ce_loss(logits, y)
        self.log("linear_eval/val_loss", loss)

        # predictions = torch.argmax(logits, dim=1).int()
        # acc = tmF.accuracy(y_pred, y)
        self.val_acc(logits, y)
        self.log("linear_eval/val_acc", self.val_acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        lr = self.config["linear"]["lr"]
        mom = self.config["linear"]["momentum"]
        w_decay = self.config["linear"]["weight_decay"]

        params = self.logreg.parameters()

        opts = {
            "adam": lambda p: torch.optim.Adam(
                p,
                lr=lr,
                weight_decay=w_decay,
            ),
            "sgd": lambda p: torch.optim.SGD(
                p,
                lr=lr,
                momentum=mom,
                weight_decay=w_decay,
            ),
        }

        opt = opts[self.config["linear"]["opt"]](params)

        return opt


class pca_net(nn.Module):
    def __init__(self, config):
        super(pca_net, self).__init__()
        self.config = config
        self.pca = IncrementalPCA(config["pca"]["n_dim"])

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.pca.transform(x)
        return torch.from_numpy(x).float()

    def fit(self, loader):
        print("Fitting PCA")
        for epoch in tqdm(np.arange(self.config["pca"]["n_epochs"])):
            print(f"Epoch {epoch}")
            for x, _ in tqdm(loader):
                x = x.view(x.shape[0], -1)
                x = x.cpu().detach().numpy()
                self.pca.partial_fit(x)


class Count_Similarity(Callback):
    """Code adapted from https://github.com/lightly-ai/lightly/blob/master/lightly/utils/benchmarking.py

    Calculates a feature bank for validation"""

    def __init__(self):
        super().__init__()

    def on_validation_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % 10 == 0:
            train_loader = pl_module.trainer.datamodule.train_dataloader()

            for batch in train_loader:
                (x, _), y = batch

                x = x.type_as(pl_module.dummy_param)
                y = y.type_as(pl_module.dummy_param).long()

                n_batch = len(y)
                r = pl_module.backbone_momentum(x.type_as(pl_module.dummy_param)).squeeze()
                r = F.normalize(r, dim=1)
                knn_k = np.clip(20, 1, n_batch - 1)
                sim_weights = knn_weight(r, r.t(), knn_k=knn_k).view(-1, 1).cpu().detach().numpy()

            # hist = np.histogram(sim_weights.cpu().detach().numpy(), bins=np.arange(13, 45))
            scaler = MinMaxScaler()
            sim_weights = scaler.fit_transform(sim_weights)

            fig, ax = plt.subplots(figsize=(13.0, 13.0))
            ax.hist(np.ravel(sim_weights), bins=10)
            ax.set_xlabel("Similarity")
            ax.set_ylabel("Count")
            img = fig2img(fig)
            pl_module.logger.log_image(key="similarity histogram", images=[img])

            # y_masked = [[value] for value in y_masked.tolist()]
            # table = wandb.Table(data=y_masked, columns=["arcsecond extension"])
            # self.logger.log({'masked arcsec histogram': wandb.plot.histogram(table, 'arcsecond extension', title:
            # pl_module.logger.log({f"arcsec masks epoch {epoch}": wandb.Histogram(np_histogram=hist)})
            # wandb.log({f"arcsec masks epoch {epoch}": wandb.Histogram(np_histogram=hist)})

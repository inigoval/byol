import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import pytorch_lightning as pl
import torchmetrics.functional as tmF
import numpy as np
from tqdm import tqdm

from statistics import mean
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from math import cos, pi


from utilities import LARSWrapper
from networks.models import MLPHead, LogisticRegression
from networks.models import ResNet18, ResNet50, WideResNet50_2
from utilities import freeze_model
from paths import Path_Handler
from eval import knn_predict


def byol_loss(x, y):
    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)
    return 2 - 2 * (x * y).sum(dim=-1)


class byol(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()  # save hyperparameters for easy inference
        self.config = config
        paths = Path_Handler()
        self.paths = paths.dict

        model_dict = {
            "resnet18": ResNet18,
            "resnet50": ResNet50,
            "wideresnet50_2": WideResNet50_2,
        }

        self.m_online = model_dict[config["model"]["arch"]](**config)
        self.m_target = model_dict[config["model"]["arch"]](**config)
        self.predictor = MLPHead(
            in_channels=self.m_online.projection.net[-1].out_features,
            **config["projection_head"],
        )

        # print(torch.cuda.get_device_name(torch.cuda.current_device()))
        # print(torch.cuda.is_available())
        self.best_acc = 0

        # Dummy parameter to move tensors to correct device
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        """Return representation"""
        return self.m_online.encoder(x)

    def training_step(self, batch, batch_idx):
        # Load both views of x (y is a dummy label)
        x, _ = batch
        x1, x2 = x

        # Get targets for each view
        with torch.no_grad():
            y1, y2 = self.m_target(x1), self.m_target(x2)

        # Get predictions for each view
        f1, f2 = self.predictor(self.m_online(x1)), self.predictor(self.m_online(x2))

        # Calculate and log loss
        loss = byol_loss(f1, y1) + byol_loss(f2, y2)
        loss = torch.mean(loss)
        self.log("train/loss", loss)

        # Update target network using EMA (no gradients)
        self.update_m_target()

        return loss

    def validation_step(self, batch, batch_idx):
        # Load batch
        x, y = batch

        # Extract + normalize features
        feature = self.m_online.encoder(x).squeeze()
        feature = F.normalize(feature, dim=1)

        # Load feature bank and labels
        feature_bank = self.feature_bank.type_as(x)
        target_bank = self.target_bank.type_as(y)

        pred_labels = knn_predict(
            feature,
            feature_bank,
            target_bank,
            self.config["data"]["classes"],
        )

        num = x.size()
        top1 = (pred_labels[:, 0] == y).sum()
        return (num, top1.item())

    def validation_epoch_end(self, outputs):
        total_num = 0
        total_top1 = 0
        for (num, top1) in outputs:
            total_num += num[0]
            total_top1 += top1

        acc = float(total_top1 / total_num)
        if acc > self.best_acc:
            self.best_acc = acc
        self.log("val/kNN_acc", acc * 100.0)
        self.log("val/max_kNN_acc", self.best_acc)

    def configure_optimizers(self):
        opts = {
            "adam": torch.optim.Adam(self.m_online.parameters(), lr=self.config["lr"]),
            "sgd": torch.optim.SGD(
                self.m_online.parameters(),
                lr=self.config["lr"],
                momentum=self.config["momentum"],
                weight_decay=self.config["weight_decay"],
            ),
        }

        opt = opts[self.config["train"]["opt"]]

        # Apply LARS wrapper if option is chosen
        if self.config["lars"]:
            opt = LARSWrapper(opt, eta=self.config["trust_coef"])

        if self.config["scheduler"] == "cosine":
            max_epochs = self.config["train"]["n_epochs"]
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max_epochs)
            return [opt], [scheduler]

        # Currently produces weird results
        elif self.config["scheduler"] == "warmupcosine":
            scheduler = LinearWarmupCosineAnnealingLR(
                opt,
                self.config["warmup_epochs"],
                max_epochs=self.config["train"]["n_epochs"],
            )
            lr_scheduler_config = {"scheduler": scheduler, "interval": "epoch"}
            return {"optimizer": opt, "scheduler": lr_scheduler_config}

        elif self.config["scheduler"] == "None":
            return opt

    @torch.no_grad()
    def update_m_target(self):
        """Update target network without gradients"""
        m = self.config["m"]

        if self.config["m_decay"]:
            epoch = self.current_epoch
            n_epochs = self.config["train"]["n_epochs"]
            m = 1 - (1 - m) * (cos(pi * epoch / n_epochs) + 1) / 2

        for param_q, param_k in zip(
            self.m_online.parameters(), self.m_target.parameters()
        ):
            param_k.data = param_k.data * m + param_q.data * (1.0 - m)

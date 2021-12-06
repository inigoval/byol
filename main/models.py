import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import pytorch_lightning as pl
import torchmetrics.functional as tmF
import numpy as np

from statistics import mean

from networks.models import ResNet18, MLPHead, LogisticRegression
from utilities import byol_loss, freeze_model
from paths import Path_Handler


class pretrain_net(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()  # save hyperparameters for easy inference
        self.config = config
        paths = Path_Handler()
        self.paths = paths.dict
        self.m_online = ResNet18(**config)
        self.m_target = ResNet18(**config)
        self.predictor = MLPHead(
            in_channels=self.m_online.projection.net[-1].out_features,
            **config["projection_head"]
        )

        self.best_acc = 0

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

    def test_step(self, batch, batch_idx):
        x, y = batch

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

        return opts[self.config["train"]["opt"]]

    @torch.no_grad()
    def update_m_target(self):
        """Update target network without gradients"""
        m = self.config["m"]
        for param_q, param_k in zip(
            self.m_online.parameters(), self.m_target.parameters()
        ):
            param_k.data = param_k.data * m + param_q.data * (1.0 - m)


class linear_net(pl.LightningModule):
    def __init__(self, config, encoder):
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.logreg = LogisticRegression(self.output_feature_dim, 2)
        self.ce_loss = torch.nn.CrossEntropyLoss()

        paths = Path_Handler()
        self.paths = paths.dict

    def forward(self, x):
        """Return prediction"""
        y = self.logreg(x)

    def training_step(self, batch, batch_idx):
        # Load data and targets
        x, y = batch
        print(x.shape)
        logits = self.forward(x)
        loss = self.ce_loss(logits, y)
        self.log("linear_eval/train/loss", loss)

        predictions = torch.argmax(logits, dim=1)
        acc = tmF.accuracy(predictions.softmax(dim=-1), y)
        self.log("linear_eval/train/acc", acc)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.ce_loss(logits, y)
        self.log("linear_eval/test/loss", loss)

        predictions = torch.argmax(logits, dim=1)
        acc = tmF.accuracy(predictions.softmax(dim=-1), y)
        self.log("linear_eval/test/acc")

    def configure_optimizers(self):
        lr = self.config["linear_model"]["lr"]
        opt = torch.optim.Adam(self.logreg.parameters(), lr=lr)

        return opt

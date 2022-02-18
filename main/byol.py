import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from math import cos, pi
from copy import deepcopy
from lightly.loss import SymNegCosineSimilarityLoss

from utilities import LARSWrapper
from networks.models import MLPHead
from networks.models import ResNet18, ResNet50, WideResNet50_2
from utilities import freeze_model
from evaluation import knn_predict


def byol_loss(x, y):
    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)
    return 2 - 2 * (x * y.detach()).sum(dim=-1)


class byol(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()  # save hyperparameters for easy inference
        self.config = config

        model_dict = {
            "resnet18": ResNet18,
            "resnet50": ResNet50,
            "wideresnet50_2": WideResNet50_2,
        }

        self.m_online = model_dict[config["model"]["arch"]](**config)
        self.predictor = MLPHead(
            in_channels=self.m_online.projection.net[-1].out_features,
            **config["projection_head"],
        )

        self.m_target = deepcopy(self.m_online)
        freeze_model(self.m_target)

        self.m = self.config["m"]

        self.best_acc = 0

        # Dummy parameter to move tensors to correct device
        self.dummy_param = nn.Parameter(torch.empty(0))

        self.criterion = SymNegCosineSimilarityLoss()

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
        # loss = byol_loss(f1, y1) + byol_loss(f2, y2)
        # loss = torch.mean(loss)
        loss = self.criterion((y1, f1), (y2, f2))
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

        num = len(y)
        top1 = (pred_labels[:, 0] == y).float().sum()
        return (num, top1.item())

    def validation_epoch_end(self, outputs):
        total_num = 0
        total_top1 = 0
        for (num, top1) in outputs:
            total_num += num
            total_top1 += top1

        acc = float(total_top1 / total_num) * 100
        if acc > self.best_acc:
            self.best_acc = acc
        self.log("val/kNN_acc", acc)
        self.log("val/max_kNN_acc", self.best_acc)

    def configure_optimizers(self):
        lr = self.config["lr"]
        mom = self.config["momentum"]
        w_decay = self.config["weight_decay"]

        opts = {
            "adam": torch.optim.Adam(
                list(self.m_online.parameters()) + list(self.predictor.parameters()),
                lr=lr,
                weight_decay=w_decay,
            ),
            "sgd": torch.optim.SGD(
                list(self.m_online.parameters()) + list(self.predictor.parameters()),
                lr=lr,
                momentum=mom,
                weight_decay=w_decay,
            ),
        }

        opt = opts[self.config["opt"]]

        # Apply LARS wrapper if option is chosen
        if self.config["lars"]:
            opt = LARSWrapper(opt, eta=self.config["trust_coef"])

        max_epochs = self.config["model"]["n_epochs"]
        if self.config["scheduler"] == "none":
            return opt

        elif self.config["scheduler"] == "cosine":
            max_epochs = max_epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max_epochs)

            return [opt], [scheduler]

        elif self.config["scheduler"] == "warmupcosine":
            scheduler = LinearWarmupCosineAnnealingLR(
                opt,
                self.config["warmup_epochs"],
                max_epochs=max_epochs,
            )

            return [opt], [scheduler]

    @torch.no_grad()
    def update_m_target(self):
        """Update target network without gradients"""
        m = self.config["m"]

        if self.config["m_decay"]:
            epoch = self.current_epoch
            n_epochs = self.config["model"]["n_epochs"]
            m = 1 - (1 - m) * (cos(pi * epoch / n_epochs) + 1) / 2

        for p_online, p_target in zip(
            self.m_online.parameters(), self.m_target.parameters()
        ):
            p_target.data = p_target.data * m + p_online.data * (1.0 - m)

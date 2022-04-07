import pytorch_lightning as pl
import torchvision
import torch
import torch.nn as nn
import lightly
import copy
import torch.nn.functional as F
import torchvision.models as M
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from math import cos, pi
from utilities import LARSWrapper
from lightly.models.modules.heads import BYOLProjectionHead
from lightly.models.utils import deactivate_requires_grad
from lightly.models.utils import update_momentum
from pytorch_lightning.callbacks import Callback

from evaluation import knn_predict
from networks.models import _get_backbone


class byol_old_lightly(pl.LightningModule):
    def __init__(self, config):
        # create a ResNet backbone and remove the classification head
        super().__init__()
        self.save_hyperparameters()  # save hyperparameters for easy inference
        self.config = config

        self.backbone = _get_backbone(config)

        # create a byol model based on ResNet
        features = self.config["model"]["features"]
        proj = self.config["projection_head"]
        self.projection_head = BYOLProjectionHead(features, proj["hidden"], proj["out"])
        self.prediction_head = BYOLProjectionHead(proj["out"], proj["hidden"], proj["out"])

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        self.criterion = lightly.loss.SymNegCosineSimilarityLoss()
        self.m = self.config["m"]

        self.dummy_param = nn.Parameter(torch.empty(0))
        self.best_acc = 0

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        return self.projection_head(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _ = batch

        # update momentum
        update_momentum(self.backbone, self.backbone_momentum, self.m)
        update_momentum(self.projection_head, self.projection_head_momentum, self.m)

        def step(x0_, x1_):
            x0_ = self.backbone(x0_).flatten(start_dim=1)
            x0_ = self.projection_head(x0_)
            x0_ = self.prediction_head(x0_)

            x1_ = self.backbone_momentum(x1_).flatten(start_dim=1)
            x1_ = self.projection_head_momentum(x1_)
            return x0_, x1_

        p0, z1 = step(x0, x1)
        p1, z0 = step(x1, x0)

        loss = self.criterion((z0, p0), (z1, p1))
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # Load batch
        x, y = batch

        # Extract + normalize features
        feature = self.backbone(x).squeeze()
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
        params = (
            list(self.backbone.parameters())
            + list(self.projection_head.parameters())
            + list(self.prediction_head.parameters())
        )

        opts = {
            "adam": lambda p: torch.optim.Adam(p, lr=lr),
            "sgd": lambda p: torch.optim.SGD(
                p,
                lr=lr,
                momentum=mom,
                weight_decay=w_decay,
            ),
        }

        opt = opts[self.config["opt"]]

        # Apply LARS wrapper if option is chosen
        if self.config["lars"]:
            opt = LARSWrapper(opt, eta=self.config["trust_coef"])

        if self.config["scheduler"] == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, self.config["model"]["n_epochs"]
            )
            return [opt], [scheduler]

        # Currently produces weird results
        elif self.config["scheduler"] == "warmupcosine":
            scheduler = LinearWarmupCosineAnnealingLR(
                opt,
                self.config["warmup_epochs"],
                max_epochs=self.config["train"]["n_epochs"],
            )
            return [opt], [scheduler]

        elif self.config["scheduler"] == "None":
            return opt

    def _get_backbone(self):
        net = _get_net()
        c_out = list(net.children())[-1].in_features

        net = torch.nn.Sequential(*list(net.children())[:-1])

        # Change first layer for color channels B/W images
        n_c = self.config["data"]["color_channels"]
        if n_c != 3:
            net[0] = nn.Conv2d(n_c, 64, 7, 2, 3)

        features = self.config["model"]["features"]
        backbone = nn.Sequential(
            *list(net.children())[:-1],
            nn.Conv2d(c_out, features, 1),
            nn.AdaptiveAvgPool2d(1),
        )

        return backbone

    def _get_net(self):
        networks = {
            "resnet18": M.resnet18,
            "resnet34": M.resnet34,
            "resnet50": M.resnet50,
            "resnet101": M.resnet101,
            "resnet152": M.resnet152,
            "wide_resnet50_2": M.wide_resnet50_2,
            "wide_resnet101_2": M.wide_resnet101_2,
            "efficientnetb7": M.efficientnet_b7,
        }

        return networks[self.config["model"]["arch"]]()


class BYOL(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()  # save hyperparameters for easy inference
        self.config = config

        self.backbone = _get_backbone(config)

        # create a byol model based on ResNet
        features = self.config["model"]["features"]
        proj = self.config["projection_head"]
        self.projection_head = BYOLProjectionHead(features, proj["hidden"], proj["out"])
        self.prediction_head = BYOLProjectionHead(proj["out"], proj["hidden"], proj["out"])

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        self.criterion = lightly.loss.NegativeCosineSimilarity()

        self.dummy_param = nn.Parameter(torch.empty(0))
        self.best_acc = 0

    def forward(self, x):
        return self.backbone(x).flatten(start_dim=1)

    def project(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        return p

    def project_momentum(self, x):
        y = self.backbone_momentum(x).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        z = z.detach()
        return z

    def training_step(self, batch, batch_idx):
        update_momentum(self.backbone, self.backbone_momentum, m=0.99)
        update_momentum(self.projection_head, self.projection_head_momentum, m=0.99)
        (x0, x1), _ = batch
        p0 = self.project(x0)
        z0 = self.project_momentum(x0)
        p1 = self.project(x1)
        z1 = self.project_momentum(x1)
        loss = 0.5 * (self.criterion(p0, z1) + self.criterion(p1, z0))
        self.log("train/loss", loss)
        return loss

    def configure_optimizers(self):
        params = (
            list(self.backbone.parameters())
            + list(self.projection_head.parameters())
            + list(self.prediction_head.parameters())
        )
        optim = torch.optim.SGD(
            params,
            lr=6e-2 * self.config["batch_size"] / 256,
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, self.config["model"]["n_epochs"]
        )
        return [optim], [scheduler]

    def validation_step(self, batch, batch_idx):
        # Load batch
        x, y = batch

        # Extract + normalize features
        feature = self.backbone(x).squeeze()
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


class Update_M(Callback):
    """Updates EMA momentum"""

    def __init__(self):
        super().__init__()

    def on_training_epoch_end(self, trainer, pl_module):
        with torch.no_grad():
            config = pl_module.config
            epoch = pl_module.current_epoch
            n_epochs = config["model"]["n_epochs"]
            pl_module.m = 1 - (1 - pl_module.m) * (cos(pi * epoch / n_epochs) + 1) / 2

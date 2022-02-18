import pytorch_lightning as pl
import torchvision
import torch
import torch.nn as nn
import lightly
import copy
import torch.nn.functional as F
import torch

from lightly.models.modules.heads import BYOLProjectionHead
from lightly.models.utils import deactivate_requires_grad
from lightly.models.utils import update_momentum

from evaluation import knn_predict


class byol(pl.LightningModule):
    def __init__(self, config):
        # create a ResNet backbone and remove the classification head
        super().__init__()
        self.save_hyperparameters()  # save hyperparameters for easy inference
        self.config = config

        resnet = torchvision.models.resnet18()
        last_conv_channels = list(resnet.children())[-1].in_features
        features = self.config["model"]["features"]
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.Conv2d(last_conv_channels, features, 1),
            nn.AdaptiveAvgPool2d(1),
        )

        # create a byol model based on ResNet
        self.projection_head = BYOLProjectionHead(features, 1024, 256)
        self.prediction_head = BYOLProjectionHead(256, 1024, 256)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        self.criterion = lightly.loss.SymNegCosineSimilarityLoss()
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.best_acc = 0

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        return self.projection_head(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _ = batch

        # update momentum
        update_momentum(self.backbone, self.backbone_momentum, 0.99)
        update_momentum(self.projection_head, self.projection_head_momentum, 0.99)

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
        params = (
            list(self.backbone.parameters())
            + list(self.projection_head.parameters())
            + list(self.prediction_head.parameters())
        )
        optim = torch.optim.SGD(params, lr=6e-2, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, self.config["model"]["n_epochs"]
        )
        return [optim], [scheduler]

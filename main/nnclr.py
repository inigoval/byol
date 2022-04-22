import torch
from torch import nn
import torchvision
import pytorch_lightning as pl

from lightly.loss import NTXentLoss
from lightly.models.modules import NNCLRProjectionHead
from lightly.models.modules import NNCLRPredictionHead
from lightly.models.modules import NNMemoryBankModule

from evaluation import Lightning_Eval
from networks.models import _get_backbone


class NNCLR(Lightning_Eval):
    def __init__(self, config):
        super().__init__(config)
        self.save_hyperparameters()  # save hyperparameters for easy inference
        self.config = config

        self.backbone = _get_backbone(config)

        features = self.config["model"]["features"]
        proj = self.config["projection_head"]

        self.projection_head = NNCLRProjectionHead(features, proj["hidden"], proj["out"])
        self.prediction_head = NNCLRProjectionHead(proj["out"], proj["hidden"], proj["out"])
        self.memory_bank = NNMemoryBankModule(size=4096)

        self.criterion = NTXentLoss()

        self.dummy_param = nn.Parameter(torch.empty(0))

    def project(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        z = z.detach()
        return z, p

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _ = batch
        x0 = x0.type_as(self.dummy_param)
        x1 = x1.type_as(self.dummy_param)

        z0, p0 = self.project(x0)
        z1, p1 = self.project(x1)
        z0 = self.memory_bank(z0, update=False)
        z1 = self.memory_bank(z1, update=True)
        loss = 0.5 * (self.criterion(z0, p1) + self.criterion(z1, p0))
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.06)
        return optim

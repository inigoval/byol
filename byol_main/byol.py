import torch
import torch.nn as nn
import lightly
import copy

from math import cos, pi
from utilities import _optimizer
from lightly.models.modules.heads import BYOLProjectionHead
from lightly.models.utils import deactivate_requires_grad
from lightly.models.utils import update_momentum
from pytorch_lightning.callbacks import Callback

from evaluation import Lightning_Eval
from networks.models import _get_backbone

# Lightning_Eval superclass adds validation step options for kNN evaluation
class BYOL(Lightning_Eval):
    def __init__(self, config):
        super().__init__(config)
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

        self.m = config["m"]

    def forward(self, x):
        return self.backbone(x)  # dimension (batch, features), features from config e.g. 512

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
        # Update momentum value
        update_momentum(self.backbone, self.backbone_momentum, m=self.m)
        update_momentum(self.projection_head, self.projection_head_momentum, m=self.m)

        # Load in data
        (x0, x1), _ = batch
        x0 = x0.type_as(self.dummy_param)
        x1 = x1.type_as(self.dummy_param)
        p0 = self.project(x0)
        z0 = self.project_momentum(x0)
        p1 = self.project(x1)
        z1 = self.project_momentum(x1)
        loss = 0.5 * (self.criterion(p0, z1) + self.criterion(p1, z0))
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        if self.config["m_decay"]:
            self.update_m()

    def configure_optimizers(self):
        params = (
            list(self.backbone.parameters())
            + list(self.projection_head.parameters())
            + list(self.prediction_head.parameters())
        )

        return _optimizer(params, self.config)

    def update_m(self):
        with torch.no_grad():
            epoch = self.current_epoch
            n_epochs = self.config["model"]["n_epochs"]
            self.m = 1 - (1 - self.m) * (cos(pi * epoch / n_epochs) + 1) / 2


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

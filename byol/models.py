import torch
import torch.nn as nn
import lightly
import copy

import logging
from math import cos, pi
from lightly.models.modules.heads import BYOLProjectionHead
from lightly.models.utils import deactivate_requires_grad
from lightly.models.utils import update_momentum
from pytorch_lightning.callbacks import Callback

from byol.evaluation import Lightning_Eval
from byol.utilities import _optimizer, _scheduler
from byol.resnet import _get_resnet


class BYOLProjectionHead(nn.Module):
    """Projection head used for BYOL.

    "This MLP consists in a linear layer with output size 4096 followed by
    batch normalization, rectified linear units (ReLU), and a final
    linear layer with output dimension 256." [0]

    [0]: BYOL, 2020, https://arxiv.org/abs/2006.07733

    """

    def __init__(self, input_dim: int = 2048, hidden_dim: int = 4096, output_dim: int = 256):
        super().__init__()

        layers = []
        batch_norm = nn.BatchNorm1d(hidden_dim)
        non_linearity = nn.ReLU()

        layers.append(nn.Linear(input_dim, hidden_dim, bias=False))
        layers.append(batch_norm)
        layers.append(non_linearity)
        layers.append(nn.Linear(hidden_dim, output_dim, bias=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        """Computes one forward pass through the projection head.

        Args:
            x:
                Input of shape bsz x num_ftrs.

        """
        return self.layers(x)

class BYOL(Lightning_Eval):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.save_hyperparameters(ignore=["encoder"])  # save hyperparameters for easy inference

        self.encoder = _get_resnet(**self.config["model"]["architecture"])
        self.encoder.dim = self.encoder.features

        # create a byol model based on ResNet
        features = self.config["model"]["architecture"]["features"]
        proj = self.config["model"]["projection_head"]
        # these are both basically small dense networks of different sizes
        # architecture is: linear w/ relu, batch-norm, linear by default: representation (features)=512, hidden (both heads)=1024, out=256
        # so projection_head is 512->1024,relu/BN,1024->256
        # and prediction_head is 256->1024,relu/BN,1024->256
        self.projection_head = BYOLProjectionHead(features, proj["hidden"], proj["out"])
        self.prediction_head = BYOLProjectionHead(proj["out"], proj["hidden"], proj["out"])

        self.encoder_momentum = copy.deepcopy(self.encoder)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.encoder_momentum)
        deactivate_requires_grad(self.projection_head_momentum)
        self.criterion = lightly.loss.NegativeCosineSimilarity()

        self.dummy_param = nn.Parameter(torch.empty(0))

        self.m = self.config["model"]["m"]

    def forward(self, x):
        return self.encoder(x)  # dimension (batch, features), features from config e.g. 512

    def project(self, x):
        # representation
        y = self.encoder(x).flatten(start_dim=1)
        # projection
        z = self.projection_head(y)
        # prediction (of proj of target network)
        p = self.prediction_head(z)
        return p

    def project_momentum(self, x):
        y = self.encoder_momentum(x).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        z = z.detach()
        return z

    def training_step(self, batch, batch_idx):
        # Update momentum value
        update_momentum(self.encoder, self.encoder_momentum, m=self.m)
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
        if self.config["model"]["m_decay"]:
            self.update_m()

    def configure_optimizers(self):
        # Scale learning rate with batch size
        self.config["model"]["optimizer"]["lr"] *= self.config["model"]["optimizer"]["batch_size"] / 256

        params = (
            list(self.encoder.parameters())
            + list(self.projection_head.parameters())
            + list(self.prediction_head.parameters())
        )

        opt = _optimizer(params, **self.config["model"]["optimizer"])

        if self.config["model"]["scheduler"]["decay_type"].lower() == "none":
            return opt
        else:
            scheduler = _scheduler(
                opt, self.config["model"]["n_epochs"], **self.config["model"]["scheduler"]
            )
            return [opt], [scheduler]

    def update_m(self):
        with torch.no_grad():
            epoch = self.current_epoch
            n_epochs = self.config["model"]["model"]["n_epochs"]
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

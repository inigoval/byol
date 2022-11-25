import torch
import torch.nn as nn
import lightly
import copy
import torchmetrics as tm

from math import cos, pi
from utilities import _optimizer
from lightly.models.modules.heads import BYOLProjectionHead
from lightly.models.utils import deactivate_requires_grad
from lightly.models.utils import update_momentum
from pytorch_lightning.callbacks import Callback
import torch.nn.functional as F
import torch.nn as nn

from evaluation import Lightning_Eval
from networks.models import _get_backbone, _get_net

## Needs to be rewritten ##


class Supervised(
    Lightning_Eval
):  # Lightning_Eval superclass adds validation step options for kNN evaluation
    def __init__(self, config):
        super().__init__(config)
        self.save_hyperparameters()  # save hyperparameters for easy inference
        self.config = config

        self.encoder = build_net(config)
        self.head = nn.Linear(config["model"]["features"], config["data"]["classes"])

        # create a byol model based on ResNet

        self.acc_val = tm.Accuracy(average="micro", threshold=0)
        self.knn_acc_val = tm.Accuracy(average="micro", threshold=0)
        self.knn_acc_test = tm.Accuracy(average="micro", threshold=0)

        self.dummy_param = nn.Parameter(torch.empty(0))

    def on_train_start(self):
        self.config["data"]["mu"] = self.trainer.datamodule.mu
        self.config["data"]["sig"] = self.trainer.datamodule.sig

    def forward(self, x):
        # dimension (batch, features), features from config e.g. 512
        return self.encoder(x)

    def predict(self, x):
        x = self.encoder(x).squeeze()
        x = self.head(x)
        x = F.softmax(x, dim=-1)
        return x

    def training_step(self, batch, batch_idx):

        # Load in data
        x, y = batch

        y_pred = self.predict(x)

        loss = F.cross_entropy(y_pred, y)
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # kNN metric
        x, y = batch
        if hasattr(self, "feature_bank") and hasattr(self, "target_bank"):
            # Load batch
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

        # classification accuracy with head
        y_pred = self.predict(x)
        self.acc_val.update(y_pred, y)

    def validation_epoch_end(self, outputs):
        if hasattr(self, "feature_bank") and hasattr(self, "target_bank"):
            self.log("val/kNN_acc", self.knn_acc_val.compute() * 100)
            self.knn_acc_val.reset()

        self.log("val/acc", self.acc_val.compute() * 100)
        self.acc_val.reset()

    def configure_optimizers(self):
        params = list(self.encoder.parameters()) + list(self.head.parameters())

        return _optimizer(params, self.config)


def build_net(config):
    net = _get_net(config)
    c_out = list(net.children())[-1].in_features

    # separate head and backbone
    net = torch.nn.Sequential(*list(net.children())[:-1])

    # output dim of e.g. resnet, once the classification layer is removed (below)

    # Change first layer for color channels B/W images
    n_c = config["data"]["color_channels"]
    if n_c != 3:
        # c_out, k, s, p = net[0].out_channels, net[0].kernel_size, net[0].stride, net[0].padding
        # net[0] = nn.Conv2d(n_c, c_out, kernel_size=k, stride=s, padding=p, bias=False)
        net[0] = nn.Conv2d(n_c, 64, kernel_size=7, stride=2, padding=2, bias=False)

    if config["model"]["downscale"]:
        net[0] = nn.Conv2d(n_c, 64, kernel_size=3, stride=1, padding=1, bias=False)

    features = config["model"]["features"]  # e.g. 512
    backbone = nn.Sequential(
        *list(net.children())[:-1],  # resnet minus classification layer
        nn.Conv2d(c_out, features, 1),  # another conv layer, to `features` channels
        nn.AdaptiveAvgPool2d((1, 1)),  # remove filter height/width, so now just (batch, features)
    )

    return backbone

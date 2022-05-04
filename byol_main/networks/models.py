import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.models as models
import torchvision.models as M


class MLPHead(nn.Module):
    """Fully connected head wtih a single hidden layer"""

    def __init__(self, in_channels, mlp_hidden_size, projection_size):
        super(MLPHead, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_size),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, projection_size),
        )

    def forward(self, x):
        return self.net(x)


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


def _get_backbone(config):
    net = _get_net(config)  # e.g. resnet
    c_out = list(net.children())[-1].in_features  # output dim of e.g. resnet, once the classification layer is removed (below)

    net = torch.nn.Sequential(*list(net.children())[:-1])  # i.e. remove the last layer of resnet (aka the classification layer) as default-defined

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
        nn.AdaptiveAvgPool2d(1),  # remove filter height/width, so now just (batch, features)
    )

    return backbone


def _get_net(config):
    networks = {
        "resnet18": M.resnet18,
        "resnet34": M.resnet34,
        "resnet50": M.resnet50,
        "resnet101": M.resnet101,
        "resnet152": M.resnet152,
        "wide_resnet50_2": M.wide_resnet50_2,
        "wide_resnet101_2": M.wide_resnet101_2,
        "efficientnetb7": M.efficientnet_b7,
        "efficientnetb0": M.efficientnet_b0  # not tested, could be v useful re zoobot
    }

    return networks[config["model"]["architecture"]]()

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.models as models
import torchvision.models as M

from networks.layers import conv_block, convT_block, linear_block, UPSoftmax


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
    net = _get_net(config)
    c_out = list(net.children())[-1].in_features

    net = torch.nn.Sequential(*list(net.children())[:-1])

    # Change first layer for color channels B/W images

    # if config["model"]["downscale"]:
    #     net[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    n_c = config["data"]["color_channels"]
    if n_c != 3:
        net[0].in_channels = n_c

    features = config["model"]["features"]
    backbone = nn.Sequential(
        *list(net.children())[:-1],
        nn.Conv2d(c_out, features, 1),
        nn.AdaptiveAvgPool2d(1),
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
    }

    return networks[config["model"]["architecture"]]()

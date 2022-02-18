import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.models as models

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


class ResNet18(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(ResNet18, self).__init__()
        resnet = models.resnet18(pretrained=False)

        self.encoder = torch.nn.Sequential(*list(resnet.children())[:-1])

        # Change first layer for color channels B/W images
        n_c = kwargs["data"]["color_channels"]
        if n_c != 3:
            self.encoder[0] = nn.Conv2d(n_c, 64, 7, 2, 3)

        features = kwargs["model"]["features"]
        if features:
            c_out = list(resnet.children())[-1].in_features
            self.encoder = nn.Sequential(
                self.encoder,
                nn.Conv2d(c_out, features, 1),
                nn.AdaptiveAvgPool2d(1),
            )

        # Add projection layer
        self.projection = MLPHead(
            in_channels=resnet.fc.in_features, **kwargs["projection_head"]
        )

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.shape[0], h.shape[1])
        return self.projection(h)

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.shape[0], h.shape[1])
        return h


class ResNet50(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(ResNet50, self).__init__()
        resnet = models.resnet50(pretrained=False)

        self.encoder = torch.nn.Sequential(*list(resnet.children())[:-1])

        # Change first layer for 1 channel B/W images
        self.encoder[0] = nn.Conv2d(1, 64, 7, 2, 3)

        c_out = list(resnet.children())[-1].in_features
        features = kwargs["model"]["features"]

        self.encoder = nn.Sequential(
            self.encoder,
            #            nn.Conv2d(c_out, features, 1),
            #             nn.AdaptiveAvgPool2d(1),
        )

        # Add projection layer
        self.projection = MLPHead(
            in_channels=resnet.fc.in_features, **kwargs["projection_head"]
        )

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.shape[0], h.shape[1])
        return self.projection(h)


class WideResNet50_2(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(WideResNet50_2, self).__init__()
        resnet = models.wide_resnet50_2(pretrained=False)

        self.encoder = torch.nn.Sequential(*list(resnet.children())[:-1])

        # Change first layer for 1 channel B/W images
        self.encoder[0] = nn.Conv2d(1, 64, 7, 2, 3)

        # Add projection layer
        self.projection = MLPHead(
            in_channels=resnet.fc.in_features, **kwargs["projection_head"]
        )

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.shape[0], h.shape[1])
        return self.projection(h)


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

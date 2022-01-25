import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.models as models

from networks.layers import conv_block, convT_block, linear_block, UPSoftmax


class Tang(nn.Module):
    def __init__(self):
        super(Tang, self).__init__()
        # Conv2D(in_channels, out_channels, kernel size, stride, padding)

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, 11, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 24, 3, 1, 1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(24, 24, 3, 1, 1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(24, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(5, stride=4),
        )

        # 8192 -> 2048
        # 2048 -> 512
        # 512  -> 512
        # 512  -> 3
        self.linear1 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(),
        )

        self.linear2 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(-1, 256)
        x = self.linear1(x)


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


class ResNet50(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(ResNet50, self).__init__()
        resnet = models.resnet50(pretrained=False)

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

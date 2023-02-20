import logging

import torch
import torch.nn as nn
import torchvision.models as M
from torchvision.models.vision_transformer import _vision_transformer


class MLPHead(nn.Module):
    """
    Fully connected head with a single hidden layer. Batchnorm applied as first layer so that
    feature space of encoder doesn't need to be normalized.
    """

    def __init__(self, in_channels, mlp_hidden_size, projection_size):
        super(MLPHead, self).__init__()

        self.net = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.Linear(in_channels, mlp_hidden_size),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, projection_size),
        )

    def forward(self, x):
        return self.net(x)


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.batchnorm = nn.BatchNorm1d(input_dim)
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.batchnorm(x)
        x = self.linear(x)
        return x


# class CNN(nn.Module):
#     def __init__(self):


class Classification_Head(nn.Module):
    """
    Fully connected head with a single hidden layer. Batchnorm applied as first layer so that
    feature space of encoder doesn't need to be normalized.
    """

    def __init__(self, in_channels, mlp_hidden_size, projection_size):
        super().__init__()

        layers = [nn.Batchnorm1d(in_channels)] + [] * n_hidden + [nn.Linear(mlp_hidden_size)]

        self.net = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.Linear(in_channels, mlp_hidden_size),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, projection_size),
        )

    def forward(self, x):
        return self.net(x)


class Encoder(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.encoder = backbone

    def forward(self, x):
        return self.encoder(x)


def _get_backbone(config):

    logging.info(config["model"]["architecture"])

    if config["model"]["architecture"] == "zoobot":

        from zoobot.pytorch.estimators import define_model, efficientnet_standard

        assert config["model"]["features"] == 1280
        backbone = define_model.get_plain_pytorch_zoobot_model(
            output_dim=0,  # doesn't matter, top not included
            include_top=False,
            channels=config["data"]["color_channels"],
            get_architecture=efficientnet_standard.efficientnet_b0,
            representation_dim=1280,
        )
        return backbone

    # otherwise, continue with torchvision

    net = _get_net(config)  # e.g. resnet

    if "resnet" in config["model"]["architecture"]:
        # c_out = channels out
        # output dim of e.g. resnet, once the classification layer is removed (below)
        c_out = list(net.children())[-1].in_features

    elif "efficientnet" in config["model"]["architecture"]:
        # sequential is -1, then 1 is linear (0 being dropout)
        c_out = list(net.children())[-1][1].in_features

    # i.e. remove the last layer (aka the classification layer) as default-defined
    # for resnet, is linear. for effnet, is sequential([dropout, linear]). Same thing.
    # net now ends with adaptivepool in both cases
    net = torch.nn.Sequential(*list(net.children())[:-1])

    # Change first layer for color channels B/W images
    n_c = config["data"]["color_channels"]
    if n_c != 3:
        logging.warning("Adapting network for greyscale images, may not match Zoobot")
        # c_out, k, s, p = net[0].out_channels, net[0].kernel_size, net[0].stride, net[0].padding
        # net[0] = nn.Conv2d(n_c, c_out, kernel_size=k, stride=s, padding=p, bias=False)
        net[0] = nn.Conv2d(n_c, 64, kernel_size=7, stride=2, padding=2, bias=False)

    if config["model"]["downscale"]:
        logging.warning("Adapting network with downscaling, may not match Zoobot")
        net[0] = nn.Conv2d(n_c, 64, kernel_size=3, stride=1, padding=1, bias=False)

    features = config["model"]["features"]  # e.g. 512
    # TODO need to check if effnet includes these conv/avg pool layers already - zoobot does

    # if 'resnet' in config["model"]["architecture"]:
    if features != c_out:
        logging.warning(
            "Requested num. features {} does not equal backbone output {} - adding 1x1 conv layer with {} features".format(
                features, c_out, features
            )
        )
        backbone = nn.Sequential(
            *list(net.children())[:-1],  # also remove adaptive pool (both cases)
            nn.Conv2d(c_out, features, 1),  # another conv layer, to `features` channels with 1x1 kernel
            nn.AdaptiveAvgPool2d(1),  # put adaptive pool back
        )
    else:
        backbone = net

    # Define layers for finetuning
    encoder = Encoder(backbone)
    encoder.dim = features
    encoder.finetuning_layers = nn.ModuleList(list(backbone.children())[4:-2])

    return encoder


def _get_transformer(config):
    return _vision_transformer(
        patch_size=config["model"]["patch_size"],
        num_layers=config["model"]["num_layers"],
        num_heads=config["model"]["num_heads"],
        hidden_dim=config["model"]["features"],
        mlp_dim=config["model"]["mlp_dim"],
        weights=None,
        progress=True,
    )


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
        "efficientnetb0": M.efficientnet_b0,  # not tested, could be v useful re zoobot
    }

    return networks[config["model"]["architecture"]]()

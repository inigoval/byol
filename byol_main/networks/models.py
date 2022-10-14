import logging

import torch
import torch.nn as nn
import torchvision.models as M
from torchvision.models.vision_transformer import _vision_transformer
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


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


# helpers


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {"cls", "mean"}, "pool type must be either cls (cls token) or mean (mean pooling)"

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


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
        c_out = list(net.children())[
            -1
        ].in_features  # output dim of e.g. resnet, once the classification layer is removed (below)
    elif "efficientnet" in config["model"]["architecture"]:
        c_out = list(net.children())[-1][
            1
        ].in_features  # sequential is -1, then 1 is linear (0 being dropout)

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

    return backbone


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

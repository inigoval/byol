import torch
import torchvision.transforms as T
import torchvision
import numpy as np
from torch import nn


class Circle_Crop(torch.nn.Module):
    """
    PyTorch transform to set all values outside largest possible circle that fits inside image to 0.
    """

    def __init__(self):
        super().__init__()

    def forward(self, img):
        """
        Returns an image with all values outside the central circle bounded by image edge masked to 0.

        !!! Support for multiple channels not implemented yet !!!
        """
        H, W, C = img.shape[-1], img.shape[-2], img.shape[-3]
        assert H == W
        x = torch.arange(W, dtype=torch.float).repeat(H, 1)
        x = (x - 74.5) / 74.5
        y = torch.transpose(x, 0, 1)
        r = torch.sqrt(torch.pow(x, 2) + torch.pow(y, 2))
        r = r / torch.max(r)
        r[r < 0.5] = -1
        r[r == 0.5] = -1
        r[r != -1] = 0
        r = torch.pow(r, 2).view(C, H, W)
        assert r.shape == img.shape
        img = torch.mul(r, img)
        return img


class GaussianBlur(object):
    """Blur a single image on CPU"""

    def __init__(self, kernel_size, color_channels=1):
        radius = kernel_size // 2
        kernel_size = radius * 2 + 1
        self.blur_h = nn.Conv2d(
            color_channels,
            color_channels,
            kernel_size=(kernel_size, 1),
            stride=1,
            padding=0,
            bias=False,
            groups=1,
        )
        self.blur_v = nn.Conv2d(
            color_channels,
            color_channels,
            kernel_size=(1, kernel_size),
            stride=1,
            padding=0,
            bias=False,
            groups=1,
        )
        self.k = kernel_size
        self.r = radius
        self.n_c = color_channels

        self.blur = nn.Sequential(nn.ReflectionPad2d(radius), self.blur_h, self.blur_v)

        self.pil_to_tensor = T.ToTensor()
        self.tensor_to_pil = T.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img


class MultiView(nn.Module):
    def __init__(self, config, n_views=1, s=1):
        super().__init__()
        cropsize = config["center_crop_size"]
        # Color jitter parameters are taken from the BYOL paper
        color_jitter = T.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)

        # Define a view
        self.view = T.Compose(
            [
                T.RandomRotation(180),
                T.CenterCrop(cropsize),
                T.RandomResizedCrop(cropsize),
                T.RandomHorizontalFlip(),
                T.RandomApply([color_jitter], p=0.8),
                # GaussianBlur(kernel_size=int(0.1 * cropsize)),
                T.ToTensor(),
            ]
        )
        self.normalize = T.Normalize((0,), (1,))
        self.n_views = n_views

    def __call__(self, x):
        views = []
        for i in np.arange(self.n_views):
            view = self.normalize(self.view(x))
            views.append(view)
        return *views

    def update_normalization(self, mu, sig):
        self.normalize = T.Normalize((mu,), (sig,))


def Identity(crop_size, mu=0, sig=1):
    return T.Compose(
        [
            T.CenterCrop(crop_size),
            T.ToTensor(),
            T.Normalize((mu,), (sig,)),
        ]
    )


class ReduceView(nn.Module):
    def __init__(self, encoder, config):
        super().__init__()

        # Define a view
        cropsize = config["center_crop_size"]
        self.aug = T.Compose(
            [
                T.RandomRotation(180),
                T.CenterCrop(cropsize),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ]
        )

        self.reduce = lambda x: encoder(x)
        self.normalize = T.Normalize((0,), (1,))

    def __call__(self, x):
        x = self.aug(x)
        x = self.reduce(x)
        x = self.normalize(x)
        return x

    def update_normalization(self, mu, sig):
        self.normalize = T.Normalize((mu,), (sig,))

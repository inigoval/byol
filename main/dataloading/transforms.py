import torch
import torchvision.transforms as T
import torchvision
import numpy as np
import astroaugmentations as AA
from torch import nn

from paths import Path_Handler
import cv2


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


class MultiView(nn.Module):
    def __init__(self, config, n_views=2, mu=(0,), sig=(1,)):
        super().__init__()
        self.config = config

        # Define a view
        self.view = self._view()
        self.normalize = T.Normalize(mu, sig)
        self.n_views = n_views

    def __call__(self, x):
        if self.n_views > 1:
            views = []
            for i in np.arange(self.n_views):
                view = self.normalize(self.view(x))
                views.append(view)
            return views
        else:
            view = self.normalize(self.view(x))
            return view

    def _view(self):
        if self.config["dataset"] == "rgz":
            # return _rgz_view(self.config)
            return _rgz_view2()

        elif self.config["dataset"] == "imagenette":
            return _simclr_view(self.config)

        elif self.config["dataset"] == "stl10":
            return _simclr_view(self.config)

        elif self.config["dataset"] == "cifar10":
            return _simclr_view(self.config)

        elif self.config["dataset"] == "gzmnist":
            return _gz_view(self.config)

    def update_normalization(self, mu, sig):
        self.normalize = T.Normalize(mu, sig)


class SimpleView(nn.Module):
    def __init__(self, config, mu=(0,), sig=(1,)):
        super().__init__()
        self.config = config

        augs = []

        if config["data"]["rotate"]:
            augs.append(T.RandomRotation(180))
            augs.append(T.RandomHorizontalFlip())
        augs.append(T.Resize(config["data"]["input_height"]))
        if config["center_crop_size"]:
            augs.append(T.CenterCrop(config["center_crop_size"]))
        augs.append(T.ToTensor())
        self.view = T.Compose(augs)

        self.normalize = T.Normalize(mu, sig)

    def __call__(self, x):
        # Use rotation if training

        x = self.normalize(self.view(x))
        return x

    def update_normalization(self, mu, sig):
        self.normalize = T.Normalize(mu, sig)


class ReduceView(nn.Module):
    def __init__(self, encoder, config, train=True):
        super().__init__()

        augs = []

        if config["data"]["rotate"] and train:
            augs.append(T.RandomRotation(180))
            augs.append(T.RandomHorizontalFlip())

        augs.append(T.Resize(config["data"]["input_height"]))
        augs.append(T.CenterCrop(config["center_crop_size"]))
        augs.append(T.ToTensor())

        self.view = T.Compose(augs)

        self.pre_normalize = T.Normalize(config["data"]["mu"], config["data"]["sig"])
        self.encoder = encoder
        self.reduce = lambda x: self.encoder(x)
        self.normalize = T.Normalize(0, 1)

    def __call__(self, x):
        x = self.view(x)
        x = self.pre_normalize(x)
        x = torch.unsqueeze(x, 0)
        x = self.reduce(x).view(-1, 1, 1)
        x = self.normalize(x)
        return x

    def update_normalization(self, mu, sig):
        self.normalize = T.Normalize(mu, sig)


def _simclr_view(config):
    # Returns a SIMCLR view

    s = config["s"]
    input_height = config["data"]["input_height"]

    color_jitter = T.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)

    # Gaussian blurring, kernel 10% of image size (SimCLR paper)
    blur_kernel = _blur_kernel(input_height)
    blur = SIMCLR_GaussianBlur(blur_kernel, p=0.5, min=0.1, max=2.0)

    # Define a view
    view = T.Compose(
        [
            T.RandomResizedCrop(input_height, scale=(0.08, 1)),
            T.RandomHorizontalFlip(),
            T.RandomApply([color_jitter], p=0.8),
            T.RandomGrayscale(p=0.2),
            blur,
            T.ToTensor(),
        ]
    )

    return view


def _gz_view(config):
    s = config["s"]
    input_height = config["data"]["input_height"]

    # Gaussian blurring, kernel 10% of image size (SimCLR paper)
    p_blur = config["p_blur"]
    blur_kernel = _blur_kernel(input_height)
    blur_sig = config["blur_sig"]
    blur = SIMCLR_GaussianBlur(blur_kernel, p=p_blur, min=blur_sig[0], max=blur_sig[1])

    # Color augs
    color_jitter = T.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    p_grayscale = config["p_grayscale"]

    # Cropping
    random_crop = config["random_crop"]

    # Define a view
    view = T.Compose(
        [
            T.RandomRotation(180),
            T.RandomResizedCrop(input_height, scale=random_crop),
            T.RandomHorizontalFlip(),
            T.RandomApply([color_jitter], p=0.8),
            T.RandomGrayscale(p=p_grayscale),
            blur,
            T.ToTensor(),
        ]
    )

    return view


def _rgz_view(config):
    # Gaussian blurring
    blur_kernel = config["blur_kernel"]
    blur_sig = config["blur_sig"]
    blur = T.GaussianBlur(blur_kernel, sigma=blur_sig)

    # Cropping
    center_crop = config["center_crop_size"]
    random_crop = config["random_crop"]

    # Color jitter
    s = config["s"]
    color_jitter = T.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0)

    # Define a view
    view = T.Compose(
        [
            T.RandomRotation(180),
            T.CenterCrop(center_crop),
            T.RandomResizedCrop(center_crop, scale=random_crop),
            T.RandomHorizontalFlip(),
            T.RandomApply([color_jitter], p=0.8),
            T.RandomApply([blur], p=config["p_blur"]),
            T.ToTensor(),
        ]
    )

    return view


def _rgz_view2():
    paths = Path_Handler()
    path_dict = paths._dict()
    kernel_path = path_dict["main"] / "dataloading" / "FIRST_kernel.npy"

    view = AA.AstroAugmentations(domain="radio", kernel=np.load(kernel_path), p=0.5)()
    return view


def _blur_kernel(input_height):
    blur_kernel = int(input_height * 0.1)
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    return blur_kernel


class SIMCLR_GaussianBlur:
    """Taken from  https://github.com/PyTorchLightning/lightning-bolts/blob/2415b49a2b405693cd499e09162c89f807abbdc4/pl_bolts/models/self_supervised/simclr/transforms.py#L17"""

    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, p=0.5, min=0.1, max=2.0):

        self.min = min
        self.max = max

        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < self.p:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample

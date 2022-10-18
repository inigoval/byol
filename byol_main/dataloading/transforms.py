import torch
import logging
import numpy as np
import albumentations as A

from astroaugmentations import image_domain
from torch import nn
from albumentations.pytorch import ToTensorV2

from byol_main.paths import Path_Handler
from byol_main.dataloading.utils import _img_transform


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
        self.mu, self.sig = mu, sig

        # Define a view and convert into callable transform
        # transforms = self._view()
        self.view = _img_transform(self._view())

        # Define callable normalization transform
        self.normalize = _img_transform([A.Normalize(mean=mu, std=sig), ToTensorV2()])

        # Define number of views to output
        self.n_views = n_views

    def __call__(self, x):
        # Convert to numpy for albumentations
        x = np.array(x)  # Apply view transform n_view times
        if self.n_views > 1:
            views = []
            for _ in np.arange(self.n_views):
                view = self.view(x)
                view = self.normalize(view)
                views.append(view)
            return views
        else:
            view = self.view(x)
            view = self.normalize(view)
            return view

    def _view(self):
        if self.config["dataset"] == "rgz":
            # return _rgz_view(self.config)
            return _rgz_view(self.config)

        elif self.config["dataset"] in ["imagenette", "stl10", "cifar10"]:
            return _simclr_view(self.config)

        elif self.config["dataset"] == "gzmnist":
            return _gzmnist_view(self.config)

        # TODO could get ugly
        elif self.config["dataset"] in ["gz2", "decals_dr5", "legs", "rings", "legs_and_rings"]:
            return _gz2_view(self.config)  # now badly named TODO

        elif self.config["dataset"] == "mixed":
            # return _zoobot_default_view(self.config)
            return _gz2_view(self.config)  # now badly named TODO

        else:
            raise ValueError(self.config["dataset"])

    def update_normalization(self, mu, sig):
        self.normalize = _img_transform([A.Normalize(mean=mu, std=sig), ToTensorV2()])


class MAEView(nn.Module):
    def __init__(self, config, mu=(0,), sig=(1,)):
        super().__init__()
        self.config = config

        augs = []
        # if config['dataset'] == 'gz2':  # is a tensor, needs to be a PIL to later call T.ToTensor
        #     augs.append(T.ToPILImage())

        if config["data"]["rotate"]:
            augs.append(A.Rotate(limit=180))

        input_height = config["data"]["input_height"]
        augs.append(A.Resize(input_height, input_height))

        if config["augmentations"]["center_crop_size"]:
            crop_size = config["augmentations"]["center_crop_size"]
            augs.append(A.CenterCrop(crop_size, crop_size))
            input_height = crop_size

        augs.append(
            A.RandomResizedCrop(
                input_height, input_height, scale=config["augmentations"]["random_crop_scale"]
            )
        )

        self.view = _img_transform(augs)
        self.n_views = 1
        self.normalize = _img_transform([A.Normalize(mean=mu, std=sig), ToTensorV2()])
        # self.normalize = lambda x: x  # TODO temporarily disable normalisation (see also MultiView)

    def __call__(self, x):
        x = np.array(x)
        x = self.view(x)
        x = self.normalize(x)
        return x

    def update_normalization(self, mu, sig):
        self.normalize = _img_transform([A.Normalize(mean=mu, std=sig), ToTensorV2()])


class SimpleView(nn.Module):
    def __init__(self, config, mu=(0,), sig=(1,)):
        super().__init__()
        self.config = config

        augs = []
        # if config['dataset'] == 'gz2':  # is a tensor, needs to be a PIL to later call T.ToTensor
        #     augs.append(T.ToPILImage())

        # if config["data"]["rotate"]:
        #     augs.append(T.RandomRotation(180))

        input_height = config["data"]["input_height"]
        augs.append(A.Resize(input_height, input_height))

        if config["augmentations"]["center_crop_size"]:
            crop_size = config["augmentations"]["center_crop_size"]
            augs.append(A.CenterCrop(crop_size, crop_size))

        # augs.append(
        #     A.Lambda(
        #         name="expand_channels", image=Expand_Channels(config["data"]["color_channels"]), p=1
        #     )
        # )

        self.view = _img_transform(augs)

        self.normalize = _img_transform([A.Normalize(mean=mu, std=sig), ToTensorV2()])
        # self.normalize = lambda x: x  # TODO temporarily disable normalisation (see also MultiView)

    def __call__(self, x):
        x = np.array(x)
        x = self.view(x)
        x = self.normalize(x)
        return x

    def update_normalization(self, mu, sig):
        self.normalize = _img_transform([A.Normalize(mean=mu, std=sig), ToTensorV2()])


class SupervisedView(nn.Module):
    def __init__(self, config, mu=(0,), sig=(1,)):
        super().__init__()
        self.config = config

        augs = []

        if config["data"]["rotate"]:
            augs.append(T.RandomRotation(180))

        augs.append(T.Resize(config["data"]["input_height"]))

        if config["center_crop_size"]:
            augs.append(T.CenterCrop(config["center_crop_size"]))
        augs.append(T.ToTensor())
        self.view = T.Compose(augs)

        self.normalize = T.Normalize(mu, sig)

    def __call__(self, x):
        # Use rotation if training
        x = self.view(x)
        x = self.normalize(x)
        return x

    def update_normalization(self, mu, sig):
        self.normalize = T.Normalize(mu, sig)


def _simclr_view(config):
    # Returns a SIMCLR view

    s = config["augmentations"]["s"]
    input_height = config["data"]["input_height"]

    # Gaussian blurring, kernel 10% of image size (SimCLR paper)
    blur_kernel = _blur_kernel(input_height)

    # Define a view
    view = [
        A.RandomResizedCrop(input_height, input_height, scale=(0.08, 1)),
        A.HorizontalFlip(p=0.5),
        # T.RandomVerticalFlip(),
        A.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s, p=0.8),
        A.ToGray(p=0.2),
        A.Blur(blur_limit=blur_kernel, p=0.5),
    ]

    return view


def _rgz_view(config):

    paths = Path_Handler()
    path_dict = paths._dict()
    kernel_path = path_dict["main"] / "dataloading" / "FIRST_kernel.npy"
    kernel = np.load(kernel_path)

    cfg_augs = config["augmentations"]
    p = cfg_augs["p_radio"]

    # Cropping
    center_crop = config["augmentations"]["center_crop_size"]
    # random_crop = config["random_crop_scale"]

    radio_augs = [
        #############################
        # Change source perspective #
        #############################
        # Segment image and give different pixels a different brightness shift.
        # Accounts for older/younger sources having different spectral index
        # A.Lambda(
        #     name="Superpixel spectral index change",
        #     image=image_domain.radio.SpectralIndex(
        #         mean=-0.8, std=0.2, super_pixels=True, n_segments=100, seed=None
        #     ),
        #     p=p,
        # ),  # With segmentation
        # # A.Lambda(
        #     name="Brightness perspective distortion",
        #     image=image_domain.BrightnessGradient(limits=(0.0, 1.0)),
        #     p=p,
        # ),  # No noise
        # A.ElasticTransform(  # Elastically transform the source
        #     alpha=1, sigma=100, alpha_affine=25, interpolation=1, border_mode=1, value=0, p=p
        # ),
        A.ShiftScaleRotate(
            shift_limit=0,
            scale_limit=0.1,
            rotate_limit=0,
            interpolation=2,
            border_mode=0,
            value=0,
            p=1,
        ),
        A.Flip(),
        ###########################################################
        # Change properties of noise / imaging artefacts globally #
        ###########################################################
        # A.Lambda(
        #     name="Spectral index change of whole image",
        #     image=image_domain.radio.SpectralIndex(mean=-0.8, std=0.2, seed=None),
        #     p=p,
        # ),  # Across the whole image
        # A.Emboss(
        #     alpha=(0.2, 0.5), strength=(0.2, 0.5), p=p
        # ),  # Quick emulation of incorrect w-kernels # Doesnt force the maxima to 1
        # Brightness
        # A.Lambda(
        #     name="Brightness perspective distortion",
        #     image=image_domain.BrightnessGradient(limits=(0.0, 1), primary_beam=True, noise=0.01),
        #     p=p,
        # ),  # Gaussian Noise and pb brightness scaling
        # Modelling based transforms
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0,
            rotate_limit=180,
            interpolation=2,
            border_mode=0,
            value=0,
            p=p,
        ),
        # A.Lambda(
        #     name="Dirty beam convolution",
        #     image=image_domain.radio.CustomKernelConvolution(
        #         kernel=kernel, rfi_dropout=0.1, psf_radius=None, sidelobe_scaling=1, mode="sum"
        #     ),
        #     p=p,
        # ),  # Add sidelobes
        A.CenterCrop(width=center_crop, height=center_crop, p=1),
    ]

    input_height = config["data"]["input_height"]
    s = cfg_augs["s"]
    vision_augs = [
        A.RandomResizedCrop(input_height, input_height, scale=cfg_augs["random_crop_scale"]),
        # A.Lambda(name="expand_channels", image=Expand_Channels(config["data"]["color_channels"]), p=1),
        A.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s, p=0.8),
    ]

    view = radio_augs + vision_augs
    return view


def _blur_kernel(input_height):
    blur_kernel = int(input_height * 0.1)
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    return blur_kernel


def _train_view(config):
    if config["type"] in ["mae"]:
        return MAEView
    elif config["type"] in ["byol"]:
        return MultiView


class Expand_Channels(torch.nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels

    def forward(self, img):
        H, W = img.shape[-1], img.shape[-2]
        return img.expand(-1, -1, self.n_channels)


def _rgz_view_old(config):

    paths = Path_Handler()
    path_dict = paths._dict()
    kernel_path = path_dict["main"] / "dataloading" / "FIRST_kernel.npy"
    kernel = np.load(kernel_path)

    cfg_augs = config["augmentations"]
    p = cfg_augs["p_radio"]

    # Cropping
    center_crop = config["augmentations"]["center_crop_size"]
    # random_crop = config["random_crop_scale"]

    radio_augs = [
        A.Lambda(
            name="Superpixel spectral index change",
            image=image_domain.radio.SpectralIndex(
                mean=-0.8, std=0.2, super_pixels=True, n_segments=100, seed=None
            ),
            p=p,
        ),  # With segmentation
        A.Lambda(
            name="Brightness perspective distortion",
            image=image_domain.BrightnessGradient(limits=(0.0, 1.0)),
            p=p,
        ),  # No noise
        A.ElasticTransform(  # Elastically transform the source
            alpha=1, sigma=100, alpha_affine=25, interpolation=1, border_mode=1, value=0, p=p
        ),
        A.ShiftScaleRotate(
            shift_limit=0,
            scale_limit=0.1,
            rotate_limit=0,
            interpolation=2,
            border_mode=0,
            value=0,
            p=1,
        ),
        A.Flip(),
        # Change properties of noise / imaging artefacts
        A.Lambda(
            name="Spectral index change of whole image",
            image=image_domain.radio.SpectralIndex(mean=-0.8, std=0.2, seed=None),
            p=p,
        ),  # Across the whole image
        A.Emboss(
            alpha=(0.2, 0.5), strength=(0.2, 0.5), p=p
        ),  # Quick emulation of incorrect w-kernels # Doesnt force the maxima to 1
        A.Lambda(
            name="Dirty beam convlolution",
            image=image_domain.radio.CustomKernelConvolution(
                kernel=kernel, rfi_dropout=0.4, psf_radius=1.3, sidelobe_scaling=1, mode="sum"
            ),
            p=p,
        ),  # Add sidelobes
        A.Lambda(
            name="Brightness perspective distortion",
            image=image_domain.BrightnessGradient(limits=(0.0, 1), primary_beam=True, noise=0.01),
            p=p,
        ),  # Gaussian Noise and pb brightness scaling
        # Modelling based transforms
        A.CenterCrop(width=center_crop, height=center_crop, p=1),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0,
            rotate_limit=180,
            interpolation=2,
            border_mode=0,
            value=0,
            p=p,
        ),
    ]

    input_height = config["data"]["input_height"]
    s = cfg_augs["s"]
    vision_augs = [
        A.RandomResizedCrop(input_height, input_height, scale=cfg_augs["random_crop_scale"]),
        A.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s, p=0.8),
    ]

    view = radio_augs + vision_augs
    return view

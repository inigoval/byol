import torch
import torchvision
import numpy as np
import torchvision.transforms as T
import pytorch_lightning as pl
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.trainer.supporters import CombinedLoader
import torch.utils.data as D
from sklearn.model_selection import train_test_split

from paths import Path_Handler
from utilities import batch_eval


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


def data_splitter_strat(dset, seed=None, split=1, val_frac=0.2, u_cut=False):
    if seed == None:
        seed = np.random.randint(9999999)

    n = len(dset)
    idx = np.arange(n)
    labels = np.array(dset.targets)

    data_dict, idx_dict = {"train_val": dset}, {"train_val": idx}

    # Split into train/val #
    idx_dict["train"], idx_dict["val"] = train_test_split(
        idx_dict["train_val"],
        test_size=val_frac,
        stratify=labels[idx_dict["train_val"]],
        random_state=seed,
    )

    # Split into unlabelled/labelled #
    idx_dict["l"], idx_dict["u"] = train_test_split(
        idx_dict["train"],
        train_size=split,
        stratify=labels[idx_dict["train"]],
        random_state=seed,
    )

    for key, idx in idx_dict.items():
        data_dict[key] = torch.utils.data.Subset(dset, idx)

    return data_dict, idx_dict


def subindex(idx, fraction):
    """Return a ```fraction``` of all given ```idx```"""
    n = len(idx)
    n_sub = int(fraction * n)
    sub_idx, rest_idx = idx[:n_sub], idx[n_sub:]
    return sub_idx, rest_idx


def random_subset(dset, size):
    """Randomly subset a given data-set to a given size"""
    idx = np.arange(size)
    subset_idx = np.random.choice(idx, size=size)
    return D.Subset(dset, subset_idx)


def size_cut(threshold, dset, inplace=True):
    """Cut the RGZ DR1 dataset based on angular size"""
    length = len(dset)
    idx = np.argwhere(dset.sizes > threshold)
    if inplace == True:
        dset.data = dset.data[idx, ...]
        dset.names = dset.names[idx, ...]
        dset.rgzid = dset.rgzid[idx, ...]
        dset.sizes = dset.sizes[idx, ...]
        dset.mbflg = dset.mbflg[idx, ...]
    else:
        return idx
    print(f"RGZ dataset cut from {length} to {len(dset)} samples")


def mb_cut(dset, inplace=True):
    length = len(dset)
    idx = np.argwhere(dset.mbflg == 0)
    if inplace == True:
        dset.data = dset.data[idx, ...]
        dset.names = dset.names[idx, ...]
        dset.rgzid = dset.rgzid[idx, ...]
        dset.sizes = dset.sizes[idx, ...]
        dset.mbflg = dset.mbflg[idx, ...]
    else:
        return idx
    print(f"RGZ dataset cut from {length} to {len(dset)} samples")


def dset2tens(dset):
    """Return a tuple (x, y) containing the entire input dataset (carefuwith large datasets)"""
    return next(iter(DataLoader(dset, int(len(dset)))))


def compute_mu_sig(dset, batch_size=0):
    """Compute mean and standard variance of a dataset (careful with large datasets)"""
    if batch_size == True:
        # Load samples in batches
        loader = DataLoader(dset, batch_size)
        n, x_sum = 0, 0
        for x, _ in loader:
            n += torch.numel(x)
            x_sum += torch.sum(x)
        # Calculate average

    else:
        x, _ = dset2tens(dset)
        return torch.mean(x), torch.std(x)

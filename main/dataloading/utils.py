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


def rgz_cut(rgz_dset, threshold, mb_cut=True):
    """Cut rgz data-set based on angular size and whether data-point is contained in MiraBest"""

    n_i = len(rgz_dset)
    idx_bool = rgz_dset.sizes > threshold

    n_cut = n_i - np.count_nonzero(idx_bool)
    print(f"Removing {n_cut} samples below angular size threshold.")

    if mb_cut:
        idx_bool *= rgz_dset.mbflg == 0

        # Print number of MB samples removed
        n_mb = np.count_nonzero(rgz_dset.mbflg == 1)
        print(f"Removed {n_mb} MiraBest samples from RGZ")

    idx = np.argwhere(idx_bool)

    subset = D.Subset(rgz_dset, idx)
    print(f"RGZ dataset cut from {n_i} to {len(subset)} samples")
    return subset


def dset2tens(dset):
    """Return a tuple (x, y) containing the entire input dataset (carefuwith large datasets)"""
    return next(iter(DataLoader(dset, int(len(dset)))))


def compute_mu_sig(dset, batch_size=0):
    """Compute mean and standard variance of a dataset (use batching with large datasets)"""
    if batch_size:
        # Load samples in batches
        n_dset = len(dset)
        loader = DataLoader(dset, batch_size)

        # Calculate mean
        mean = 0
        for x, _ in loader:
            x = x[0]
            weight = x.shape[0] / n_dset
            mean += weight * torch.mean(x)

        # Calculate std
        D_sq = 0
        for x, _ in loader:
            x = x[0]
            D_sq += torch.sum((x - mean) ** 2)
        std = (D_sq / (n_dset * x.shape[-1] * x.shape[-2])) ** 0.5

        print(f"mean: {mean}, std: {std}")
        return mean.item(), std.item()

    else:
        x, _ = dset2tens(dset)
        return torch.mean(x).item(), torch.std(x).item()

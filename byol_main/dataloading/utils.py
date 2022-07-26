import torch
import torchvision
import numpy as np
import torchvision.transforms as T
import pytorch_lightning as pl
import torch.utils.data as D

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.trainer.supporters import CombinedLoader

from byol_main.paths import Path_Handler
from byol_main.utilities import batch_eval


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
    idx_dict["labelled"], idx_dict["u"] = train_test_split(
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


def size_cut(dset, threshold, inplace=True):
    """Cut the RGZ DR1 dataset based on angular size"""
    length = len(dset)
    idx = np.argwhere(dset.sizes > threshold)
    if inplace is True:
        dset.data = dset.data[idx, ...]
        dset.names = dset.names[idx, ...]
        dset.rgzid = dset.rgzid[idx, ...]
        dset.sizes = dset.sizes[idx, ...]
        dset.mbflg = dset.mbflg[idx, ...]
    else:
        return idx
    print(f"RGZ dataset cut from {length} to {len(dset)} samples")


def remove_duplicates(dset):
    """Remove duplicate samples from a dataset"""


def mb_cut(dset, inplace=True):
    length = len(dset)
    idx = np.argwhere(dset.mbflg == 0)
    if inplace is True:
        dset.data = dset.data[idx, ...]
        dset.names = dset.names[idx, ...]
        dset.rgzid = dset.rgzid[idx, ...]
        dset.sizes = dset.sizes[idx, ...]
        dset.mbflg = dset.mbflg[idx, ...]
    else:
        return idx
    print(f"RGZ dataset cut from {length} to {len(dset)} samples")


def rgz_cut(rgz_dset, threshold, mb_cut=True, remove_duplicates=False):
    """Cut rgz data-set based on angular size and whether data-point is contained in MiraBest"""

    n = len(rgz_dset)
    idx_bool = np.ones(n, dtype=bool)
    idx = np.arange(n)

    if remove_duplicates:
        idx_bool = np.zeros(n, dtype=bool)
        _, idx_unique = np.unique(rgz_dset.data, axis=0, return_index=True)
        idx_bool[idx_unique] = True

        print(f"Removed {n - np.count_nonzero(idx_bool)} duplicate samples")
        n = np.count_nonzero(idx_bool)

    idx_bool *= rgz_dset.sizes > threshold
    print(f"Removing {n - np.count_nonzero(idx_bool)} samples below angular size threshold.")
    n = np.count_nonzero(idx_bool)

    if mb_cut:
        idx_bool *= rgz_dset.mbflg == 0

        # Print number of MB samples removed
        print(f"Removed {n - np.count_nonzero(idx_bool)} MiraBest samples from RGZ")

    idx = np.argwhere(idx_bool)

    subset = D.Subset(rgz_dset, idx)
    print(f"RGZ dataset cut from {n} to {len(subset)} samples")
    return subset


def dset2tens(dset):
    """Return a tuple (x, y) containing the entire input dataset (carefuwith large datasets)"""
    return next(iter(DataLoader(dset, int(len(dset)))))


def compute_mu_sig(dset, batch_size=0):
    """Compute mean and standard variance of a dataset (use batching with large datasets)"""
    print("Computing mean and std of dataset")
    if batch_size:
        # Load samples in batches
        n_dset = len(dset)
        loader = DataLoader(dset, batch_size)
        x, _ = next(iter(loader))
        n, c, h, w = x.shape

        # Calculate mean
        mu = 0
        print("Computing mean")
        for x, _ in tqdm(loader):
            x = x
            weight = x.shape[0] / n_dset
            mu += weight * torch.mean(x)

        # Calculate std
        D_sq = 0
        print("Computing std")
        for x, _ in tqdm(loader):
            D_sq += torch.sum((x - mu) ** 2)
        std = (D_sq / (n_dset * h * w)) ** 0.5

        print(f"mean: {mu}, std: {std}")
        return mu, std.item()

    else:
        x, _ = dset2tens(dset)
        return torch.mean(x).item(), torch.std(x).item()


def compute_mu_sig_images(dset, batch_size=0):
    """Compute mean and standard variance of a dataset (use batching with large datasets)"""
    if batch_size:
        # Load samples in batches
        n_dset = len(dset)
        loader = DataLoader(dset, batch_size)
        n_channels = next(iter(loader))[0].shape[1]

        # Calculate mean
        mu = torch.zeros(n_channels)
        for x, _ in loader:
            print(x.shape, _.shape)
            for c in np.arange(n_channels):
                x_c = x[:, c, :, :]
                weight = x.shape[0] / n_dset
                mu[c] += weight * torch.mean(x_c).item()

        # Calculate std
        D_sq = torch.zeros(n_channels)
        for x, _ in loader:
            for c in np.arange(n_channels):
                x_c = x[:, c, :, :]
                D_sq += torch.sum((x_c - mu[c]) ** 2)
        sig = (D_sq / (n_dset * x.shape[-1] * x.shape[-2])) ** 0.5

        mu, sig = tuple(mu.tolist()), tuple(sig.tolist())
        print(f"mu: {mu}, std: {sig}")
        return mu, sig

    else:
        x, _ = dset2tens(dset)
        n_channels = x.shape[1]
        mu, sig = [], []
        for c in np.arange(n_channels):
            x_c = x[:, c, :, :]
            mu.append(torch.mean(x).item())
            sig.append(torch.std(x).item())
        return tuple(mu), tuple(sig)


def compute_mu_sig_features(dset, batch_size=0):
    """Compute mean and standard variance of a dataset (use batching with large datasets)"""
    print("Computing mean and std of dataset")
    if batch_size:
        # Load samples in batches
        n_dset = len(dset)
        loader = DataLoader(dset, batch_size)
        x, _ = next(iter(loader))
        n, c, h, w = x.shape

        # Calculate mean
        mu = 0
        print("Computing mean")
        for x, _ in tqdm(loader):
            x = x
            weight = x.shape[0] / n_dset
            mu += weight * torch.mean(x)

        # Calculate std
        D_sq = 0
        print("Computing std")
        for x, _ in tqdm(loader):
            D_sq += torch.sum((x - mu) ** 2)
        std = (D_sq / (n_dset * h * w)) ** 0.5

        print(f"mean: {mu}, std: {std}")
        return mu, std.item()

    else:
        x, _ = dset2tens(dset)
        return torch.mean(x).item(), torch.std(x).item()


def _get_imagenet_norms():
    return {"mu": (0.485, 0.456, 0.406), "sig": (0.229, 0.224, 0.225)}


def _get_cifar10_norms():
    return {"mu": (0.4914, 0.4822, 0.4465), "sig": (0.2023, 0.1994, 0.2010)}

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
    print("Computing mean and std of dataset")
    if batch_size:
        # Load samples in batches
        n_dset = len(dset)
        loader = DataLoader(dset, batch_size)
        x, _ = next(iter(loader))
        n, c, h, w = x.shape

        # Calculate mean
        mean = 0
        print("Computing mean")
        for x, _ in tqdm(loader):
            x = x
            weight = x.shape[0] / n_dset
            mean += weight * torch.mean(x)

        # Calculate std
        D_sq = 0
        print("Computing std")
        for x, _ in tqdm(loader):
            D_sq += torch.sum((x - mean) ** 2)
        std = (D_sq / (n_dset * h * w)) ** 0.5

        print(f"mean: {mean}, std: {std}")
        return mean, std.item()

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
        mean = torch.zeros(n_channels)
        for x, _ in loader:
            for c in np.arange(n_channels):
                x_c = x[:, c, :, :]
                weight = x.shape[0] / n_dset
                mean[c] += weight * torch.mean(x_c).item()

        # Calculate std
        D_sq = torch.zeros(n_channels)
        for x, _ in loader:
            for c in np.arange(n_channels):
                x_c = x[:, c, :, :]
                D_sq += torch.sum((x_c - mean[c]) ** 2)
        sig = (D_sq / (n_dset * x.shape[-1] * x.shape[-2])) ** 0.5

        mean, sig = tuple(mean.tolist()), tuple(sig.tolist())
        print(f"mean: {mean}, std: {sig}")
        return mean, sig

    else:
        x, _ = dset2tens(dset)
        n_channels = x.shape[1]
        mean, sig = [], []
        for c in n_channels:
            x_c = x[:, c, :, :]
            mean.append(torch.mean(x).item())
            sig.append(torch.std(x).item())
        return tuple(mean), tuple(sig)


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
        mean = 0
        print("Computing mean")
        for x, _ in tqdm(loader):
            x = x
            weight = x.shape[0] / n_dset
            mean += weight * torch.mean(x)

        # Calculate std
        D_sq = 0
        print("Computing std")
        for x, _ in tqdm(loader):
            D_sq += torch.sum((x - mean) ** 2)
        std = (D_sq / (n_dset * h * w)) ** 0.5

        print(f"mean: {mean}, std: {std}")
        return mean, std.item()

    else:
        x, _ = dset2tens(dset)
        return torch.mean(x).item(), torch.std(x).item()


def _get_imagenet_norms():
    return {"mean": (0.485, 0.456, 0.406), "sig": (0.229, 0.224, 0.225)}

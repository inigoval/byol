from byol import BYOL
from dataloading.datamodules.rgz import RGZ108k
import umap
from dataloading.transforms import SimpleView
from paths import Path_Handler
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as T
import logging

from astroaugmentations.datasets.MiraBest_F import MBFRFull, MiraBest_F
from utilities import embed_dataset

paths = Path_Handler()
path_dict = paths._dict()

size = (24, 24)

## BYOL
## MB
if False:
    byol = BYOL.load_from_checkpoint("byol.ckpt")
    byol.eval()
    config = byol.config

    center_crop = config["augmentations"]["center_crop_size"]
    mu, sig = config["data"]["mu"], config["data"]["sig"]

    transform = T.Compose(
        [
            T.CenterCrop(center_crop),
            T.ToTensor(),
            T.Normalize((0.008008896,), (0.05303395,)),
        ]
    )

    data = RGZ108k(path_dict["rgz"], train=True, transform=transform, download=True)
    data, y = embed_dataset(byol.encoder, data)

    # data = MBFRFull(path_dict["rgz"], train=True, transform=view, download=True, aug_type="torchvision")
    train_data = RGZ20k(path_dict["rgz"], train=True, transform=view, download=True)
    data = MiraBest_F(
        path_dict["rgz"], train=True, transform=view, download=True, aug_type="torchvision"
    )
    data, y = next(iter(DataLoader(data, int(len(data)))))
    train_data, _ = next(iter(DataLoader(train_data, int(len(train_data)))))
    data = byol.backbone(data).squeeze().cpu().detach().numpy()
    train_data = byol.backbone(train_data).squeeze().cpu().detach().numpy()

    # fr1_list = [0, 1, 2, 3, 4, 5, 6, 7]
    # exclude_list = [8, 9]

    # targets = np.array(self.targets)
    # exclude = np.array(exclude_list).reshape(1, -1)
    # exclude_mask = ~(targets.reshape(-1, 1) == exclude).any(axis=1)
    # fr1 = np.array(fr1_list).reshape(1, -1)
    # fr2 = np.array(fr2_list).reshape(1, -1)
    # fr1_mask = (targets.reshape(-1, 1) == fr1).any(axis=1)
    # fr2_mask = (targets.reshape(-1, 1) == fr2).any(axis=1)
    # targets[fr1_mask] = 0  # set all FRI to Class~0
    # targets[fr2_mask] = 1  # set all FRII to Class~1
    # self.data = self.data[exclude_mask]
    # self.targets = targets[exclude_mask].tolist()
    y = np.array(y)

    y_fr1 = np.array([0, 1, 2, 3, 4]).reshape(1, -1)
    y_fr2 = np.array([5, 6, 7]).reshape(1, -1)
    y_hybrid = np.array([8, 9]).reshape(1, -1)

    fr1_mask = (y.reshape(-1, 1) == y_fr1).any(axis=1)
    y[fr1_mask] = 0  # set all FRII to Class~1

    fr2_mask = (y.reshape(-1, 1) == y_fr2).any(axis=1)
    y[fr2_mask] = 1  # set all FRII to Class~1

    hybrid_mask = (y.reshape(-1, 1) == y_hybrid).any(axis=1)
    y[hybrid_mask] = 2  # set all hybrid to Class~2

    reducer = umap.UMAP(n_neighbors=20)
    reducer.fit(train_data)
    embedding = reducer.transform(data)
    # plt.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap="Spectral", s=2, vmin=15, vmax=35)
    # plt.figure(figsize=(12, 12))
    fig, ax = plt.subplots()
    fig.set_size_inches(24, 24)
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap="Spectral", s=100)
    plt.gca().set_aspect("equal", "datalim")
    # plt.axes(visible=False)
    # plt.colorbar(boundaries=np.arange(0, 25) - 0.5).set_ticks(np.arange(0, 25))
    cbar = fig.colorbar(scatter)
    cbar.ax.tick_params(labelsize=25)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.savefig("byol_umap_mb.png", bbox_inches="tight", pad_inches=0.5)
    plt.close("all")

## RGZ
if True:
    logging.info("Umapping RGZ data-set with arcsecond colorbar")
    byol = BYOL.load_from_checkpoint("byol.ckpt")
    byol.eval()
    config = byol.config

    center_crop = config["augmentations"]["center_crop_size"]
    mu, sig = config["data"]["mu"], config["data"]["sig"]

    transform = T.Compose(
        [
            T.CenterCrop(center_crop),
            T.ToTensor(),
            T.Normalize((0.008008896,), (0.05303395,)),
        ]
    )

    logging.info("Embedding RGZ108k...")
    data = RGZ108k(path_dict["rgz"], train=True, transform=transform, download=True)
    data, y = embed_dataset(byol.encoder, data)
    logging.info("RGZ108k embedded")

    logging.info("UMAP fitting...")
    reducer = umap.UMAP(n_neighbors=25)
    reducer.fit(data)
    loggin.info("UMAP fitted")

    logging.info("Transforming data...")
    embedding = reducer.transform(data)

    print("Plotting figure...")
    fig, ax = plt.subplots()
    fig.set_size_inches(10 / 3, 3)
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap="Spectral", s=7, vmin=15, vmax=40)
    plt.gca().set_aspect("equal", "datalim")
    # plt.axes(visible=False)
    # plt.colorbar(boundaries=np.arange(0, 25) - 0.5).set_ticks(np.arange(0, 25))
    cbar = fig.colorbar(scatter)
    # cbar.ax.tick_params(labelsize=25)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.savefig("byol_umap_rgz.png", bbox_inches="tight", pad_inches=0.5)


## Supervised
# sup = Supervised.load_from_checkpoint("supervised.ckpt")
# sup.eval()
# config = sup.config

# view = SimpleView(config)
# data = RGZ20k(path_dict["rgz"], train=True, transform=view, download=True)
# data, y = next(iter(DataLoader(data, int(len(data)))))
# data = sup.backbone(data).squeeze().cpu().detach().numpy()

# reducer = umap.UMAP()
# reducer.fit(data)
# embedding = reducer.transform(data)
# plt.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap="Spectral", s=3, vmin=15, vmax=25)
# plt.gca().set_aspect("equal", "datalim")
# # plt.colorbar(boundaries=np.arange(0, 25) - 0.5).set_ticks(np.arange(0, 25))
# plt.colorbar()
# plt.savefig("sup_umap.png")

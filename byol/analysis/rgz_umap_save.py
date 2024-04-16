import pandas as pd
import torch
import torchvision.transforms as T
import logging
import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
import torchmetrics as tm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pickle
import joblib
from PIL import Image

from torch import Tensor
from torch.utils.data import DataLoader
from einops import rearrange
from tqdm import tqdm
from collections import Counter
from sklearn.decomposition import PCA
from umap import UMAP
from byol.models import BYOL
from byol.resnet import _get_resnet

from byol.paths import Path_Handler
from byol.dataloading.datamodules.rgz import RGZ108k
from byol.utilities import embed_dataset


class RGZ(RGZ108k):
    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        remove_duplicates=True,
        cut_threshold=0.0,
        mb_cut=False,
    ):
        super().__init__(
            root,
            train,
            transform,
            target_transform,
            download,
            remove_duplicates=remove_duplicates,
            cut_threshold=cut_threshold,
            mb_cut=mb_cut,
        )

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            image (array): Image
        """

        img = self.data[index]
        las = self.sizes[index]
        mbf = self.mbflg[index]
        rgz = self.rgzid[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = np.reshape(img, (150, 150))
        img = Image.fromarray(img, mode="L")

        if self.transform is not None:
            img = self.transform(img)

        return img, index


def set_grads(module, value: bool):
    for params in module.parameters():
        params.requires_grad = value


PCA_COMPONENTS = 200
UMAP_N_NEIGHBOURS = 150
UMAP_MIN_DIST = 0.01


# Get paths
print("Getting paths...")
path_handler = Path_Handler()
paths = path_handler._dict()

# Load in RGZ csv
print("Loading in RGZ data...")
csv_path = paths["rgz"] / "rgz_data.csv"
df_meta = pd.read_csv(csv_path)
df_meta = df_meta[["rgz_name", "radio.ra", "radio.dec", "radio.outermost_level"]]
df_meta = df_meta.rename(
    columns={"radio.dec": "dec", "radio.ra": "ra", "radio.outermost_level": "sigma"}
)

# # Add in extra columns
# df_rgz.insert(len(df_rgz.columns), "umap_x", None)
# df_rgz.insert(len(df_rgz.columns), "umap_y", None)

# feat_cols = [f"feat_{i}" for i in range(512)]
# feat_vals = [[None for i in range(512)] for j in range(len(df_rgz))]
# df_rgz = pd.concat([df_rgz, pd.DataFrame(feat_vals, columns=feat_cols)], axis=1)

# pca_cols = [f"pca_{i}" for i in range(PCA_COMPONENTS)]
# pca_vals = [[None for i in range(PCA_COMPONENTS)] for j in range(len(df_rgz))]
# df_rgz = pd.concat([df_rgz, pd.DataFrame(pca_vals, columns=pca_cols)], axis=1)

# print(df_rgz.columns)

# Load transform
transform = T.Compose(
    [
        T.CenterCrop(70),
        T.ToTensor(),
        T.Normalize((0.008008896,), (0.05303395,)),
    ]
)
# Load in RGZ data
d_rgz = RGZ(
    paths["rgz"], train=True, transform=transform, remove_duplicates=True, cut_threshold=25, mb_cut=True
)


# Prepare hashmap for umap values
# y = {id: {"umap_x": None, "umap_y": None} for id in d_rgz.rgzid}

# Load model
model_path = paths["main"] / "analysis" / "byol.ckpt"
checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
# print(checkpoint.keys())
# print(checkpoint["hyper_parameters"])
# print(checkpoint["encoder"])
model = BYOL.load_from_checkpoint(model_path)
model.eval()
encoder = model.encoder.cuda()
encoder.eval()

print("Encoding dataset...")
train_loader = DataLoader(d_rgz, 500, shuffle=False)
device = next(encoder.parameters()).device
feature_bank = []

for data in tqdm(train_loader):
    # Load data and move to correct device
    x, y = data

    x_enc = encoder(x.to(device))

    feature_bank.append(x_enc.squeeze().detach().cpu())

# Save full feature bank for validation epoch
X_rgz = torch.cat(feature_bank)


X_rgz = X_rgz.squeeze().detach().cpu().numpy()
print(X_rgz.shape)

print("Fitting PCA...")
pca = PCA(n_components=PCA_COMPONENTS)
pca.fit(X_rgz)
with open("pca.pkl", "wb") as f:
    pickle.dump(pca, f)
joblib.dump(pca, "pca.joblib")


print("Explained variance ratio: ", pca.explained_variance_ratio_.sum())

print("Fitting UMAP...")
reducer = UMAP(n_components=2, n_neighbors=UMAP_N_NEIGHBOURS, min_dist=UMAP_MIN_DIST, metric="cosine")
reducer.fit(pca.transform(X_rgz))
with open("reducer.pkl", "wb") as f:
    pickle.dump(reducer, f)
joblib.dump(reducer, "reducer.joblib")


def reduce(x):
    x = encoder(x.cuda()).squeeze().detach().cpu().numpy()
    x = pca.transform(x)
    x = reducer.transform(x)
    return x


print("Transforming data...")
embedding = reducer.transform(pca.transform((X_rgz)))

print("Plotting figure...")

plt.rc("font", family="Liberation Mono")
alpha = 0.6
marker_size = 4
fig_size = (14, 14)
seed = 42


fig, ax = plt.subplots()
fig.set_size_inches(fig_size)
scatter = ax.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=d_rgz.sizes,
    cmap="Spectral",
    s=marker_size,
    vmin=25,
    vmax=50,
    alpha=alpha,
)
plt.gca().set_aspect("equal", "datalim")
# plt.axes(visible=False)
# plt.colorbar(boundaries=np.arange(0, 25) - 0.5).set_ticks(np.arange(0, 25))
cbar = fig.colorbar(scatter)
# cbar.set_label("source extension (arcsec)", rotation=270, size=25, labelpad=100)
cbar.ax.tick_params(labelsize=25)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
fig.savefig("byol_pca+umap_rgz.png", bbox_inches="tight", pad_inches=0.5)


# for (x, _), id in tqdm(
#     zip(DataLoader(d_rgz, batch_size=1, shuffle=False), d_rgz.rgzid), total=len(d_rgz)
# ):
#     # y[id] = model.encoder(x).squeeze().detach().cpu().numpy()
#     x = encoder(x.cuda()).squeeze().detach().cpu().numpy().reshape((1, -1))
#     # Save encoder features
#     df_rgz.loc[df_rgz["rgz_name"] == id, feat_cols] = np.squeeze(x).tolist()

#     # Get and save pca features
#     x_pca = pca.transform(x.reshape((1, -1))).reshape((1, -1))
#     df_rgz.loc[df_rgz["rgz_name"] == id, pca_cols] = np.squeeze(x_pca).tolist()
#     # Save pca features

#     x_emb = np.squeeze(reducer.transform(x_pca))

#     print(x_emb.shape)

#     df_rgz.loc[df_rgz["rgz_name"] == id, "umap_x"] = x_emb[0]
#     df_rgz.loc[df_rgz["rgz_name"] == id, "umap_y"] = x_emb[1]


feat_cols = [f"feat_{i}" for i in range(512)]
pca_cols = [f"pca_{i}" for i in range(PCA_COMPONENTS)]
umap_cols = ["umap_x", "umap_y"]
float_cols = feat_cols + pca_cols + umap_cols

count = 0
df = pd.DataFrame(columns=["rgz_name"] + feat_cols + pca_cols + ["umap_x", "umap_y"])

for X, idx in tqdm(DataLoader(d_rgz, batch_size=500, shuffle=False)):
    names = d_rgz.rgzid[idx].tolist()

    B = X.shape[0]

    X_emb = encoder(X.cuda()).squeeze().detach().cpu().numpy().reshape((B, -1))

    # Get and save pca features
    X_pca = pca.transform(X_emb).reshape((B, -1))
    # Save pca features

    X_umap = np.squeeze(reducer.transform(X_pca))

    df_tmp = pd.DataFrame(
        data=np.concatenate([names, X_emb, X_pca, X_umap], axis=1),
        columns=["rgz_name"] + float_cols,
    )

    df = pd.concat([df, df_tmp], axis=0)

    # print(f"{len(df)} rows")

# Set dtypes
df = df.astype({float_col: "float32" for float_col in float_cols})
df = df.astype({"rgz_name": "string"})

print(f"\n {len(df)} rows")
print(f"columns: {df.columns} \n")

print("Combining dataframe with meta data")
df = pd.merge(df, df_meta, on=["rgz_name"], how="inner")

print(f"\n {len(df)} rows")
print(f"columns: {df.columns} \n")
print(df.head())

# Save dataframe
print("Saving dataframe...")
df.to_parquet(paths["rgz"] / "rgz_umap.parquet", index=False)
print("Done!")

from byol import BYOL
import torch
from dataloading.datamodules.rgz import RGZ108k
from astroaugmentations.datasets.MiraBest_F import MBFRFull, MBHybrid, MBFRConfident, MBFRUncertain
import umap
from einops import pack

from paths import Path_Handler
import matplotlib.pyplot as plt
import torchvision.transforms as T

from utilities import embed_dataset


plt.rc("font", family="Liberation Mono")
alpha = 0.6
marker_size = 4
fig_size = (14, 14)
seed = 42

print("Imports complete")

paths = Path_Handler()
path_dict = paths._dict()

## RGZ
print("Loading pre-trained model...")
byol = BYOL.load_from_checkpoint("byol.ckpt")
byol.eval()
encoder = byol.encoder.cuda()
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

print("Embedding RGZ108k...")
rgz = RGZ108k(path_dict["rgz"], train=True, transform=transform, download=True)
mb = MBFRFull(path_dict["rgz"], train=True, transform=transform, download=True, aug_type="torchvision")

X_rgz, y_rgz = embed_dataset(encoder, rgz)
X_mb, y_mb = embed_dataset(encoder, mb)
print("Data embedded")

#############################
### Fit UMAP and plot RGZ ###
#############################

print("UMAP fitting...")
reducer = umap.UMAP(n_neighbors=25, random_state=seed)
reducer.fit(X_rgz)
print("UMAP fitted")

print("Transforming data...")
embedding = reducer.transform(X_rgz)

print("Plotting figure...")
fig, ax = plt.subplots()
fig.set_size_inches(fig_size)
scatter = ax.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=y_rgz,
    cmap="Spectral",
    s=marker_size,
    vmin=15,
    vmax=40,
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
fig.savefig("byol_umap_rgz.png", bbox_inches="tight", pad_inches=0.5)


#########################################
### Fit UMAP and plot RGZ vs MiraBest ###
#########################################

print("UMAP fitting...")
X_comb, _ = pack([X_rgz, X_mb], "* c")
# y_comb, _ = pack([torch.full_like(y_rgz, 2), y_mb], "*")
reducer = umap.UMAP(n_neighbors=25, random_state=seed)
reducer.fit(X_comb)
print("UMAP fitted")

print("Transforming data...")
rgz_emb = reducer.transform(X_rgz)
mb_emb = reducer.transform(X_mb)

print("Plotting figure...")
fig, ax = plt.subplots()
fig.set_size_inches(fig_size)

ax.scatter(rgz_emb[:, 0], rgz_emb[:, 1], label="RGZ DR1", s=marker_size, alpha=alpha)
ax.scatter(mb_emb[:, 0], mb_emb[:, 1], label="MiraBest", s=marker_size, alpha=alpha)
plt.gca().set_aspect("equal", "datalim")
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.legend(fontsize=25, markerscale=4)
fig.savefig("byol_umap_mbrgz.png", bbox_inches="tight", pad_inches=0.5)

print("finished")
## Fit UMAP with both and show rgz vs mirabest


#########################################
### Fit UMAP and plot RGZ vs MiraBest Conf vs Unce ###
#########################################

conf = MBFRConfident(
    path_dict["rgz"], train=True, transform=transform, download=True, aug_type="torchvision"
)
unc = MBFRUncertain(
    path_dict["rgz"], train=True, transform=transform, download=True, aug_type="torchvision"
)

X_conf, y_conf = embed_dataset(encoder, conf)
X_unc, y_unc = embed_dataset(encoder, unc)

print("UMAP fitting...")
X_comb, _ = pack([X_rgz, X_mb], "* c")
# y_comb, _ = pack([torch.full_like(y_rgz, 2), y_mb], "*")
reducer = umap.UMAP(n_neighbors=25, random_state=seed)
reducer.fit(X_comb)
print("UMAP fitted")

print("Transforming data...")
rgz_emb = reducer.transform(X_rgz)
conf_emb = reducer.transform(X_conf)
unc_emb = reducer.transform(X_unc)

print("Plotting figure...")
fig, ax = plt.subplots()
fig.set_size_inches(fig_size)

ax.scatter(rgz_emb[:, 0], rgz_emb[:, 1], label="RGZ DR1", s=marker_size, alpha=alpha)
ax.scatter(conf_emb[:, 0], conf_emb[:, 1], label="MiraBest Confident", s=marker_size, alpha=alpha)
ax.scatter(unc_emb[:, 0], unc_emb[:, 1], label="MiraBest Uncertain", s=marker_size, alpha=alpha)
plt.gca().set_aspect("equal", "datalim")
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.legend(fontsize=25, markerscale=4)
fig.savefig("byol_umap_mbrgz_split.png", bbox_inches="tight", pad_inches=0.5)

print("finished")
## Fit UMAP with both and show rgz vs mirabest

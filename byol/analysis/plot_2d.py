import torch
import matplotlib
import matplotlib.pyplot as plt
import torchvision.transforms as T
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from einops import pack
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import ConcatDataset
from umap import UMAP
from pathlib import Path

from byol.utilities import embed_dataset
from byol.datamodules import RGZ108k
from byol.datamodules import MBFRFull, MBHybrid, MBFRConfident, MBFRUncertain
from byol.models import BYOL
from byol.paths import Path_Handler
from byol.resnet import ResNet, BasicBlock


# font_dirs = ["/share/nas/inigovs/fonts/"]
# fonts = font_manager.findSystemFonts(fontpaths=font_dirs, fontext="ttf")

# for font_path in fonts:
#     font_name = font_manager.get_font(font_path).family_name
#     print(font_name)

# prop = font_manager.FontProperties(fname="/usr/share/fonts/liberation/LiberationMono-Regular.ttf")

# font_manager.fontManager.addfont("libmono.ttf")

# print(font_manager.findSystemFonts(fontpaths=None, fontext="ttf"))

font_dirs = ["/home/inigovs/.fonts/"]
font_files = matplotlib.font_manager.findSystemFonts(fontpaths=font_dirs)

font_manager = matplotlib.font_manager.FontManager()

# font_manager.ttflist.extend(font_list)
for font in font_files:
    print(font)
    font_manager.addfont(font)

# Print the available font families
# font_manager.addfont("/share/nas/inigovs/fonts/LiberationMono-Regular.ttf")
font_manager.addfont("/home/inigovs/.fonts/LiberationMono-Regular.ttf")

# Get the available font families
print("Available font families:")
font_families = matplotlib.font_manager.findSystemFonts()
available_families = set([matplotlib.font_manager.get_font(font).family_name for font in font_families])

for family in available_families:
    print(family)

# plt.rcParams["font.family"] = "LiberationMono-Regular.ttf"
# fpath = Path(mpl.get_data_path(), "/share/nas/inigovs/fonts/LiberationMono-Regular.ttf")
plt.rc("font", family="Liberation Mono")
plt.rcParams["font.family"] = "Liberation Mono"


alpha = 0.6
marker_size = 0.1
fig_size = (10 / 3, 3)
seed = 69
fontsize = 9
marker = "o"


print("Imports complete")

paths = Path_Handler()._dict()

# HPARAMS
PCA_COMPONENTS = 200
UMAP_N_NEIGHBOURS = 75
UMAP_MIN_DIST = 0.01
METRIC = "cosine"

## RGZ
print("----------------------------")
print("Loading pre-trained model...")
print("----------------------------")

encoder = ResNet(block=BasicBlock, layers=[2, 2, 2, 2], n_c=1, downscale=True, features=512).cuda()
encoder.load_state_dict(torch.load("encoder.pt"))
encoder.eval()

# byol = BYOL.load_from_checkpoint(paths["main"] / "byol.ckpt")
# byol.eval()
# encoder = byol.encoder.cuda()
# encoder.eval()
# config = byol.config
# # center_crop = config["augmentations"]["center_crop_size"]
# mu, sig = config["data"]["mu"], config["data"]["sig"]


class Reducer:
    def __init__(self, encoder):
        self.encoder = encoder
        self.pca = PCA(n_components=PCA_COMPONENTS, random_state=seed)
        self.umap = UMAP(
            n_components=2,
            n_neighbors=UMAP_N_NEIGHBOURS,
            min_dist=UMAP_MIN_DIST,
            metric="cosine",
            random_state=seed,
        )

    def embed_dataset(self, data, batch_size=400):
        train_loader = DataLoader(data, batch_size, shuffle=False)
        device = next(encoder.parameters()).device
        feature_bank = []
        target_bank = []
        for data in tqdm(train_loader):
            # Load data and move to correct device
            x, y = data

            x_enc = encoder(x.to(device))

            feature_bank.append(x_enc.squeeze().detach().cpu())
            target_bank.append(y.to(device).detach().cpu())

        # Save full feature bank for validation epoch
        features = torch.cat(feature_bank)
        targets = torch.cat(target_bank)

        return features, targets

    def fit(self, data):
        print("Fitting reducer")
        features, targets = self.embed_dataset(data)
        self.features = features
        self.targets = targets

        self.pca.fit(self.features)
        self.umap.fit(self.pca.transform(self.features))

    def transform(self, data):
        # x = self.encoder(x.cuda()).squeeze().detach().cpu().numpy()
        x, _ = self.embed_dataset(data)
        x = self.pca.transform(x)
        x = self.umap.transform(x)
        return x

    def transform_pca(self, data):
        x, _ = self.embed_dataset(data)
        x = self.pca.transform(x)
        return x


#############################
### Fit UMAP and plot RGZ ###
#############################
print("------------------------------------------------")
print("Plotting RGZ data set with UMAP")
print("------------------------------------------------")

transform = T.Compose(
    [
        T.CenterCrop(70),
        T.ToTensor(),
        T.Normalize((0.008008896,), (0.05303395,)),
    ]
)

rgz = RGZ108k(
    paths["rgz"],
    train=True,
    transform=transform,
    download=True,
    remove_duplicates=False,
    cut_threshold=25,
    mb_cut=True,
)
mb = MBFRFull(paths["rgz"], train=True, transform=transform, download=True, aug_type="torchvision")


reducer = Reducer(encoder)
reducer.fit(rgz)
X_umap = reducer.transform(rgz)

#############
# df = pd.read_parquet("embedded_rgz.parquet")
# features = df[[f"feat_{i}" for i in range(512)]].values
# pca = PCA(n_components=PCA_COMPONENTS)
# pca.fit(features)

# reducer = UMAP(n_components=2, n_neighbors=UMAP_N_NEIGHBOURS, min_dist=UMAP_MIN_DIST, metric=METRIC)
# reducer.fit(pca.transform(features))
# reduce = lambda x: reducer.transform(pca.transform(x))

# X_umap = reducer.transform(pca.transform(features))
#############

print("Plotting figure...")
print(matplotlib.rcParams["font.family"])

fig, ax = plt.subplots()
fig.set_size_inches(fig_size)
scatter = ax.scatter(
    X_umap[:, 0],
    X_umap[:, 1],
    c=reducer.targets,
    cmap="Spectral",
    s=marker_size,
    marker=marker,
    vmin=25,
    vmax=100,
    alpha=alpha,
)
plt.gca().set_aspect("equal", "datalim")
# plt.axes(visible=False)
# plt.colorbar(boundaries=np.arange(0, 25) - 0.5).set_ticks(np.arange(0, 25))
cbar = fig.colorbar(scatter)
# cbar.set_label("source extension (arcsec)", rotation=270, size=25, labelpad=100)
cbar.ax.tick_params(labelsize=fontsize)
# ax.set_xlabel("umap x", fontsize=fontsize, fontproperties=prop)
ax.set_xlabel("umap x", fontsize=fontsize)
ax.set_ylabel("umap y", fontsize=fontsize)
# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)
fig.savefig("byol_umap_rgz.png", bbox_inches="tight", pad_inches=0.05, dpi=600)

#########################################
# Generate interpolations of embedding ##
#########################################
print("------------------------------------------------")
print("Interpolating latent space")
print("------------------------------------------------")

# # Get extreme co-ordinates
# idx_max = np.argmax(embedding[:, 0] * embedding[:, 1])
# idx_min = np.argmin(embedding[:, 0] * embedding[:, 1])

# # Generate x and y values for interpolation
# x_max, y_max = embedding[idx_max]
# x_min, y_min = embedding[idx_min]

# x_max = np.argmax(embedding[:, 0])
# x_min = np.argmin(embedding[:, 0])
# x_mean = np.mean(embedding[:, 0])

# y_max = np.argmax(embedding[:, 1])
# y_min = np.argmin(embedding[:, 1])
# y_mean = np.mean(embedding[:, 1])

# X = np.linspace(x_min, x_max, N_IMAGES)
# Y = np.linspace(y_min, y_max, N_IMAGES)

# # XY = np.stack([X, Y], axis=1)

# # Fit kNN
# nbrs = NearestNeighbors(n_neighbors=5, algorithm="ball_tree").fit(embedding)

# # Get nearest neighbours at each (x,y)
# # dist, idx = nbrs.kneighbors(XY, n_neighbours=1)

# interpolated_imgs = []
# interpolated_dist = []
# for i in range(len(X)):
#     # Get distance and index of nearest neighbour
#     dist, idx = nbrs.kneighbors(np.array([X[i], y_mean]).reshape(1, -1), n_neighbors=1)

#     # Add image to list
#     interpolated_imgs.append(rgz[idx])
#     interpolated_dist.append(dist)

# # Plot interpolation images
# fig, ax = plt.subplots(1, N_IMAGES)
# fig.set_size_inches(N_IMAGES, 1)
# for i in range(N_IMAGES):
#     ax[i].imshow(interpolated_imgs[i][0][0], cmap="hot")
#     ax[i].axis("off")


N_IMAGES = 5


def calculate_interpolations(start_point, end_point, num_interpolations):
    # Calculate the Euclidean distance between the start and end points
    distance = np.linalg.norm(end_point - start_point)

    # Determine the step size based on the number of interpolations
    step_size = distance / num_interpolations

    # Initialize an array to store the interpolated points
    interpolated_points = []

    # Calculate the interpolated points for each step
    for i in range(num_interpolations + 1):
        # Interpolate each dimension separately
        interpolated_point = start_point + (i * step_size * (end_point - start_point) / distance)
        interpolated_points.append(interpolated_point)

    return interpolated_points


# rgz = RGZ108k(paths["rgz"], train=True, transform=transform, download=True, remove_duplicates=False, cut_threshold=25, mb_cut=True)

for n in range(16):
    idx = np.random.choice(len(rgz), 2)
    X_pca = reducer.transform_pca(rgz).reshape((-1, PCA_COMPONENTS))

    x1, x2 = X_pca[idx, :]

    interpolations = calculate_interpolations(x1, x2, N_IMAGES)
    nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(X_pca)

    # Get nearest neighbours at each (x,y)
    dist, idx = nbrs.kneighbors(interpolations, n_neighbors=1)

    interpolated_imgs = []
    for j in np.squeeze(idx):
        interpolated_imgs.append(rgz[j][0].squeeze())

    # Plot interpolation images
    # fig.set_size_inches(N_IMAGES * 4, 4)
    # nrows = 2
    # ncols = int(N_IMAGES // nrows)
    # fig, ax = plt.subplots(nrows=nrows, ncols=ncols)

    fig, ax = plt.subplots(1, N_IMAGES)
    fig.set_size_inches(fig_size)

    # center_crop = T.CenterCrop(100)

    for i in range(N_IMAGES):
        img = interpolated_imgs[i]
        ax[i].imshow(interpolated_imgs[i], cmap="hot")
        ax[i].axis("off")

    # for i in range(nrows):
    #     for j in range(ncols):
    #         n_img = j + i * ncols
    #         img = interpolated_imgs[n_img]
    #         img = center_crop(torch.tensor(img)).detach().numpy()
    #         ax[i, j].imshow(img, cmap="hot")
    #         ax[i, j].axis("off")

    fig.tight_layout()
    fig.savefig(f"interpolations/{n}.png", bbox_inches="tight", pad_inches=0.05, dpi=600)
    plt.close(fig)


# interpolated_imgs = []
# interpolated_dist = []
# for i in range(len(X)):
#     # Get distance and index of nearest neighbour
#     dist, idx = nbrs.kneighbors(np.array([X[i], y_mean]).reshape(1, -1), n_neighbors=1)

#     # Add image to list
#     interpolated_imgs.append(rgz[fonts])
#     interpolated_dist.append(dist)

# # Plot interpolation images
# fig, ax = plt.subplots(1, N_IMAGES)
# fig.set_size_inches(N_IMAGES, 1)
# for i in range(N_IMAGES):
#     ax[i].imshow(interpolated_imgs[i][0][0], cmap="hot")
#     ax[i].axis("off")


#########################################
### Fit UMAP and plot RGZ vs MiraBest ###
#########################################

print("------------------------------------------------")
print("Plotting MiraBest vs RGZ data in embedding space")
print("------------------------------------------------")
reducer = Reducer(encoder)
reducer.fit(ConcatDataset([rgz, mb]))

rgz_umap = reducer.transform(rgz)
mb_umap = reducer.transform(mb)

print("Plotting figure...")
fig, ax = plt.subplots()
fig.set_size_inches(fig_size)

ax.scatter(rgz_umap[:, 0], rgz_umap[:, 1], label="RGZ DR1", marker=marker, s=marker_size, alpha=alpha)
ax.scatter(mb_umap[:, 0], mb_umap[:, 1], label="MiraBest", marker=marker, s=marker_size, alpha=alpha)
plt.gca().set_aspect("equal", "datalim")
# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)
ax.set_xlabel("umap x", fontsize=fontsize)
ax.set_ylabel("umap y", fontsize=fontsize)
ax.legend(fontsize=fontsize, markerscale=10)
fig.tight_layout()
fig.savefig("byol_umap_mbrgz.png", bbox_inches="tight", pad_inches=0.05, dpi=600)

print("finished")
## Fit UMAP with both and show rgz vs mirabest


#########################################
### Fit UMAP and plot RGZ vs MiraBest Conf vs Unce ###
#########################################

# conf = MBFRConfident(
#     paths["rgz"], train=True, transform=transform, download=True, aug_type="torchvision"
# )

# unc = MBFRUncertain(
#     paths["rgz"], train=True, transform=transform, download=True, aug_type="torchvision"
# )

# X_conf, y_conf = embed_dataset(encoder, conf)
# X_unc, y_unc = embed_dataset(encoder, unc)

# print("UMAP fitting...")
# X_comb, _ = pack([X_rgz, X_mb], "* c")
# # y_comb, _ = pack([torch.full_like(y_rgz, 2), y_mb], "*")
# reducer = umap.UMAP(n_neighbors=25, random_state=seed)
# reducer.fit(X_comb)
# print("UMAP fitted")

# print("Transforming data...")
# rgz_emb = reducer.transform(X_rgz)
# conf_emb = reducer.transform(X_conf)
# unc_emb = reducer.transform(X_unc)

# print("Plotting figure...")
# fig, ax = plt.subplots()
# fig.set_size_inches(fig_size)

# ax.scatter(rgz_emb[:, 0], rgz_emb[:, 1], label="RGZ DR1", s=marker_size, alpha=alpha)
# ax.scatter(conf_emb[:, 0], conf_emb[:, 1], label="MiraBest Confident", s=marker_size, alpha=alpha)
# ax.scatter(unc_emb[:, 0], unc_emb[:, 1], label="MiraBest Uncertain", s=marker_size, alpha=alpha)
# plt.gca().set_aspect("equal", "datalim")
# # ax.get_xaxis().set_visible(False)
# # ax.get_yaxis().set_visible(False)
# ax.legend(fontsize=25, markerscale=4)
# fig.savefig("byol_umap_mbrgz_split.png", bbox_inches="tight", pad_inches=0.5)

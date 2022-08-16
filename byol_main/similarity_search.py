from byol import BYOL
from supervised import Supervised
from dataloading.datasets import RGZ20k
from dataloading.transforms import SimpleView
from paths import Path_Handler
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import ImageGrid
import torch.utils.data as D
import torch.nn.functional as F
import string

from astroaugmentations.datasets.MiraBest_F import MBHybrid, MiraBest_F, MBFRFull
from utilities import embed, fig2img
from dataloading.utils import rgz_cut, remove_duplicates


def similarity_search(
    feature: torch.Tensor,
    feature_bank: torch.Tensor,
    n: int = 8,
    knn_t: float = 0.1,
    leave_first_out=False,
) -> torch.Tensor:
    """Code modified from https://github.com/lightly-ai/lightly/blob/master/lightly/utils/benchmarking.py

    Run kNN predictions on features based on a feature bank
    This method is commonly used to monitor performance of self-supervised
    learning methods.
    The default parameters are the ones
    used in https://arxiv.org/pdf/1805.01978v1.pdf.
    Args:
        feature:
            Tensor of shape [N, D] for which you want predictions
        feature_bank:
            Tensor of a database of features used for kNN, of shape [D, N] where N is len(l datamodule)
        target_bank:
            Labels for the features in our feature_bank, of shape ()
        num_classes:
            Number of classes (e.g. `10` for CIFAR-10)
        knn_k:
            Number of k neighbors used for kNN
        knn_t:
            Temperature parameter to reweights similarities for kNN
    Returns:
        A tensor containing the kNN predictions
    Examples:
        >>> images, targets, _ = batch
        >>> feature = backbone(images).squeeze()
        >>> # we recommend to normalize the features
        >>> feature = F.normalize(feature, dim=1)
        >>> pred_labels = knn_predict(
        >>>     feature,
        >>>     feature_bank,
        >>> )
    """

    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    # (B, D) matrix. mult (D, N) gives (B, N) (as feature dim got summed over to got cos sim)
    sim_matrix = torch.mm(feature, feature_bank)

    # [B, K]    if leave_first_out is True:
    if leave_first_out:
        sim_weight, sim_idx = sim_matrix.topk(k=n + 1, dim=-1)
        sim_weight, sim_idx = sim_weight[:, 1:], sim_idx[:, 1:]
    elif leave_first_out is False:
        sim_weight, sim_idx = sim_matrix.topk(k=n, dim=-1)

    # sim_weight, sim_idx = sim_matrix.topk(k=n, dim=-1)
    # this will be slow if feature_bank is large (e.g. 100k datapoints)
    sim_weight = sim_weight.squeeze()

    # we do a reweighting of the similarities
    # sim_weight = (sim_weight / knn_t).exp().squeeze().type_as(feature_bank)
    # sim_weight = (sim_weight / knn_t).exp()

    return sim_weight, sim_idx


def _label(y):
    if y == -1:
        return "hybrid"
    elif y == 0:
        return "FRI"
    elif y == 1:
        return "FRII"
    else:
        print(f"invalid label {y}")


paths = Path_Handler()
path_dict = paths._dict()

grid_length = 3
dataset = "rgz"
cat_hybrids = False


def _dataset(transform, train=True, tensor=False):
    if dataset == "rgz":
        data = RGZ20k(path_dict["rgz"], transform=transform, train=train)
        data = rgz_cut(data, 0, remove_duplicates=True)

    elif dataset == "mb":
        data = MBFRFull(path_dict["rgz"], train=train, transform=transform, aug_type="torchvision")

    if cat_hybrids:
        hybrids = MBHybrid(path_dict["rgz"], train=True, transform=view, aug_type="torchvision")
        hybrids.targets = torch.full_like(torch.Tensor(hybrids.targets), -1).long().tolist()
        data = D.ConcatDataset([data, hybrids])

    if tensor:
        data, _ = next(iter(DataLoader(data, int(len(data)))))

    return data


## BYOL
byol = BYOL.load_from_checkpoint("byol.ckpt")
byol.eval()
config = byol.config
view = SimpleView(config)
view.update_normalization(config["data"]["mu"], config["data"]["sig"])
config["center_crop_size"] = 88
less_zoom_view = SimpleView(config)
less_zoom_view.update_normalization(config["data"]["mu"], config["data"]["sig"])

# MB data
data = _dataset(view, train=True, tensor=True)
embedding = F.normalize(byol.backbone(data).squeeze(), dim=1)

data = _dataset(less_zoom_view, train=True, tensor=False)

hybrids = MBHybrid(path_dict["rgz"], train=False, transform=view, aug_type="torchvision")
hybrid_loader = DataLoader(hybrids, batch_size=1)

hybrids = MBHybrid(path_dict["rgz"], train=False, transform=less_zoom_view, aug_type="torchvision")
hybrids.targets = torch.full_like(torch.Tensor(hybrids.targets), -1).long().tolist()


for i, (x, _) in enumerate(hybrid_loader):
    hybrid_embedding = F.normalize(byol.backbone(x).view(1, -1))
    img = x.squeeze()

    topk_weights, topk_idx = similarity_search(
        hybrid_embedding,
        embedding.t(),
        n=grid_length**2 - 1,
        leave_first_out=False,
    )
    # topk_imgs = [data.data[i].squeeze().cpu().detach().numpy() for i in topk_idx.squeeze()]
    # topk_labels = [data.targets[i].item() for i in topk_idx.squeeze()]
    topk_imgs = [data.__getitem__(i)[0].squeeze().cpu().detach().numpy() for i in topk_idx.squeeze()]
    topk_labels = [data.__getitem__(i)[1] for i in topk_idx.squeeze()]

    fig = plt.figure(figsize=(13.0, 13.0))
    fig.subplots_adjust(0, 0, 1, 1)
    grid = ImageGrid(fig, 111, nrows_ncols=(grid_length, grid_length), axes_pad=0)

    x_plot, y = hybrids.__getitem__(i)

    imgs = [x_plot.squeeze().cpu().detach().numpy()] + topk_imgs
    labels = [y] + topk_labels
    img_weights = [1] + topk_weights.tolist()
    letters = list(string.ascii_uppercase)

    for ax, im, label, weight, letter in zip(grid, imgs, labels, img_weights, letters):
        ax.axis("off")
        text = f"{letter}, S = {weight:.2f}"
        if dataset == "mb":
            text = _label(label) + text

        ax.text(1, 70, text, fontsize=23, color="yellow")

        # contours
        threshold = 1
        ax.contour(np.where(im > threshold, 1, 0), cmap="cool", alpha=0.1)

        ax.imshow(im, cmap="hot")

    plt.axis("off")
    pil_img = fig2img(fig)
    plt.savefig(f"{dataset} hybrid{i}.png")
    plt.close(fig)


# ## SUpervised
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

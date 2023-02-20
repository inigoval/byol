from byol import BYOL
import torch
from dataloading.datamodules.rgz import RGZ108k
from astroaugmentations.datasets.MiraBest_F import MBHybrid
import string
import numpy as np

from torch.utils.data import DataLoader
import torch.utils.data as D
from paths import Path_Handler
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torch.nn.functional as F
from mpl_toolkits.axes_grid1 import ImageGrid
from dataloading.utils import rgz_cut

from utilities import embed_dataset, fig2img


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
    The default parameters are the ones used in https://arxiv.org/pdf/1805.01978v1.pdf.
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
    feature = feature.to(feature_bank.device)
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


plt.rc("font", family="Liberation Mono")
alpha = 0.6
marker_size = 4
fig_size = (14, 14)
grid_length = 3

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
rgz = RGZ108k(path_dict["rgz"], shuffle=False, train=True, transform=transform, download=True)
rgz = rgz_cut(rgz, 0, mb_cut=True, remove_duplicates=True)
rgz_loader = DataLoader(rgz, batch_size=1)

X_rgz, y_rgz = embed_dataset(encoder, rgz)
X_rgz = F.normalize(X_rgz)
print("Data embedded")


for (x, _) in rgz_loader:
    hybrid_embedding = F.normalize(encoder(x.cuda()).view(1, -1))
    img = x.squeeze()

    topk_weights, topk_idx = similarity_search(
        hybrid_embedding,
        X_rgz.t(),
        n=grid_length**2 - 1,
        leave_first_out=False,
    )
    # topk_imgs = [data.data[i].squeeze().cpu().detach().numpy() for i in topk_idx.squeeze()]
    # topk_labels = [data.targets[i].item() for i in topk_idx.squeeze()]
    topk_imgs = [rgz.__getitem__(i)[0].squeeze().cpu().detach().numpy() for i in topk_idx.squeeze()]
    topk_labels = [rgz.__getitem__(i)[1] for i in topk_idx.squeeze()]

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

        ax.text(1, 66, text, fontsize=23, color="yellow")

        # contours
        threshold = 1
        ax.contour(np.where(im > threshold, 1, 0), cmap="cool", alpha=0.1)

        ax.imshow(im, cmap="hot")

    plt.axis("off")
    pil_img = fig2img(fig)
    plt.savefig(f"sim_search_hybrid{i}.png")
    plt.close(fig)

print("Finished")

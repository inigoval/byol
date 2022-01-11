import yaml
import torch.nn as nn
import torch.nn.functional as F
from torch import logsumexp
import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import wandb

from torch.optim import Optimizer
from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid
from torch.utils.data import DataLoader

from paths import Path_Handler


# Define paths
paths = Path_Handler()
path_dict = paths._dict()


def entropy(p, eps=0.0000001, loss=False):
    """
    Calculate the entropy of a binary classification prediction given a probability for either of the two classes.

    Keyword arguments:
    eps -- small additive factor to avoid log(0)
    loss -- boolean value determines whether to return detached value for inference (False) or differentiable value for training (True)
    """
    H_i = -torch.log(p + eps) * p
    H = torch.sum(H_i, 1).view(-1)

    if not loss:
        # Clamp to avoid negative values due to eps
        H = torch.clamp(H, min=0)
        return H.detach().cpu().numpy()

    H = torch.mean(H)

    return H


def byol_loss(x, y):
    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)
    return 2 - 2 * (x * y).sum(dim=-1)


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io

    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)
    return img


def flip(labels, p_flip):
    """Flip a number of labels"""
    n_labels = labels.shape[0]
    n_flip = int(p_flip * n_labels)
    if n_flip:
        idx = torch.randint(labels.shape[0], (n_flip))
    else:
        return labels


def batch_eval(fn_dict, dset, batch_size=200):
    """
    Take functions which acts on data x,y and evaluates over the whole dataset in batches, returning a list of results for each calculated metric
    """
    n = len(dset)
    loader = DataLoader(dset, batch_size)

    # Fill the output dictionary with empty lists
    outs = {}
    for key in fn_dict.keys():
        outs[key] = []

    for x, y in loader:
        # Take only weakly augmented sample if passing through unlabelled data
        #        if strong_T:
        #            x = x[0]

        # Append result from each batch to list in outputs dictionary
        for key, fn in fn_dict.items():
            outs[key].append(fn(x, y))

    return outs


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


def log_examples(wandb_logger, dset, n=18):
    save_list = []
    count = 0
    for x, _ in DataLoader(dset, 1):
        if count > n:
            break
        x1, x2 = x
        fig = plt.figure(figsize=(13.0, 13.0))
        grid = ImageGrid(fig, 111, nrows_ncols=(1, 2), axes_pad=0)

        img_list = [x1, x2]
        for ax, im in zip(grid, img_list):
            im = im.reshape((im.shape[-1], im.shape[-2]))
            ax.axis("off")
            ax.imshow(im, cmap="hot")

        plt.axis("off")
        pil_img = fig2img(fig)
        save_list.append(pil_img)
        plt.close(fig)
        count += 1

    wandb_logger.log_image(key=f"image_pairs", images=save_list)


# Yoinked from pl_bolts
class LARSWrapper(object):
    def __init__(self, optimizer, eta=0.02, clip=True, eps=1e-8):
        """
        Wrapper that adds LARS scheduling to any optimizer. This helps stability with huge batch sizes.
        Args:
            optimizer: torch optimizer
            eta: LARS coefficient (trust)
            clip: True to clip LR
            eps: adaptive_lr stability coefficient
        """
        self.optim = optimizer
        self.eta = eta
        self.eps = eps
        self.clip = clip

        # transfer optim methods
        self.state_dict = self.optim.state_dict
        self.load_state_dict = self.optim.load_state_dict
        self.zero_grad = self.optim.zero_grad
        self.add_param_group = self.optim.add_param_group
        self.__setstate__ = self.optim.__setstate__
        self.__getstate__ = self.optim.__getstate__
        self.__repr__ = self.optim.__repr__

    @property
    def __class__(self):
        return Optimizer

    @property
    def state(self):
        return self.optim.state

    @property
    def param_groups(self):
        return self.optim.param_groups 
    @param_groups.setter
    def param_groups(self, value):
        self.optim.param_groups = value

    @torch.no_grad()
    def step(self):
        weight_decays = []

        for group in self.optim.param_groups:
            weight_decay = group.get("weight_decay", 0)
            weight_decays.append(weight_decay)

            # reset weight decay
            group["weight_decay"] = 0

            # update the parameters
            [
                self.update_p(p, group, weight_decay)
                for p in group["params"]
                if p.grad is not None
            ]

        # update the optimizer
        self.optim.step()

        # return weight decay control to optimizer
        for group_idx, group in enumerate(self.optim.param_groups):
            group["weight_decay"] = weight_decays[group_idx]

    def update_p(self, p, group, weight_decay):
        # calculate new norms
        p_norm = torch.norm(p.data)
        g_norm = torch.norm(p.grad.data)

        if p_norm != 0 and g_norm != 0:
            # calculate new lr
            new_lr = (self.eta * p_norm) / (g_norm + p_norm * weight_decay + self.eps)

            # clip lr
            if self.clip:
                new_lr = min(new_lr / group["lr"], 1)

            # update params with clipped lr
            p.grad.data += weight_decay * p.data
            p.grad.data *= new_lr

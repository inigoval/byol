import logging
from typing import Type, Any, Callable, Union, List, Optional

import torch
import matplotlib.pyplot as plt

from torch.optim import Optimizer
from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

from byol.paths import Path_Handler


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


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io

    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)
    return img


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
    # model.eval()


def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True
    # model.train()


def _scheduler(
    opt: Optimizer,
    n_epochs: int,
    decay_type: str = "warmupcosine",
    warmup_epochs: int = 10,
):
    if decay_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
        return scheduler

    elif decay_type == "warmupcosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            opt,
            warmup_epochs,
            max_epochs=n_epochs,
        )
        return scheduler

    else:
        raise ValueError(decay_type)


def _optimizer(
    params,
    type: str = "sgd",
    lr: float = 0.2,
    momentum: float = 0.9,
    weight_decay: float = 0.0000015,
    beta_1: float = 0.9,
    beta_2: float = 0.999,
    **kwargs,
):
    betas = (beta_1, beta_2)

    # for adam, lr is the step size and is modified by exp. moving av. of prev. gradients
    opts = {
        "adam": lambda p: torch.optim.Adam(p, lr=lr, betas=betas, weight_decay=weight_decay),
        "adamw": lambda p: torch.optim.AdamW(p, lr=lr, betas=betas, weight_decay=weight_decay),
        "sgd": lambda p: torch.optim.SGD(p, lr=lr, momentum=momentum, weight_decay=weight_decay),
    }

    if type == "adam" and lr > 0.01:
        logging.warning(f"Learning rate {lr} may be too high for adam")

    opt = opts[type](params)

    return opt

    # Apply LARS wrapper if option is chosen
    # if config["optimizer"]["lars"]:
    #     opt = LARSWrapper(opt, eta=config["optimizer"]["trust_coef"])


def embed_dataset(encoder, data, batch_size=400):
    print("Embedding dataset...")
    train_loader = DataLoader(data, batch_size)
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
    feature_bank = torch.cat(feature_bank)
    target_bank = torch.cat(target_bank)

    return feature_bank, target_bank


def log_examples(wandb_logger, dset, n=18):
    save_list = []
    count = 0
    for x, _ in DataLoader(dset, 1):
        if count > n:
            break
        x1, x2 = x
        C, H, W = x1.shape[-3], x1.shape[-2], x1.shape[-1]

        if C == 1:
            fig = plt.figure(figsize=(13.0, 13.0))
            grid = ImageGrid(fig, 111, nrows_ncols=(1, 2), axes_pad=0)

            img_list = [x1, x2]
            for ax, im in zip(grid, img_list):
                im = im.reshape((H, W, C))
                ax.axis("off")
                ax.imshow(im, cmap="hot")

            plt.axis("off")
            pil_img = fig2img(fig)
            save_list.append(pil_img)
            plt.close(fig)

        else:
            img_list = [x_i.view(C, H, W) for x_i in x]
            img = make_grid(img_list)
            # tens2pil = ToPILImage()
            save_list.append(img)

        count += 1

    wandb_logger.log_image(key="image_pairs", images=save_list)


def compute_encoded_mu_sig(dset, pl_module, batch_size=250):
    """
    Compute the mean and standard deviation of the encoded features of the dataset
    """
    # Load samples in batches
    n_dset = len(dset)
    loader = DataLoader(dset, batch_size)
    channels = pl_module.config["model"]["features"]

    # Calculate mean
    mu = 0
    # print("Computing mean")
    for x, _ in loader:
        x = pl_module.encoder(x.to(pl_module.device)).squeeze()
        weight = x.shape[0] / n_dset
        mu += weight * torch.mean(x)

    # Calculate std
    D_sq = 0
    # print("Computing std")
    for x, _ in loader:
        x = pl_module.encoder(x.to(pl_module.device)).squeeze()
        D_sq += torch.sum(x - mu) ** 2
    std = (D_sq / (n_dset * channels)) ** 0.5

    # print(f"mean: {mu}, std: {std}")
    return mu, std.item()


def check_unique_list(lst):
    return len(set(lst)) == len(lst)


"""
Directly copied from pl_bolts
https://github.com/PyTorchLightning/lightning-bolts/blob/0.3.0/pl_bolts/optimizers/lars_scheduling.py#L62-L81

References:
    - https://github.com/NVIDIA/apex/blob/master/apex/parallel/LARC.py
    - https://arxiv.org/pdf/1708.03888.pdf
    - https://github.com/noahgolmant/pytorch-lars/blob/master/lars.py
"""


class LARSWrapper(object):
    """
    Wrapper that adds LARS scheduling to any optimizer. This helps stability with huge batch sizes.
    """

    def __init__(self, optimizer, eta=0.02, clip=True, eps=1e-8):
        """
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
    def defaults(self):
        return self.optim.defaults

    @defaults.setter
    def defaults(self, defaults):
        self.optim.defaults = defaults

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
    def step(self, closure=None):
        weight_decays = []

        for group in self.optim.param_groups:
            weight_decay = group.get("weight_decay", 0)
            weight_decays.append(weight_decay)

            # reset weight decay
            group["weight_decay"] = 0

            # update the parameters
            [self.update_p(p, group, weight_decay) for p in group["params"] if p.grad is not None]

        # update the optimizer
        self.optim.step(closure=closure)

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

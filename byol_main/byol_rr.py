import torch
import wandb
import numpy as np

from pytorch_lightning.callbacks import Callback
from torch.utils.data import DataLoader
from math import ceil
import matplotlib.pyplot as plt

from byol import BYOL
from evaluation import knn_weight
from lightly.models.utils import update_momentum
from torch.nn.functional import cosine_similarity
import torch.nn.functional as F
from utilities import fig2img


class BYOL_RR(BYOL):
    def __init__(self, config):
        super().__init__(config)

    def training_step(self, batch, batch_idx):
        # Update momentum value
        update_momentum(self.backbone, self.backbone_momentum, m=self.m)
        update_momentum(self.projection_head, self.projection_head_momentum, m=self.m)

        # Load in data
        (x0, x1), _ = batch
        n_batch = x0.shape[0]

        x0 = x0.type_as(self.dummy_param)
        x1 = x1.type_as(self.dummy_param)

        with torch.no_grad():
            r = self.backbone_momentum(x0).squeeze()
            r = F.normalize(r, dim=1)
            knn_k = np.clip(self.config["n_knn"], 1, n_batch - 1)
            sim_weights = knn_weight(r, r.t(), knn_k=knn_k)

            # _, idx_m = torch.topk(sim_weights, int(n_batch * 0.1))

            # find threshold for similarity and create boolean mask for loss
            n_mask = int(n_batch * self.config["r_batch"])
            torch.use_deterministic_algorithms(False)
            sim_max = -torch.kthvalue(-sim_weights, n_mask)[0].item()
            torch.use_deterministic_algorithms(True)

            mask = sim_weights.lt(sim_max)

        p0 = self.project(x0)
        z0 = self.project_momentum(x0)
        p1 = self.project(x1)
        z1 = self.project_momentum(x1)
        # loss = 0.5 * (self.criterion(p0, z1) + self.criterion(p1, z0))

        loss = -0.5 * (cosine_similarity(p0, z1) + cosine_similarity(p1, z0))

        # mask out values with too high similarity
        loss = mask * loss

        loss = loss.mean()

        self.log("train/loss", loss, on_step=False, on_epoch=True)
        return loss


class Count_Masks(Callback):
    """Code adapted from https://github.com/lightly-ai/lightly/blob/master/lightly/utils/benchmarking.py

    Calculates a feature bank for validation"""

    def __init__(self):
        super().__init__()
        self.y_masked = []

    def on_validation_epoch_start(self, trainer, pl_module):
        # epoch = trainer.current_epoch
        # if epoch % 1 == 0:
        config = pl_module.config
        if trainer.current_epoch >= config["model"]["n_epochs"] - pl_module.config["topk"] - 1:
            train_loader = pl_module.trainer.datamodule.train_dataloader()

            for batch in train_loader:
                (x, _), y = batch

                x = x.type_as(pl_module.dummy_param)
                y = y.type_as(pl_module.dummy_param).long()

                n_batch = len(y)
                r = pl_module.backbone_momentum(x.type_as(pl_module.dummy_param)).squeeze()
                r = F.normalize(r, dim=1)
                knn_k = np.clip(pl_module.config["n_knn"], 1, n_batch - 1)
                sim_weights = knn_weight(r, r.t(), knn_k=knn_k)

                # find threshold for similarity and create boolean mask for loss
                n_mask = int(n_batch * pl_module.config["r_batch"])
                torch.use_deterministic_algorithms(False)
                sim_max = -torch.kthvalue(-sim_weights, n_mask)[0].item()
                torch.use_deterministic_algorithms(True)

                mask = sim_weights.lt(sim_max)

                # record angular sizes/labels of masked values
                self.y_masked.append(y[(y * ~mask).nonzero()])

            # y_masked = [[value] for value in y_masked.tolist()]
            # table = wandb.Table(data=y_masked, columns=["arcsecond extension"])
            # self.logger.log({'masked arcsec histogram': wandb.plot.histogram(table, 'arcsecond extension', title:
            # pl_module.logger.log({f"arcsec masks epoch {epoch}": wandb.Histogram(np_histogram=hist)})

    def on_test_epoch_start(self, trainer, pl_module):
        if not pl_module.config["debug"]:

            y_masked = torch.cat(self.y_masked).cpu().numpy()

            fig, ax = plt.subplots(figsize=(13.0, 13.0))
            ax.hist(np.ravel(y_masked), bins=np.arange(13, 45))
            ax.set_xlabel("Similarity")
            # ax.set_ylabel("Count")
            ax.set(yticklabels=[])  # remove the tick labels
            ax.tick_params(left=False)  # remove the ticks
            img = fig2img(fig)
            pl_module.logger.log_image(key="mask histogram", images=[img])


def count_masks(self, pl_module, train_loader):
    # train_loader = pl_module.trainer.datamodule.train_dataloader(

    y_masked = []

    for batch in train_loader:
        (x0, x1), y = batch
        n_batch = len(y)

        r = pl_module.backbone_momentum(x0).squeeze()
        r = F.normalize(r, dim=1)
        sim_weights = knn_weight(r, r.t(), knn_k=pl_module.config["n_knn"])

        # find threshold for similarity and create boolean mask for loss
        n_mask = int(n_batch * pl_module.config["r_batch"])
        torch.use_deterministic_algorithms(False)
        sim_max = -torch.kthvalue(-sim_weights, n_mask)[0].item()
        torch.use_deterministic_algorithms(True)

        mask = sim_weights.lt(sim_max)

        # record angular sizes/labels of masked values
        y_masked.append(y[(y * ~mask).nonzero()])

    y_masked = torch.cat(y_masked)

    return y_masked

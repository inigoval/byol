import wandb
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import numpy as np
import torch
import torchmetrics.functional as tmF
import umap
import umap.plot

from dataloading.utils import dset2tens
from utilities import fig2img


def plot_z_real(X, y, E, epoch, n_z):
    with torch.no_grad():
        fri_idx, frii_idx, hybrid_idx = class_idx(y.numpy())
        embedding = E(X.cuda())[0].view(-1, n_z).cpu().detach().numpy()
        reducer = umap.UMAP()
        umap_embedding = reducer.fit_transform(embedding)
        plt.scatter(
            umap_embedding[fri_idx, 0],
            embedding[fri_idx, 1],
            c="red",
            label="fri",
            s=2,
            marker="x",
        )
        plt.scatter(
            umap_embedding[frii_idx, 0],
            embedding[frii_idx, 1],
            c="blue",
            label="frii",
            s=2,
            marker="x",
        )
        plt.scatter(
            umap_embedding[hybrid_idx, 0],
            embedding[hybrid_idx, 1],
            c="green",
            label="hybrid",
            s=2,
            marker="x",
        )
        plt.legend()
        plt.savefig(EMBEDDING_PATH_REAL + "/embedding_real_{}.pdf".format(epoch))
        plt.close()

    def on_validation_epoch_end(self, trainer, pl_module):
        x = pl_module.generate(self.n_grid ** 2).detach().cpu().numpy()
        img = self.plot(x)
        trainer.logger.experiment.log(
            {
                "generated images": [wandb.Image(img)],
            },
        )

    def plot(self, img_array):
        img_list = list(img_array)
        fig = plt.figure(figsize=(13.0, 13.0))
        grid = ImageGrid(fig, 111, nrows_ncols=(self.n_grid, self.n_grid), axes_pad=0)

        for ax, im in zip(grid, img_list):
            im = im.reshape((150, 150))
            ax.axis("off")
            ax.imshow(im, cmap="hot")
        plt.axis("off")
        pil_img = fig2img(fig)
        plt.close(fig)
        return pil_img


class UmapPlot(pl.Callback):
    def __init__(self):
        super().__init__()

    def on_test_epoch_end(self, trainer, pl_module):
        x_train, y_train = dset2tens(pl_module.trainer.datamodule.data["train"])
        x_train = x_train.type_as(pl_module.m_online.encoder[0].weight)
        y_train = y_train.type_as(pl_module.m_online.encoder[0].weight)
        x_train = x_train.cpu().detach().numpy()

        mapper = umap.UMAP().fit(x_train)

        x_rgz, y_rgz = dset2tens(pl_module.trainer.datamodule.data["rgz"])
        x_rgz = x_rgz.type_as(pl_module.m_online.encoder[0].weight)
        y_rgz = y_rgz.type_as(pl_module.m_online.encoder[0].weight)
        x_rgz = x_rgz.cpu().detach().numpy()

        x_test, y_test = dset2tens(pl_module.trainer.datamodule.data["test"])
        x_test = x_test.type_as(pl_module.m_online.encoder[0].weight)
        y_test = y_test.type_as(pl_module.m_online.encoder[0].weight)
        x_test = x_test.cpu().detach().numpy()

        x_mb, y_mb = dset2tens(pl_module.trainer.datamodule.data["mb"])
        x_mb = x_mb.type_as(pl_module.m_online.encoder[0].weight)
        y_mb = y_mb.type_as(pl_module.m_online.encoder[0].weight)
        x_mb = x_mb.cpu().detach().numpy()

        plot_dict = {}

        for x, y, name in zip(
            [x_u, x_test, x_l],
            [y_u, y_test, y_l],
            ["unlabelled", "test", "labelled"],
        ):

            logits = pl_module.forward(x, logit=True)

            data = torch.cat((logits, y.view(-1, 1)), 1).tolist()

            table = wandb.Table(data=data, columns=["fr1", "fr2", "label"])

            plot_dict[f"test/logits {name}"] = wandb.plot.scatter(
                table, "fr1", "fr2", title=f"{name} logits"
            )

        trainer.logger.experiment.log(plot_dict)

import pandas as pd
import torch
import torchvision.transforms as T
import logging
import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
import torchmetrics as tm
import torch.nn.functional as F

from torch import Tensor
from torch.utils.data import DataLoader
from einops import rearrange
from tqdm import tqdm
from collections import Counter
from byol.finetuning import FineTune

from byol.paths import Path_Handler
from byol.datasets import RGZ108k


def set_grads(module, value: bool):
    for params in module.parameters():
        params.requires_grad = value


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.batchnorm = nn.BatchNorm1d(input_dim)
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.batchnorm(x)
        x = self.linear(x)
        return x


# class FineTune(pl.LightningModule):
#     """
#     Parent class for self-supervised LightningModules to perform linear evaluation with multiple
#     data-sets.
#     """

#     def __init__(
#         self,
#         encoder: nn.Module,
#         dim: int,
#         n_classes,
#         n_epochs=100,
#         n_layers=0,
#         batch_size=1024,
#         lr_decay=0.75,
#         seed=69,
#         **kwargs,
#     ):
#         super().__init__()

#         self.save_hyperparameters(ignore=["encoder", "head"])

#         self.n_layers = n_layers
#         self.batch_size = batch_size
#         self.encoder = encoder
#         self.lr_decay = lr_decay
#         self.n_epochs = n_epochs
#         self.seed = seed
#         self.n_classes = n_classes
#         self.layers = []

#         self.head = LogisticRegression(input_dim=dim, output_dim=n_classes)
#         self.head_type = "linear"

#         # Set finetuning layers for easy access
#         if self.n_layers:
#             layers = self.encoder.finetuning_layers
#             assert self.n_layers <= len(
#                 layers
#             ), f"Network only has {len(layers)} layers, {self.n_layers} specified for finetuning"

#             self.layers = layers[::-1][:n_layers]

#         self.train_acc = tm.Accuracy(
#             task="multiclass", average="micro", threshold=0, num_classes=self.n_classes
#         ).to(self.device)

#         self.val_acc = tm.Accuracy(
#             task="multiclass", average="micro", threshold=0, num_classes=self.n_classes
#         ).to(self.device)

#         self.test_acc = tm.Accuracy(
#             task="multiclass", average="micro", threshold=0, num_classes=self.n_classes
#         ).to(self.device)

#     def forward(self, x: Tensor) -> Tensor:
#         x = self.encoder(x)
#         x = rearrange(x, "b c h w -> b (c h w)")
#         x = self.head(x)
#         return x

#     def on_fit_start(self):
#         # Log size of data-sets #

#         self.train_acc = tm.Accuracy(
#             task="multiclass", average="micro", threshold=0, num_classes=self.n_classes
#         ).to(self.device)
#         self.val_acc = tm.Accuracy(
#             task="multiclass", average="micro", threshold=0, num_classes=self.n_classes
#         ).to(self.device)

#         self.test_acc = nn.ModuleList(
#             [
#                 tm.Accuracy(
#                     task="multiclass", average="micro", threshold=0, num_classes=self.n_classes
#                 ).to(self.device)
#             ]
#             * len(self.trainer.datamodule.data["test"])
#         )

#         logging_params = {f"n_{key}": len(value) for key, value in self.trainer.datamodule.data.items()}
#         self.logger.log_hyperparams(logging_params)

#         # Make sure network that isn't being finetuned is frozen
#         # probably unnecessary but best to be sure
#         set_grads(self.encoder, False)
#         if self.n_layers:
#             for layer in self.layers:
#                 set_grads(layer, True)

#     def training_step(self, batch, batch_idx):
#         # Load data and targets
#         x, y = batch
#         logits = self.forward(x)
#         y_pred = logits.softmax(dim=-1)
#         loss = F.cross_entropy(y_pred, y, label_smoothing=0.1 if self.n_layers else 0)
#         self.log("finetuning/train_loss", loss, on_step=False, on_epoch=True)
#         return loss

#     def validation_step(self, batch, batch_idx, dataloader_idx=0):
#         x, y = batch
#         preds = self.forward(x)
#         self.val_acc(preds, y)
#         self.log("finetuning/val_acc", self.val_acc, on_step=False, on_epoch=True)

#     # def test_step(self, batch, batch_idx, dataloader_idx=0):
#     #     x, y = batch
#     #     preds = self.forward(x)
#     #     self.test_acc(preds, y)
#     #     self.log("finetuning/test_acc", self.test_acc, on_step=False, on_epoch=True)

#     def test_step(self, batch, batch_idx, dataloader_idx=0):
#         x, y = batch
#         name = list(self.trainer.datamodule.data["test"].keys())[dataloader_idx]

#         preds = self.forward(x)
#         self.test_acc[dataloader_idx](preds, y)
#         self.log(
#             f"finetuning/test/{name}_acc", self.test_acc[dataloader_idx], on_step=False, on_epoch=True
#         )

#     def configure_optimizers(self):
#         if not self.n_layers and self.head_type == "linear":
#             # Scale base lr=0.1
#             lr = 0.1 * self.batch_size / 256
#             params = self.head.parameters()
#             return torch.optim.SGD(params, momentum=0.9, lr=lr)
#         else:
#             lr = 0.001 * self.batch_size / 256
#             params = [{"params": self.head.parameters(), "lr": lr}]
#             # layers.reverse()

#             # Append parameters of layers for finetuning along with decayed learning rate
#             for i, layer in enumerate(self.layers):
#                 params.append({"params": layer.parameters(), "lr": lr * (self.lr_decay**i)})

#             # Initialize AdamW optimizer with cosine decay learning rate
#             opt = torch.optim.AdamW(params, weight_decay=0.05, betas=(0.9, 0.999))
#             scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.n_epochs)
#             return [opt], [scheduler]


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


if __name__ == "__main__":
    # Get paths
    print("Getting paths...")
    path_handler = Path_Handler()
    paths = path_handler._dict()

    # Load in RGZ csv
    print("Loading in RGZ data...")
    csv_path = paths["rgz"] / "rgz_data.csv"
    df_rgz = pd.read_csv(csv_path)

    # Add fr_prediction/confidence columns
    df_rgz.insert(len(df_rgz.columns), "fr_prediction", -1)
    df_rgz.insert(len(df_rgz.columns), "fr_vote_fraction", -1)
    df_rgz.insert(len(df_rgz.columns), "fr_entropy", None)
    df_rgz.insert(len(df_rgz.columns), "fr_logit_1", None)
    df_rgz.insert(len(df_rgz.columns), "fr_logit_2", None)

    # Load transform
    transform = T.Compose(
        [
            T.CenterCrop(70),
            T.ToTensor(),
            T.Normalize((0.008008896,), (0.05303395,)),
        ]
    )
    # Load in RGZ data
    d_rgz = RGZ108k(paths["rgz"], train=True, transform=transform, remove_duplicates=False)

    print("Making predictions...")
    model_dir = paths["main"] / "analysis" / "weights" / "25"
    # Make predictions on RGZ data
    y = {id: {"pred": [], "softmax": [], "logits": [], "entropy": []} for id in d_rgz.rgzid}
    for i, model_path in enumerate(model_dir.iterdir()):
        checkpoint = torch.load(model_path)
        print(f"Loading model {i}...")
        model = FineTune.load_from_checkpoint(model_path)
        model.eval()

        print("Performing inference on RGZ data...")
        for (x, _), id in tqdm(
            zip(DataLoader(d_rgz, batch_size=1, shuffle=False), d_rgz.rgzid), total=len(d_rgz)
        ):
            logits = model(x.to(model.device))
            preds = logits.softmax(dim=-1).detach().cpu()
            y_softmax, y_pred = torch.max(preds, dim=-1)
            y_pred += 1

            y[id]["pred"].append(y_pred.item())
            y[id]["softmax"].append(y_softmax.item())
            y[id]["logits"].append(logits.detach().cpu().numpy())
            y[id]["entropy"].append(entropy(preds, loss=False).item())

    print("Aggregating predictions...")

    def aggregate(preds: list) -> tuple:
        counter = Counter(preds)
        agg_pred, count = counter.most_common(1)[0]
        vote_frac = count / len(preds)

        return agg_pred, vote_frac

    for id, value in tqdm(y.items()):
        preds, softmax, logits, entropy = (
            value["pred"],
            value["softmax"],
            value["logits"],
            value["entropy"],
        )
        agg_pred, vote_frac = aggregate(preds)

        agg_logits = np.mean(logits, axis=0).squeeze()

        pred_idx = np.argwhere(np.array(preds) == agg_pred).flatten().tolist()

        df_rgz.loc[df_rgz["rgz_name"] == id, "fr_prediction"] = agg_pred
        df_rgz.loc[df_rgz["rgz_name"] == id, "fr_vote_fraction"] = vote_frac
        df_rgz.loc[df_rgz["rgz_name"] == id, "fr_entropy"] = np.mean(entropy)
        df_rgz.loc[df_rgz["rgz_name"] == id, "fr_logit_1"] = agg_logits[0].item()
        df_rgz.loc[df_rgz["rgz_name"] == id, "fr_logit_2"] = agg_logits[1].item()

    df_rgz.to_csv(paths["rgz"] / "rgz_data_preds_25.csv", index=False)
    print(df_rgz[["fr_prediction", "fr_vote_fraction"]].head())
    print("Done!")

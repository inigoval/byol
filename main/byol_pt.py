import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import pytorch_lightning as pl
import torchmetrics.functional as tmF
import numpy as np
import pl_bolts.models.self_supervised as pl_ssl

from statistics import mean
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pl_bolts.callbacks.byol_updates import BYOLMAWeightUpdate
from pl_bolts.models.self_supervised.byol.models import SiameseArm
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from tqdm import tqdm
from torch.optim import Adam, SGD
from math import cos, pi
from copy import deepcopy
from typing import List, Sequence, Tuple, Union, Any

from utilities import LARSWrapper
from networks.models import MLPHead, LogisticRegression
from networks.models import ResNet18, ResNet50, WideResNet50_2
from utilities import freeze_model
from paths import Path_Handler
from evaluation import knn_predict


def byol_loss(x, y):
    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)
    return 2 - 2 * (x * y).sum(dim=-1)



class byol(pl.LightningModule):
    """PyTorch Lightning implementation of Bootstrap Your Own Latent (BYOL_)_
    Paper authors: Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre H. Richemond, \
    Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Daniel Guo, Mohammad Gheshlaghi Azar, \
    Bilal Piot, Koray Kavukcuoglu, Rémi Munos, Michal Valko.
    Model implemented by:
        - `Annika Brundyn <https://github.com/annikabrundyn>`_
    .. warning:: Work in progress. This implementation is still being verified.
    TODOs:
        - verify on CIFAR-10
        - verify on STL-10
        - pre-train on imagenet
    Example::
        model = BYOL(num_classes=10)
        dm = CIFAR10DataModule(num_workers=0)
        dm.train_transforms = SimCLRTrainDataTransform(32)
        dm.val_transforms = SimCLREvalDataTransform(32)
        trainer = pl.Trainer()
        trainer.fit(model, datamodule=dm)
    Train::
        trainer = Trainer()
        trainer.fit(model)
    CLI command::
        # cifar10
        python byol_module.py --gpus 1
        # imagenet
        python byol_module.py
            --gpus 8
            --dataset imagenet2012
            --data_dir /path/to/imagenet/
            --meta_dir /path/to/folder/with/meta.bin/
            --batch_size 32
    .. _BYOL: https://arxiv.org/pdf/2006.07733.pdf
    """

    def __init__(
        self,
        num_classes,
        learning_rate: float = 0.2,
        weight_decay: float = 1.5e-6,
        input_height: int = 32,
        batch_size: int = 32,
        num_workers: int = 0,
        warmup_epochs: int = 10,
        max_epochs: int = 40,
        base_encoder: Union[str, torch.nn.Module] = "resnet50",
        encoder_out_dim: int = 2048,
        projector_hidden_size: int = 4096,
        projector_out_dim: int = 256,
        **kwargs
    ):
        """
        Args:
            datamodule: The datamodule
            learning_rate: the learning rate
            weight_decay: optimizer weight decay
            input_height: image input height
            batch_size: the batch size
            num_workers: number of workers
            warmup_epochs: num of epochs for scheduler warm up
            max_epochs: max epochs for scheduler
            base_encoder: the base encoder module or resnet name
            encoder_out_dim: output dimension of base_encoder
            projector_hidden_size: hidden layer size of projector MLP
            projector_out_dim: output size of projector MLP
        """
        super().__init__()
        self.save_hyperparameters(ignore="base_encoder")
        self.n_classes = num_classes

        self.encoder_out_dim = encoder_out_dim

        self.online_network = SiameseArm(base_encoder, encoder_out_dim, projector_hidden_size, projector_out_dim)
        self.target_network = deepcopy(self.online_network)
        self.weight_callback = BYOLMAWeightUpdate()

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        # Add callback for user automatically since it's key to BYOL weight update
        self.weight_callback.on_train_batch_end(self.trainer, self, outputs, batch, batch_idx, dataloader_idx)

    def forward(self, x):
        y, _, _ = self.online_network(x)
        return y

    def shared_step(self, batch, batch_idx):
        imgs, y = batch
        img_1, img_2 = imgs[:2]

        # Image 1 to image 2 loss
        y1, z1, h1 = self.online_network(img_1)
        with torch.no_grad():
            y2, z2, h2 = self.target_network(img_2)
        loss_a = -2 * F.cosine_similarity(h1, z2).mean()

        # Image 2 to image 1 loss
        y1, z1, h1 = self.online_network(img_2)
        with torch.no_grad():
            y2, z2, h2 = self.target_network(img_1)
        # L2 normalize
        loss_b = -2 * F.cosine_similarity(h1, z2).mean()

        # Final loss
        total_loss = loss_a + loss_b

        return loss_a, loss_b, total_loss

    def training_step(self, batch, batch_idx):
        loss_a, loss_b, total_loss = self.shared_step(batch, batch_idx)

        # log results
        # self.log_dict({"1_2_loss": loss_a, "2_1_loss": loss_b, "train_loss": total_loss})
        self.log('train/loss', total_loss)

        return total_loss


    def validation_step(self, batch, batch_idx):
        # Load batch
        x, y = batch

        # Extract + normalize features
        feature = self.forward.squeeze()
        feature = F.normalize(feature, dim=1)

        # Load feature bank and labels
        feature_bank = self.feature_bank.type_as(x)
        target_bank = self.target_bank.type_as(y)

        pred_labels = knn_predict(
            feature,
            feature_bank,
            target_bank,
            self.config["data"]["classes"],
        )

        num = len(y)
        top1 = (pred_labels[:, 0] == y).sum()
        return (num, top1.item())

    def validation_epoch_end(self, outputs):
        total_num = 0
        total_top1 = 0
        for (num, top1) in outputs:
            total_num += num
            total_top1 += top1

        acc = float(total_top1 / total_num) * 100
        if acc > self.best_acc:
            self.best_acc = acc
        self.log("val/kNN_acc", acc)
        self.log("val/max_kNN_acc", self.best_acc)


    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=self.hparams.warmup_epochs, max_epochs=self.hparams.max_epochs
        )
        return [optimizer], [scheduler]




    # def configure_optimizers(self):
    #     lr = self.config["lr"]
    #     mom = self.config["momentum"]
    #     w_decay = self.config["weight_decay"]

    #     opts = {
    #         "adam": torch.optim.Adam(
    #             self.m_online.parameters(),
    #             lr=lr,
    #             weight_decay=w_decay,
    #         ),
    #         "sgd": torch.optim.SGD(
    #             self.m_online.parameters(),
    #             lr=lr,
    #             momentum=mom,
    #             weight_decay=w_decay,
    #         ),
    #     }

    #     opt = opts[self.config["opt"]]

    #     # Apply LARS wrapper if option is chosen
    #     if self.config["lars"]:
    #         opt = LARSWrapper(opt, eta=self.config["trust_coef"])

    #     if self.config["scheduler"] == "cosine":
    #         max_epochs = self.config["train"]["n_epochs"]
    #         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max_epochs)
    #         return [opt], [scheduler]

    #     # Currently produces weird results
    #     elif self.config["scheduler"] == "warmupcosine":
    #         scheduler = LinearWarmupCosineAnnealingLR(
    #             opt,
    #             self.config["warmup_epochs"],
    #             max_epochs=self.config["train"]["n_epochs"],
    #         )
    #         return [opt], [scheduler]

    #     elif self.config["scheduler"] == "None":
    #         return opt

    # @torch.no_grad()
    # def update_m_target(self):
    #     """Update target network without gradients"""
    #     m = self.config["m"]

    #     if self.config["m_decay"]:
    #         epoch = self.current_epoch
    #         n_epochs = self.config["train"]["n_epochs"]
    #         m = 1 - (1 - m) * (cos(pi * epoch / n_epochs) + 1) / 2

    #     for param_q, param_k in zip(
    #         self.m_online.parameters(), self.m_target.parameters()
    #     ):
    #         param_k.data = param_k.data * m + param_q.data * (1.0 - m)

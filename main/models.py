import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import pytorch_lightning as pl
import torchmetrics.functional as tmF
import numpy as np
from tqdm import tqdm
from utilities import LARSWrapper

from statistics import mean
from sklearn.decomposition import IncrementalPCA

from networks.models import MLPHead, LogisticRegression
from networks.models import ResNet18, ResNet50, WideResNet50_2
from utilities import byol_loss, freeze_model
from paths import Path_Handler


class byol(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()  # save hyperparameters for easy inference
        self.config = config
        paths = Path_Handler()
        self.paths = paths.dict

        model_dict = {
            "resnet18": ResNet18,
            "resnet50": ResNet50,
            "wideresnet50_2": WideResNet50_2,
        }

        self.m_online = model_dict[config["model"]["arch"]](**config)
        self.m_target = model_dict[config["model"]["arch"]](**config)
        self.predictor = MLPHead(
            in_channels=self.m_online.projection.net[-1].out_features,
            **config["projection_head"],
        )

        # print(torch.cuda.get_device_name(torch.cuda.current_device()))
        # print(torch.cuda.is_available())
        self.best_acc = 0

    def forward(self, x):
        """Return representation"""
        return self.m_online.encoder(x)

    def training_step(self, batch, batch_idx):
        # Load both views of x (y is a dummy label)
        x, _ = batch
        x1, x2 = x

        # Get targets for each view
        with torch.no_grad():
            y1, y2 = self.m_target(x1), self.m_target(x2)

        # Get predictions for each view
        f1, f2 = self.predictor(self.m_online(x1)), self.predictor(self.m_online(x2))

        # Calculate and log loss
        loss = byol_loss(f1, y1) + byol_loss(f2, y2)
        loss = torch.mean(loss)
        self.log("train/loss", loss)

        # Update target network using EMA (no gradients)
        self.update_m_target()

        return loss

    def configure_optimizers(self):
        opts = {
            "adam": torch.optim.Adam(self.m_online.parameters(), lr=self.config["lr"]),
            "sgd": torch.optim.SGD(
                self.m_online.parameters(),
                lr=self.config["lr"],
                momentum=self.config["momentum"],
                weight_decay=self.config["weight_decay"],
            ),
        }

        opt = opts[self.config["train"]["opt"]]

        if self.config["lars"]:
            opt = LARSWrapper(opt)

        return opt

    @torch.no_grad()
    def update_m_target(self):
        """Update target network without gradients"""
        m = self.config["m"]
        for param_q, param_k in zip(
            self.m_online.parameters(), self.m_target.parameters()
        ):
            param_k.data = param_k.data * m + param_q.data * (1.0 - m)


class linear_net(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        n_classes = self.config["data"]["classes"]
        self.logreg = LogisticRegression(self.config["model"]["output_dim"], n_classes)
        self.ce_loss = torch.nn.CrossEntropyLoss()

        paths = Path_Handler()
        self.paths = paths.dict

    def forward(self, x):
        """Return prediction"""
        return self.logreg(x)

    def training_step(self, batch, batch_idx):
        # Load data and targets
        x, y = batch
        x = x.view(x.shape[0], -1)
        logits = self.forward(x)
        y_pred = logits.softmax(dim=-1)
        loss = self.ce_loss(logits, y)
        self.log("linear_eval/train_loss", loss)

        # predictions = torch.argmax(logits, dim=1).int()
        acc = tmF.accuracy(y_pred, y)
        self.log("linear_eval/train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.shape[0], -1)
        logits = self.forward(x)
        y_pred = logits.softmax(dim=-1)
        loss = self.ce_loss(logits, y)
        self.log("linear_eval/val_loss", loss)

        # predictions = torch.argmax(logits, dim=1).int()
        acc = tmF.accuracy(y_pred, y)
        self.log("linear_eval/val_acc", acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.shape[0], -1)
        logits = self.forward(x)
        y_pred = logits.softmax(dim=-1)
        loss = self.ce_loss(logits, y)
        self.log("linear_eval/test_loss", loss)

        # predictions = torch.argmax(logits, dim=1).int()
        acc = tmF.accuracy(y_pred, y)
        self.log("linear_eval/test_acc", acc)

    def configure_optimizers(self):
        opt = torch.optim.SGD(
            self.logreg.parameters(),
            lr=self.config["linear"]["lr"],
            momentum=self.config["linear"]["momentum"],
            weight_decay=self.config["linear"]["weight_decay"],
        )
        return opt


class pca_net(nn.Module):
    def __init__(self, config):
        super(pca_net, self).__init__()
        self.config = config
        self.pca = IncrementalPCA(config["pca"]["n_dim"])

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.pca.transform(x)
        return torch.from_numpy(x).float()

    def fit(self, loader):
        print("Fitting PCA")
        for epoch in tqdm(np.arange(self.config["pca"]["n_epochs"])):
            print(f"Epoch {epoch}")
            for x, _ in tqdm(loader):
                x = x.view(x.shape[0], -1)
                x = x.cpu().detach().numpy()
                self.pca.partial_fit(x)

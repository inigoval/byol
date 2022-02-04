import pytorch_lightning as pl
import umap
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as tmF

from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
from pytorch_lightning.callbacks import Callback
from torch.utils.data import DataLoader

from dataloading.utils import dset2tens
from paths import Path_Handler
from networks.models import MLPHead, LogisticRegression


def umap(x, y):
    mapper = umap.UMAP().fit(x.view(x.shape[0], -1))
    umap.plot.points(mapper, labels=y)


def knn_predict(
    feature: torch.Tensor,
    feature_bank: torch.Tensor,
    target_bank: torch.Tensor,
    num_classes: int,
    knn_k: int = 200,
    knn_t: float = 0.1,
) -> torch.Tensor:
    """Code copied from https://github.com/lightly-ai/lightly/blob/master/lightly/utils/benchmarking.py

    Run kNN predictions on features based on a feature bank
    This method is commonly used to monitor performance of self-supervised
    learning methods.
    The default parameters are the ones
    used in https://arxiv.org/pdf/1805.01978v1.pdf.
    Args:
        feature:
            Tensor of shape [N, D] for which you want predictions
        feature_bank:
            Tensor of a database of features used for kNN
        target_bank:
            Labels for the features in our feature_bank
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
        >>>     target_bank,
        >>>     num_classes=10,
        >>> )
    """

    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)

    # [B, K]
    sim_weight, sim_idx = sim_matrix.topk(k=knn_k, dim=-1)

    # [B, K]
    sim_labels = torch.gather(
        target_bank.expand(feature.size(0), -1), dim=-1, index=sim_idx
    )
    # we do a reweighting of the similarities
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_y = torch.zeros(feature.size(0) * knn_k, num_classes)
    one_hot_y = one_hot_y.type_as(target_bank)

    # [B*K, C]
    one_hot_y = one_hot_y.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)

    # weighted score ---> [B, C]
    pred_scores = torch.sum(
        one_hot_y.view(feature.size(0), -1, num_classes) * sim_weight.unsqueeze(dim=-1),
        dim=1,
    )
    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels


class Feature_Bank(Callback):
    """Code adapted from https://github.com/lightly-ai/lightly/blob/master/lightly/utils/benchmarking.py

    Calculates a feature bank for validation"""

    def __init__(self):
        super().__init__()

    def on_validation_epoch_start(self, trainer, pl_module):
        data_bank = pl_module.trainer.datamodule.data["bank"]
        data_bank_loader = DataLoader(data_bank, 2000)
        feature_bank = []
        target_bank = []
        for data in data_bank_loader:
            # Load data and move to correct device
            x, y = data
            x = x.type_as(pl_module.dummy_param)
            y = y.type_as(pl_module.dummy_param).long()
            # y = y.to(torch.long)
            # y = y.to(pl_module.dummy_param.device)

            # Encode data and normalize features (for kNN)
            feature = pl_module.m_online.encoder(x).squeeze()
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
            target_bank.append(y)

        # Save full feature bank for validation epoch
        pl_module.feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        pl_module.target_bank = torch.cat(target_bank, dim=0).t().contiguous()


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

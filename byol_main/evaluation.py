import logging

import pytorch_lightning as pl
import umap
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as tmF
import torchmetrics as tm
import wandb

from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
from pytorch_lightning.callbacks import Callback
from torch.utils.data import DataLoader

from byol_main.dataloading.utils import dset2tens
from byol_main.paths import Path_Handler
from byol_main.networks.models import MLPHead, LogisticRegression
from byol_main.utilities import log_examples


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
        >>>     target_bank,
        >>>     num_classes=10,
        >>> )
    """

    assert target_bank.min() >= 0

    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(
        feature, feature_bank
    )  # (B, D) matrix. mult (D, N) gives (B, N) (as feature dim got summed over to got cos sim)

    # [B, K]
    sim_weight, sim_idx = sim_matrix.topk(
        k=knn_k, dim=-1
    )  # this will be slow if feature_bank is large (e.g. 100k datapoints)

    # [B, K]
    # target_bank is (1, N) (due to .t() in init)
    # feature.size(0) is the validation batch size
    # expand copies target_bank to (val_batch, N)
    # gather than indexes the N dimension to place the right labels (of the top k features), making sim_labels (val_batch) with values of the correct labels
    # if these aren't true, will get index error when trying to index target_bank
    assert sim_idx.min() >= 0
    assert sim_idx.max() < target_bank.size(0)
    sim_labels = torch.gather(target_bank.expand(feature.size(0), -1), dim=-1, index=sim_idx)
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


class Lightning_Eval(pl.LightningModule):
    # for many knn eval datasets
    def __init__(self, config):
        super().__init__()
        self.config = config
        

    def on_train_start(self):
        self.config["data"]["mu"] = self.trainer.datamodule.mu
        self.config["data"]["sig"] = self.trainer.datamodule.sig
        # self.log("train/mu", self.trainer.datamodule.mu)
        # self.log("train/sig", self.trainer.datamodule.sig)

        # logger = self.logger.experiment
        if not self.config['debug']:
            log_examples(self.logger, self.trainer.datamodule.data["train"])

    def on_validation_start(self):
        self.setup_knn_validation()
        if hasattr(self, self.supervised_head):
            self.setup_supervised_validation()

    def setup_knn_validation(self):
        with torch.no_grad():
            self.knn_eval_datasets = []
            # deprecating self.trainer.datamodule.data["labelled"] in favor of both being part of val_knn
            for knn_dataset_name, (_, val_databank) in self.data['val_knn'].items():
                logging.info('Using knn dataset {}, {} bank labels'.format(knn_dataset_name, len(val_databank)))
                # each KNN_Eval does the actual val work
                self.knn_eval_datasets += KNN_Eval(knn_dataset_name, val_databank, self.config, self.dummy_param, self.forward)

    def setup_supervised_validation(self):
        self.supervised_dataset = Supervised_Eval(self.represent, self.supervised_head, self.supervised_loss_func, self.dummy_param)

    # will only work if datamodule has self.data['val_supervised'] key
    # else will not be passed the extra dataloader_idx argument
    def validation_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx < len(self.knn_eval_datasets):  # first N are assumed dataloaders for knn eval
            self.knn_eval_datasets[dataloader_idx].validation_step(batch)  # knn validation
        else:
            assert hasattr(self.supervised_dataset)
            self.supervised_dataset.validation_step(batch)


    def forward(x):
        raise NotImplementedError('Must be subclassed by e.g. BYOL, which implements .forward(x)')

# for a single knn eval dataset
class KNN_Eval(pl.LightningModule):   # lightning subclass purely for self.log

    def __init__(self, name, data_bank, config, dummy_param, forward: callable):
        
        self.name = name
        self.data_bank = data_bank
        self.dummy_param = dummy_param
        self.config = config
        self.forward = forward  # explictly passed to __init__, composition-style

        self.knn_acc = tm.Accuracy(average="micro", threshold=0)

        self.setup_knn_features_and_targets()


    def setup_knn_features_and_targets(self):
        # TODO will have many databanks, each with different knn problem
        data_bank_loader = DataLoader(
                self.data_bank, 200
            )  # 200 is the batch size used for the unpacking below
        feature_bank = []
        target_bank = []
        for data in data_bank_loader:
            # Load data and move to correct device
            # (x, y,) = data  # supervised-style batch of (images, labels), with batch size from above
            x, y = data  # should be equiv.?
            x = x.type_as(self.dummy_param)
            y = y.type_as(self.dummy_param).long()

            # Encode data and normalize features (for kNN)
            feature = self.forward(x).squeeze()  # (batch, features)  e.g. (200, 512)
            feature = F.normalize(feature, dim=1)
            feature_bank.append(
                    feature
                )  # tensor of all features, within which to find nearest-neighbours. Each is (B, N_features), N_features being the output dim of self.forward e.g. BYOL
            target_bank.append(y)  # tensor with labels of those features

            # Save full feature bank for validation epoch
        self.feature_bank = (
                torch.cat(feature_bank, dim=0).t().contiguous()
            )  # (features, len(l datamodule)) due to the .t() transpose
        self.target_bank = (
                torch.cat(target_bank, dim=0).t().contiguous()
            )  # either (label_dim, len) or just (len) depending on if labels should have singleton label_dim dimension
        assert all(self.target_bank >= 0)


    def validation_step(self, batch):
        # Load batch
        x, y = batch
        # Extract + normalize features
        feature = self.forward(x).squeeze()  # from init(forward), likely BYOL's forward 
        feature = F.normalize(feature, dim=1)

        # Load feature bank and labels
        feature_bank = self.feature_bank.type_as(x)
        target_bank = self.target_bank.type_as(y)

        pred_labels = knn_predict(
                feature,  # feature to search for
                feature_bank,  # feature bank to identify NN within
                target_bank,  # labels of those features in feature_bank, same index
                self.config["data"]["classes"],
                knn_k=self.config["knn"]["neighbors"],
                knn_t=self.config["knn"]["temperature"],
            )

        top1 = pred_labels[:, 0]

            # Compute accuracy
            # assert top1.min() >= 0
        self.knn_acc.update(top1, y)

    def validation_epoch_end(self, outputs):
        if hasattr(self, "feature_bank") and hasattr(self, "target_bank"):
            self.log("val/kNN_acc/{}".format(self.name), self.knn_acc.compute() * 100)
            self.knn_acc.reset()

class Supervised_Eval(pl.LightningModule):

    def __init__(self, represent, supervised_head, supervised_loss_func, dummy_param) -> None:
        # this is getting a bit ugly because these get to Lightning_Eval's subclass via inheritance,
        # but I'm trying to use composition here to have a small object that does one thing
        # would work better to also get the above into Lightning_Eval via comp.
        self.represent = represent
        self.supervised_head = supervised_head
        self.supervised_loss_func = supervised_loss_func
        self.dummy_param = dummy_param

    def validation_step(self, batch):
        # get contrastive and supervised loss on validation set
        x, labels = batch
        # logging.info('x')
        # logging.info(x)
        x = x.type_as(self.dummy_param)
        y = self.represent(x)  # not a great name - this is the representation, pre-projection
        # logging.info('y')
        # logging.info(y)

        supervised_head_out = self.supervised_head(y)
        # p = self.project(y)
        # supervised_head_out = self.supervised_head(p)

        # logging.info('supervised_head_out')
        # logging.info(supervised_head_out)
        supervised_loss = self.supervised_loss_func(supervised_head_out, labels)  
        self.log("val/supervised_loss", supervised_loss, on_step=False, on_epoch=True) 


        

class Feature_Bank(Callback):
    """Code adapted from https://github.com/lightly-ai/lightly/blob/master/lightly/utils/benchmarking.py

    Calculates a feature bank for validation"""

    def __init__(self):
        super().__init__()

    def on_validation_epoch_start(self, trainer, pl_module):
        with torch.no_grad():
            encoder = pl_module.backbone

            data_bank = pl_module.trainer.datamodule.data["labelled"]
            data_bank_loader = DataLoader(data_bank, 200)
            feature_bank = []
            target_bank = []
            for data in data_bank_loader:
                # Load data and move to correct device
                x, y = data
                x = x.type_as(pl_module.dummy_param)
                y = y.type_as(pl_module.dummy_param).long()

                # Encode data and normalize features (for kNN)
                feature = encoder(x).squeeze()
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
        output_dim = self.config["model"]["output_dim"]
        self.logreg = LogisticRegression(output_dim, n_classes)
        self.ce_loss = torch.nn.CrossEntropyLoss()

        self.train_acc = tm.Accuracy(average="micro", top_k=1, threshold=0)
        self.val_acc = tm.Accuracy(average="micro", top_k=1, threshold=0)
        self.test_acc = tm.Accuracy(average="micro", top_k=1, threshold=0)

        self.dummy_param = nn.Parameter(torch.empty(0))

        paths = Path_Handler()
        self.paths = paths._dict()

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
        self.log("linear_eval/train_loss", loss, on_step=False, on_epoch=True)

        # predictions = torch.argmax(logits, dim=1).int()
        # acc = tmF.accuracy(y_pred, y)
        # self.log("linear_eval/train_acc", acc, on_step=False, on_epoch=True)

        self.train_acc(y_pred, y)
        self.log("linear_eval/train_acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.type_as(self.dummy_param)

        x = x.view(x.shape[0], -1)
        logits = self.forward(x)
        # y_pred = logits.softmax(dim=-1)

        loss = self.ce_loss(logits, y)
        self.log("linear_eval/val_loss", loss)

        # predictions = torch.argmax(logits, dim=1).int()
        # acc = tmF.accuracy(y_pred, y)
        self.val_acc(logits, y)
        self.log("linear_eval/val_acc", self.val_acc, on_step=False, on_epoch=True)

    # renamed duplicate of self.validation_step
    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.type_as(self.dummy_param)

        x = x.view(x.shape[0], -1)
        logits = self.forward(x)
        # y_pred = logits.softmax(dim=-1)

        loss = self.ce_loss(logits, y)
        self.log("linear_eval/test_loss", loss)

        # predictions = torch.argmax(logits, dim=1).int()
        # acc = tmF.accuracy(y_pred, y)
        self.test_acc(logits, y)
        self.log("linear_eval/test_acc", self.test_acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        lr = self.config["linear"]["lr"]
        mom = self.config["linear"]["momentum"]
        w_decay = self.config["linear"]["weight_decay"]

        params = self.logreg.parameters()

        opts = {
            "adam": lambda p: torch.optim.Adam(
                p,
                lr=lr,
                weight_decay=w_decay,
            ),
            "sgd": lambda p: torch.optim.SGD(
                p,
                lr=lr,
                momentum=mom,
                weight_decay=w_decay,
            ),
        }

        opt = opts[self.config["linear"]["opt"]](params)

        return opt


# currently not used
# class pca_net(nn.Module):
#     def __init__(self, config):
#         super(pca_net, self).__init__()
#         self.config = config
#         self.pca = IncrementalPCA(config["pca"]["n_dim"])

#     def forward(self, x):
#         x = x.view(x.shape[0], -1)
#         x = self.pca.transform(x)
#         return torch.from_numpy(x).float()

#     def fit(self, loader):
#         print("Fitting PCA")
#         for epoch in tqdm(np.arange(self.config["pca"]["n_epochs"])):
#             print(f"Epoch {epoch}")
#             for x, _ in tqdm(loader):
#                 x = x.view(x.shape[0], -1)
#                 x = x.cpu().detach().numpy()
#                 self.pca.partial_fit(x)

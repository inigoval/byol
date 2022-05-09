import torch
from byol import BYOL
from lightly.models.utils import update_momentum
from torch.nn.functional import cosine_similarity
import torch.nn.functional as F


def knn_weight(
    feature: torch.Tensor,
    feature_bank: torch.Tensor,
    knn_k: int = 5,
    knn_t: float = 0.1,
) -> torch.Tensor:
    """Code modified from https://github.com/lightly-ai/lightly/blob/master/lightly/utils/benchmarking.py

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
        >>> )
    """

    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(
        feature.squeeze(), feature_bank.squeeze()
    )  # (B, D) matrix. mult (D, N) gives (B, N) (as feature dim got summed over to got cos sim)

    # [B, K]
    sim_weight, sim_idx = sim_matrix.topk(
        k=knn_k, dim=-1
    )  # this will be slow if feature_bank is large (e.g. 100k datapoints)
    sim_weight = sim_weight.squeeze()

    # [B, K]
    # target_bank is (1, N) (due to .t() in init)
    # feature.size(0) is the validation batch size
    # expand copies target_bank to (val_batch, N)
    # gather than indexes the N dimension to place the right labels (of the top k features), making sim_labels (val_batch) with values of the correct labels
    # if these aren't true, will get index error when trying to index target_bank
    assert sim_idx.min() >= 0
    # we do a reweighting of the similarities
    # sim_weight = (sim_weight / knn_t).exp().squeeze().type_as(feature_bank)
    sim_weight = (sim_weight / knn_t).exp()

    return torch.mean(sim_weight, dim=-1, keepdim=False)


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
            sim_weights = knn_weight(r, r.t(), knn_k=self.config["n_knn"])

            # _, idx_m = torch.topk(sim_weights, int(n_batch * 0.1))

            # find threshold for similarity and create boolean mask for loss
            n_mask = int(n_batch * self.config["r_batch"])
            torch.use_deterministic_algorithms(False)
            sim_max = -torch.kthvalue(-sim_weights, n_mask)[0].item()
            torch.use_deterministic_algorithms(True)

            mask = sim_weights.lt(sim_max)

            print(mask)
            # mask = sim_weights.gt(sim_weights, torch.full_like(sim_weights, sim_max))

        p0 = self.project(x0)
        z0 = self.project_momentum(x0)
        p1 = self.project(x1)
        z1 = self.project_momentum(x1)
        # loss = 0.5 * (self.criterion(p0, z1) + self.criterion(p1, z0))

        loss = -0.5 * (cosine_similarity(p0, z1) + cosine_similarity(p1, z0))

        # mask out values with too high similarity
        loss = mask * loss

        print(loss)

        loss = loss.mean()

        self.log("train/loss", loss, on_step=False, on_epoch=True)
        return loss

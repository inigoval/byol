import torch
import torch.nn as nn
import lightly
import copy

import logging
from math import cos, pi
from byol_main.utilities import _optimizer
from lightly.models.modules.heads import BYOLProjectionHead
from lightly.models.utils import deactivate_requires_grad
from lightly.models.utils import update_momentum
from pytorch_lightning.callbacks import Callback

from zoobot.pytorch.estimators import efficientnet_custom
from zoobot.pytorch.training import losses

from byol_main.evaluation import Lightning_Eval
from byol_main.networks.models import _get_backbone


class BYOL(
    Lightning_Eval
):  # Lightning_Eval superclass adds validation step options for kNN evaluation
    def __init__(self, config):
        super().__init__(config)
        self.save_hyperparameters()  # save hyperparameters for easy inference
        self.config = config

        self.backbone = _get_backbone(config)

        # create a byol model based on ResNet
        features = self.config["model"]["features"]
        proj = self.config["projection_head"]
        # these are both basically small dense networks of different sizes
        # architecture is: linear w/ relu, batch-norm, linear
        # by default: representation (features)=512, hidden (both heads)=1024, out=256
        # so projection_head is 512->1024,relu/BN,1024->256
        # and prediction_head is 256->1024,relu/BN,1024->256
        self.projection_head = BYOLProjectionHead(features, proj["hidden"], proj["out"])
        self.prediction_head = BYOLProjectionHead(proj["out"], proj["hidden"], proj["out"])

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        self.criterion = lightly.loss.NegativeCosineSimilarity()

        self.dummy_param = nn.Parameter(torch.empty(0))

        self.m = config["m"]

    def forward(self, x):
        return self.backbone(x)  # dimension (batch, features), features from config e.g. 512

    def project(self, x):
        # representation
        y = self.backbone(x).flatten(start_dim=1)
        # projection
        z = self.projection_head(y)
        # prediction (of proj of target network)
        p = self.prediction_head(z)
        return p

    def project_momentum(self, x):
        y = self.backbone_momentum(x).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        z = z.detach()
        return z

    def training_step(self, batch, batch_idx):
        # Update momentum value
        update_momentum(self.backbone, self.backbone_momentum, m=self.m)
        update_momentum(self.projection_head, self.projection_head_momentum, m=self.m)

        # Load in data
        (x0, x1), _ = batch
        x0 = x0.type_as(self.dummy_param)
        x1 = x1.type_as(self.dummy_param)
        p0 = self.project(x0)
        z0 = self.project_momentum(x0)
        p1 = self.project(x1)
        z1 = self.project_momentum(x1)
        loss = 0.5 * (self.criterion(p0, z1) + self.criterion(p1, z0))
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        if self.config["m_decay"]:
            self.update_m()

    def configure_optimizers(self):
        params = (
            list(self.backbone.parameters())
            + list(self.projection_head.parameters())
            + list(self.prediction_head.parameters())
        )

        return _optimizer(params, self.config)

    def update_m(self):
        with torch.no_grad():
            epoch = self.current_epoch
            n_epochs = self.config["model"]["n_epochs"]
            self.m = 1 - (1 - self.m) * (cos(pi * epoch / n_epochs) + 1) / 2



class Update_M(Callback):
    """Updates EMA momentum"""

    def __init__(self):
        super().__init__()

    def on_training_epoch_end(self, trainer, pl_module):
        with torch.no_grad():
            config = pl_module.config
            epoch = pl_module.current_epoch
            n_epochs = config["model"]["n_epochs"]
            pl_module.m = 1 - (1 - pl_module.m) * (cos(pi * epoch / n_epochs) + 1) / 2


class BYOL_Supervised(BYOL):

    def __init__(self, config):
        super().__init__(config)
        # re-use the projection head pattern
        # also re-use the dimension
        features = self.config["model"]["features"]
        supervised_head_params = self.config["supervised_head"]

        if supervised_head_params['training_mode'] == 'classification':
            num_classes = config['data']['classes']
            logging.info('Adding supervised head in classification mode, {} classes'.format(num_classes))
            # remember this has batch-norm
            self.supervised_head = nn.Sequential(
                BYOLProjectionHead(features, supervised_head_params["hidden"], ),
                torch.nn.Softmax(dim=-1)
            )
            # ignore targets with value (aka class index) of -1
            self.supervised_loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1)
        elif supervised_head_params['training_mode'] == 'dirichlet':
            num_outputs = supervised_head_params['out']
            logging.info('Adding supervised head in dirichlet mode, {} outputs'.format(num_outputs))
            self.supervised_head = nn.Sequential(
                BYOLProjectionHead(features, supervised_head_params["hidden"], num_outputs),
                efficientnet_custom.ScaledSigmoid()   # sigmoid from 1 to 100
            )
            # my losses. code uses the wrong input convention (torch does preds, labels, but I did labels, preds) - adjust with lambda
            # sum is over questions as losses.multiquestion_loss gives loss like (batch, neg_log_prob_per_question)
            def dirichlet_loss_aggregated_to_scalar(preds, labels):
                dirichlet_loss = losses.calculate_multiquestion_loss(labels, preds, supervised_head_params['question_index_groups'])
                return torch.mean(dirichlet_loss)  # over both (batch, question)
                # p of (N=0, k=0) = 1 -> neg log p = 0 -> no effect on sum, but will reduce mean. Only absolute value though, not gradients per se if normalised

            self.supervised_loss_func = dirichlet_loss_aggregated_to_scalar

# I'm splitting self.project into two steps, self.represent then self.project, so I can use the rep without recalculating

    def represent(self, x):
        # representation
        return self.backbone(x).flatten(start_dim=1)

    def project(self, y):
        # now takes representation y as input
        # projection
        z = self.projection_head(y)
        # prediction (of proj of target network)
        p = self.prediction_head(z)
        return p


    def training_step(self, batch, batch_idx):

        # Update momentum value
        # aka update self.backbone_momentum with exp. moving av. of self.backbone
        # (similarly for heads)
        update_momentum(self.backbone, self.backbone_momentum, m=self.m)
        update_momentum(self.projection_head, self.projection_head_momentum, m=self.m)
        # prediction head not EMA'd as target (averaged) network doesn't need one
        # similarly don't need to EMA the supervised head as target network doesn't need one

        contrastive_loss, supervised_loss = self.calculate_losses(batch)

        # keep same name for wandb comparison
        # print('supervised vs contrastive: ', supervised_loss, contrastive_loss)
        self.log("train/loss", contrastive_loss, on_step=False, on_epoch=True)
        self.log("train/supervised_loss", supervised_loss, on_step=False, on_epoch=True) 

        # normalise supervised loss by contrastive loss
        # will have the gradients from supervised loss, but scaled to by similar to contrastive loss
        supervised_normalising_constant = torch.abs(contrastive_loss.detach()) / supervised_loss.detach()
        loss = contrastive_loss + self.config['supervised_loss_weight'] * supervised_normalising_constant * supervised_loss  
 
        # print('train/total_weighted_loss: ', loss)
        self.log("train/total_weighted_loss", loss, on_step=False, on_epoch=True)  
        return loss


    def calculate_losses(self, batch):
        # used in both train and val, so must have no side-effects on model(s)

        # Load in data
        # transforms.MultiView gives 2 views of same image
        (x0, x1), labels = batch
        x0 = x0.type_as(self.dummy_param)
        x1 = x1.type_as(self.dummy_param)

        y0 = self.represent(x0)  # supervised head uses y0
        p0 = self.project(y0)
        
        y1 = self.represent(x1)  # y1 not used except below
        p1 = self.project(y1)  # but re-using same funcs

        # I'm not splitting project_momentum as target network doesn't have/need supervised head
        # TODO should probably rename
        z0 = self.project_momentum(x0)
        z1 = self.project_momentum(x1)

        contrastive_loss = 0.5 * (self.criterion(p0, z1) + self.criterion(p1, z0))

        supervised_head_out = self.supervised_head(y0)

        supervised_loss = self.supervised_loss_func(supervised_head_out, labels)  

        return contrastive_loss, supervised_loss
    

    def validation_step(self, batch, batch_idx):
        
        # get contrastive and supervised loss on validation set
        contrastive_loss, supervised_loss = self.calculate_losses(batch)
        self.log("val/loss", contrastive_loss, on_step=False, on_epoch=True)
        self.log("val/supervised_loss", supervised_loss, on_step=False, on_epoch=True) 

        super().validation_step(batch, batch_idx)  # knn validation


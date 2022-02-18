"""
Note that this benchmark also supports a multi-GPU setup. If you run it on
a system with multiple GPUs make sure that you kill all the processes when
killing the application. Due to the way we setup this benchmark the distributed
processes might continue the benchmark if one of the nodes is killed.
If you know how to fix this don't hesitate to create an issue or PR :)
You can download the ImageNette dataset from here: https://github.com/fastai/imagenette
Code has been tested on a V100 GPU with 16GBytes of video memory.
Code to reproduce the benchmark results:
| Model       | Epochs | Batch Size | Test Accuracy | Peak GPU usage |
|-------------|--------|------------|---------------|----------------|
| MoCo        |  800   | 256        | 0.83          | 4.4 GBytes     |
| SimCLR      |  800   | 256        | 0.85          | 4.4 GBytes     |
| SimSiam     |  800   | 256        | 0.84          | 4.5 GBytes     |
| BarlowTwins |  200   | 256        | 0.80          | 4.5 GBytes     |
| BYOL        |  200   | 256        | 0.85          | 4.6 GBytes     |
| NNCLR       |  200   | 256        | 0.83          | 4.5 GBytes     |
| NNSimSiam   |  800   | 256        | 0.82          | 4.9 GBytes     |
| NNBYOL      |  800   | 256        | 0.85          | 4.6 GBytes     |
"""

import copy
import os

import lightly
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import logging
import wandb

from lightly.models.modules.heads import BYOLProjectionHead
from lightly.models.modules.heads import ProjectionHead
from lightly.models.utils import batch_shuffle
from lightly.models.utils import batch_unshuffle
from lightly.models.utils import deactivate_requires_grad
from lightly.models.utils import update_momentum
from lightly.utils import BenchmarkModule
from torchvision import transforms
from torchvision.transforms.transforms import CenterCrop

from config import load_config, update_config
from paths import Path_Handler
from dataloading.datamodules import imagenette_DataModule, imagenette_DataModule_eval
from evaluation import linear_net

paths = Path_Handler()
path_dict = paths._dict()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

config = load_config()
update_config(config)

pl.seed_everything(config["seed"])


num_workers = 12

logs_root_dir = os.path.join(os.getcwd(), "benchmark_logs")

# set max_epochs to 800 for long run (takes around 10h on a single V100)
max_epochs = 200
knn_k = 200
knn_t = 0.1
classes = 10
input_size = 128
num_ftrs = 512
nn_size = 2 ** 16

# benchmark
n_runs = 1  # optional, increase to create multiple runs and report mean + std
batch_sizes = [256]

# use a GPU if available
gpus = -1 if torch.cuda.is_available() else 0
distributed_backend = "ddp" if torch.cuda.device_count() > 1 else None

# The dataset structure should be like this:

path_to_train = path_dict["imagenette"] / "train"
path_to_test = path_dict["imagenette"] / "val"

# Use SimCLR augmentations
collate_fn = lightly.data.SimCLRCollateFunction(
    input_size=input_size,
)

# No additional augmentations for the test set
test_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(input_size),
        torchvision.transforms.CenterCrop(128),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=lightly.data.collate.imagenet_normalize["mean"],
            std=lightly.data.collate.imagenet_normalize["std"],
        ),
    ]
)

dataset_train_ssl = lightly.data.LightlyDataset(input_dir=path_to_train)

# we use test transformations for getting the feature for kNN on train data
dataset_train_kNN = lightly.data.LightlyDataset(
    input_dir=path_to_train, transform=test_transforms
)

dataset_test = lightly.data.LightlyDataset(
    input_dir=path_to_test, transform=test_transforms
)


def get_data_loaders(batch_size: int, multi_crops: bool = False):
    """Helper method to create dataloaders for ssl, kNN train and kNN test
    Args:
        batch_size: Desired batch size for all dataloaders
    """
    dataloader_train_ssl = torch.utils.data.DataLoader(
        dataset_train_ssl,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=num_workers,
    )

    dataloader_train_kNN = torch.utils.data.DataLoader(
        dataset_train_kNN,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    return dataloader_train_ssl, dataloader_train_kNN, dataloader_test


class BYOLModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)
        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()
        last_conv_channels = list(resnet.children())[-1].in_features
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.Conv2d(last_conv_channels, num_ftrs, 1),
            nn.AdaptiveAvgPool2d(1),
        )

        # create a byol model based on ResNet
        self.projection_head = BYOLProjectionHead(512, 1024, 256)
        self.prediction_head = BYOLProjectionHead(256, 1024, 256)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        self.criterion = lightly.loss.SymNegCosineSimilarityLoss()

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        return self.projection_head(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch

        # update momentum
        update_momentum(self.backbone, self.backbone_momentum, 0.99)
        update_momentum(self.projection_head, self.projection_head_momentum, 0.99)

        def step(x0_, x1_):
            x0_ = self.backbone(x0_).flatten(start_dim=1)
            x0_ = self.projection_head(x0_)
            x0_ = self.prediction_head(x0_)

            x1_ = self.backbone_momentum(x1_).flatten(start_dim=1)
            x1_ = self.projection_head_momentum(x1_)
            return x0_, x1_

        p0, z1 = step(x0, x1)
        p1, z0 = step(x1, x0)

        loss = self.criterion((z0, p0), (z1, p1))
        self.log("train/loss", loss)
        return loss

    def configure_optimizers(self):
        params = (
            list(self.backbone.parameters())
            + list(self.projection_head.parameters())
            + list(self.prediction_head.parameters())
        )
        optim = torch.optim.SGD(params, lr=6e-2, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]


batch_size = config["batch_size"]

dataloader_train_ssl, dataloader_train_kNN, dataloader_test = get_data_loaders(
    batch_size, multi_crops=False
)

model = BYOLModel(dataloader_train_kNN, classes)

config["model"]["output_dim"] = 512

# Save model with best accuracy for test evaluation, model will be saved in wandb and also #
pretrain_checkpoint = pl.callbacks.ModelCheckpoint(
    monitor="train/loss",
    mode="min",
    every_n_epochs=1,
    save_on_train_epoch_end=True,
    auto_insert_metric_name=False,
    verbose=True,
    dirpath="wandb/",
    filename="{train/loss:.3f}",
    save_weights_only=True,
)


# Initialise wandb logger, change this if you want to use a different logger #
logger = pl.loggers.WandbLogger(
    project="byol-lightly",
    save_dir=path_dict["files"],
    reinit=True,
    config=config,
)


trainer_settings = {
    "slurm": {"gpus": 1, "num_nodes": 4, "strategy": "ddp"},
    "gpu": {"devices": 1, "accelerator": "gpu"},
}


pre_trainer = pl.Trainer(
    # gpus=1,
    **trainer_settings[config["compute"]],
    fast_dev_run=config["debug"],
    max_epochs=config["train"]["n_epochs"],
    logger=logger,
    deterministic=True,
    callbacks=[pretrain_checkpoint],
)

pre_trainer.fit(
    model,
    train_dataloader=dataloader_train_ssl,
    val_dataloaders=dataloader_test,
)


##################################################
########## LINEAR EVALUATION PROTOCOL ############
##################################################


# Extract and load best encoder from pretraining
if config["debug"] is True:
    encoder = model.backbone
else:
    best_model_path = pretrain_checkpoint.best_model_path
    pretrained_model = BYOLModel.load_from_checkpoint(best_model_path)
    encoder = pretrained_model.backbone

# Freeze encoder weights
deactivate_requires_grad(encoder)

# Switch data-loader to linear evaluation mode
eval_data = imagenette_DataModule_eval(encoder, config)
eval_data.prepare_data()
eval_data.setup()

if not config["debug"]:
    config["eval"]["mu"] = eval_data.mu
    config["eval"]["sig"] = eval_data.sig

linear_checkpoint = pl.callbacks.ModelCheckpoint(
    monitor="linear_eval/val_acc",
    mode="max",
    every_n_epochs=1,
    verbose=True,
)

linear_trainer = pl.Trainer(
    **trainer_settings[config["compute"]],
    fast_dev_run=config["debug"],
    max_epochs=config["linear"]["n_epochs"],
    logger=logger,
    deterministic=True,
)

linear_model = linear_net(config)
linear_trainer.fit(linear_model, eval_data)
# linear_trainer.test(linear_model, dataloaders=eval_data, ckpt_path="best")

logger.experiment.finish()

import wandb
import torch
import numpy as np
import pytorch_lightning as pl

from paths import Path_Handler
from dataloading.datamodules import mbDataModule, reduce_mbDataModule
from models import pretrain_net, linear_net
from config import load_config
from utilities import freeze_model, log_examples

config = load_config()

paths = Path_Handler()
path_dict = paths._dict()


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
wandb_logger = pl.loggers.WandbLogger(
    project=config["project_name"],
    save_dir=path_dict["files"],
    reinit=True,
    config=config,
)

# Load data and record hyperparameters #
data = mbDataModule(config)
data.prepare_data()
data.setup()
wandb_logger.log_hyperparams(data.hyperparams)

# Record mean and standard deviation used in normalisation for inference #
config["data"]["mu"] = data.mu.item()
config["data"]["sig"] = data.sig.item()
log_examples(wandb_logger, data.data["train"])


# List of callbacks
callbacks = [pretrain_checkpoint]


pre_trainer = pl.Trainer(
    # gpus=1,
    devices=1,
    accelerator="gpu",
    max_epochs=config["train"]["n_epochs"],
    logger=wandb_logger,
    deterministic=True,
    callbacks=callbacks,
    #    check_val_every_n_epoch=3,
    #    log_every_n_steps=10,
)

# Initialise model #
model = pretrain_net(config)
config["model"]["output_dim"] = model.m_online.projection.net[0].in_features

# Train model #
pre_trainer.fit(model, data)

# Run test loop #
# pre_trainer.test(ckpt_path="best")

# Save model in wandb #
wandb.save(pretrain_checkpoint.best_model_path)


##################################################
########## LINEAR EVALUATION PROTOCOL ############
##################################################

# Switch loader to linear evaluation mode
linear_checkpoint = pl.callbacks.ModelCheckpoint(
    monitor="val/loss",
    mode="min",
    every_n_epochs=1,
    verbose=True,
)

best_model_path = pretrain_checkpoint.best_model_path
pretrained_model = pretrain_net.load_from_checkpoint(best_model_path)
encoder = pretrained_model.m_online.encoder
freeze_model(encoder)

eval_data = reduce_mbDataModule(encoder, config)
eval_data.prepare_data()
eval_data.setup()

config["eval"]["mu"] = eval_data.mu.item()
config["eval"]["sig"] = eval_data.sig.item()

linear_trainer = pl.Trainer(
    devices=1,
    accelerator="gpu",
    max_epochs=config["linear"]["n_epochs"],
    logger=wandb_logger,
    deterministic=True,
    #    check_val_every_n_epoch=3,
    #    log_every_n_steps=10,
)

linear_model = linear_net(config)
linear_trainer.fit(linear_model, eval_data)
linear_trainer.test(linear_model, dataloaders=eval_data, ckpt_path="best")

wandb_logger.experiment.finish()

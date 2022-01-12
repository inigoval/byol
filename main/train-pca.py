import wandb
import torch
import numpy as np
import pytorch_lightning as pl


from paths import Path_Handler
from dataloading.datamodules import mbDataModule, reduce_mbDataModule
from models import pretrain_net, linear_net, pca_net
from config import load_config
from utilities import freeze_model, log_examples
from eval import lin_eval_protocol

config = load_config()

paths = Path_Handler()
path_dict = paths._dict()


##################################################
########## LINEAR EVALUATION PROTOCOL ############
##################################################

# Make sure linear model uses correct dimension & record pca use
config["model"]["output_dim"] = config["pca"]["n_dim"]
config["type"] = "pca"
config["batch_size"] = config["pca"]["batch_size"]

# Initialise wandb logger, change this if you want to use a different logger #
wandb_logger = pl.loggers.WandbLogger(
    project=config["project_name"],
    save_dir=path_dict["files"],
    reinit=True,
    config=config,
)

# Load data from datamodule
data = mbDataModule(config)
data.prepare_data()
data.setup()
wandb_logger.log_hyperparams(data.hyperparams)

# Create dataloader
loader = data.train_dataloader()

# Initialise and fit PCA
encoder = pca_net(config)
encoder.fit(loader)

lin_eval_protocol(config, encoder, wandb_logger)

# Switch loader to linear evaluation mode
# linear_checkpoint = pl.callbacks.ModelCheckpoint(
#    monitor="linear_eval/val_acc",
#    mode="max",
#    every_n_epochs=1,
#    verbose=True,
# )
#
#
# eval_data = reduce_mbDataModule(encoder, config)
# eval_data.prepare_data()
# eval_data.setup()
#
# config["eval"]["mu"] = eval_data.mu.item()
# config["eval"]["sig"] = eval_data.sig.item()
#
# linear_trainer = pl.Trainer(
#    devices=1,
#    accelerator="gpu",
#    max_epochs=config["linear"]["n_epochs"],
#    logger=wandb_logger,
#    deterministic=True,
#    #    check_val_every_n_epoch=3,
#    #    log_every_n_steps=10,
# )
#
# linear_model = linear_net(config)
# linear_trainer.fit(linear_model, eval_data)
# linear_trainer.test(linear_model, dataloaders=eval_data, ckpt_path="best")
#
wandb_logger.experiment.finish()

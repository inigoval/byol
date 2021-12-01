import wandb
import torch
import numpy as np
import pytorch_lightning as pl

from paths import Path_Handler
from dataloading.datamodules import mbDataModule
from model import net
from config import load_config

config = load_config()

paths = Path_Handler()
path_dict = paths._dict()


# Save model with best accuracy for test evaluation, model will be saved in wandb and also #
checkpoint_callback = pl.callbacks.ModelCheckpoint(
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

# you can add ImpurityLogger if NOT using rgz unlabelled data to track impurities and mask rate
callbacks = [
    checkpoint_callback,
]


trainer = pl.Trainer(
    gpus=1,
    max_epochs=config["train"]["n_epochs"],
    logger=wandb_logger,
    deterministic=True,
    callbacks=callbacks,
    #    check_val_every_n_epoch=3,
    #    log_every_n_steps=10,
)

# Initialise model #
model = net(config)

# Train model #
trainer.fit(model, data)

# Run test loop #
trainer.test(ckpt_path="best")

# Save model in wandb #
wandb.save(checkpoint_callback.best_model_path)

wandb_logger.experiment.finish()

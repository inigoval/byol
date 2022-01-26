import pytorch_lightning as pl
import umap

from models import linear_net
from dataloading.utils import dset2tens


def lin_eval_protocol(config, data, encoder, wandb_logger):
    # Switch loader to linear evaluation mode
    linear_checkpoint = pl.callbacks.ModelCheckpoint(
        monitor="linear_eval/val_acc",
        mode="max",
        every_n_epochs=1,
        verbose=True,
    )

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
    linear_trainer.fit(linear_model, data)
    linear_trainer.test(linear_model, dataloaders=data, ckpt_path="best")


def umap(dset):
    x, y = dset2tens(dset)
    mapper = umap.UMAP().fit(x.view(x.shape[0], -1))
    umap.plot.points(mapper, labels=y)

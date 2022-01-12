import pytorch_lightning as pl

from dataloading.datamodules import reduce_mbDataModule
from models import linear_net


def lin_eval_protocol(config, encoder, wandb_logger):
    # Switch loader to linear evaluation mode
    linear_checkpoint = pl.callbacks.ModelCheckpoint(
        monitor="linear_eval/val_acc",
        mode="max",
        every_n_epochs=1,
        verbose=True,
    )

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

import wandb
import pytorch_lightning as pl
import logging
import torch

from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.profiler import AdvancedProfiler, PyTorchProfiler

from byol_main.byol import BYOL
from byol_main.nnclr import NNCLR
from byol_main.config import load_config, update_config
from byol_main.utilities import log_examples
from dataloading.datamodules import datasets
from paths import Path_Handler, create_path

from byol_rr import BYOL_RR
from byol_pretext import BYOL_Pretext
from supervised import Supervised

# TODO put elsewhere
# https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch-lightning/Profile_PyTorch_Code.ipynb#scrollTo=qRoUXZdtJIUD


class TorchTensorboardProfilerCallback(pl.Callback):
    """Quick-and-dirty Callback for invoking TensorboardProfiler during training.

    For greater robustness, extend the pl.profiler.profilers.BaseProfiler. See
    https://pytorch-lightning.readthedocs.io/en/stable/advanced/profiler.html"""

    def __init__(self, profiler):
        super().__init__()
        self.profiler = profiler

    def on_train_batch_end(self, trainer, pl_module, outputs, *args, **kwargs):
        self.profiler.step()
        pl_module.log_dict(outputs)  # also logging the loss, while we're here


def run_contrastive_pretraining(config, wandb_logger):

    pl.seed_everything(config["seed"])

    # Save model for test evaluation
    # TODO might be better to use val/supervised_loss when available
    loss_to_monitor = "train/loss"

    if (config["type"] == "byol_supervised") and (config["supervised_loss_weight"] > 0):
        loss_to_monitor = "val/supervised_loss/dataloader_idx_2"

    checkpoint_mode = {
        "min_loss": {"mode": "min", "monitor": loss_to_monitor},
        "last": {"monitor": None},
    }

    ## Creates experiment path if it doesn't exist already ##
    experiment_dir = config["files"] / config["run_id"]
    create_path(experiment_dir)

    ## Initialise checkpoint ##
    pretrain_checkpoint = pl.callbacks.ModelCheckpoint(
        **checkpoint_mode[config["checkpoint_mode"]],
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        auto_insert_metric_name=False,
        verbose=True,
        dirpath=experiment_dir / "checkpoints",
        # e.g. byol/files/(run_id)/checkpoints/12-344-18.134.ckpt.
        filename="{epoch}-{step}-{loss_to_monitor:.4f}",  # filename may not work here TODO
        save_weights_only=True,
        # save_top_k=3,
    )
    logging.info(f"checkpoint monitoring: {checkpoint_mode[config['checkpoint_mode']]}")

    ## Initialise data and run set up ##
    pretrain_data = datasets[config["dataset"]](config)
    pretrain_data.prepare_data()
    pretrain_data.setup()
    logging.info(f"mean: {pretrain_data.mu}, sigma: {pretrain_data.sig}")

    ## Initialise callbacks ##
    callbacks = [pretrain_checkpoint]

    # add learning rate monitor, only supported with a logger
    if wandb_logger is not None:
        # change to step, may be slow
        callbacks += [LearningRateMonitor(logging_interval="step")]

    # if config['profiler'] == 'kineto':
    # callbacks += [profiler_callback]

    ## Add profiler ##
    if config["profiler"] == "advanced":
        logging.info("Using advanced profiler")
        profiler = AdvancedProfiler(dirpath=experiment_dir, filename="advanced_profile")  # .txt
    elif config["profiler"] == "pytorch":
        logging.info("Using pytorch profiler")
        # .txt
        profiler = PyTorchProfiler(dirpath=experiment_dir, filename="pytorch_profile", row_limit=-1)
    else:
        logging.info("No profiler used")
        profiler = None

    logging.info(f"Threads: {torch.get_num_threads()}")

    # TODO could probably be directly included in config rather than config['compute'] indexing this
    # This has been written like this so that you only need to change one setting in the config for swapping between slurm and direct on gpu
    trainer_settings = {
        "slurm": {"gpus": 1, "num_nodes": 1},
        "gpu": {"devices": 1, "accelerator": "gpu"},
    }
    config["trainer_settings"] = trainer_settings[config["compute"]]

    ## Initialise pytorch lightning trainer ##
    pre_trainer = pl.Trainer(
        # gpus=1,
        **trainer_settings[config["compute"]],
        fast_dev_run=config["debug"],
        max_epochs=config["model"]["n_epochs"],  # note that this will affect momentum of BYOL ensemble!
        logger=wandb_logger,
        deterministic=True,
        callbacks=callbacks,
        precision=config["precision"],
        check_val_every_n_epoch=config["evaluation"]["check_val_every_n_epoch"],
        log_every_n_steps=200,
        profiler=profiler,
        # max_steps = 200  # TODO temp
    )

    # Initialise model #
    models = {
        "byol_pretext": BYOL_Pretext,
        "byol_rr": BYOL_RR,
        "byol": BYOL,
        "supervised": Supervised,
        "nnclr": NNCLR,
    }
    model = models[config["type"]](config)

    # profile_art = wandb.Artifact(f"trace-{wandb.run.id}", type="profile")
    # profile_art.add_file(glob.glob(str(experiment_dir / "*.pt.trace.json"))[0], "trace.pt.trace.json")
    # wandb.run.log_artifact(profile_art)

    config["model"]["output_dim"] = config["model"]["features"]

    ## Record some data-points and their augmentations if not in debugging mode ##
    if not config["debug"]:
        log_examples(wandb_logger, pretrain_data.data["train"])

    # Train model #
    pre_trainer.fit(model, pretrain_data)
    pre_trainer.test(model, dataloaders=pretrain_data)

    return pretrain_checkpoint, model


def main():

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    ## Load up config from yml files ##
    config = load_config()
    update_config(config)

    wandb.init(project=config["project_name"])
    config["run_id"] = str(wandb.run.id)

    # Initialise wandb logger, change this if you want to use a different logger #
    # paths = Path_Handler()
    # path_dict = paths._dict()
    # wandb_save_dir = path_dict["files"] / 'wandb'  # e.g. (repo aka byol)/files
    # independent of model checkpoint loc

    # structure will be e.g.
    # config['files'] / l5ikqywp / checkpoints / {}.ckpt
    # config['files'] / l5ikqywp / run-20220513_122412-l5ikqywp / (wandb stuff)

    path_dict = Path_Handler()._dict()

    wandb_logger = pl.loggers.WandbLogger(
        project=config["project_name"],
        save_dir=path_dict["files"]
        / config["run_id"],  # and will then add e.g. run-20220513_122412-l5ikqywp automatically
        reinit=True,
        config=config,
    )

    config["files"] = path_dict["files"]

    ## Run pretraining ##
    pretrain_checkpoint, model = run_contrastive_pretraining(config, wandb_logger)

    wandb_logger.experiment.finish()


if __name__ == "__main__":

    main()

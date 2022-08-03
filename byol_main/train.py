import wandb
import pytorch_lightning as pl
import logging
import os
import torch

from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.profiler import AdvancedProfiler, PyTorchProfiler

from byol_main.byol import BYOL, BYOL_Supervised
from byol_main.nnclr import NNCLR
from byol_main.evaluation import Linear_Eval, KNN_Eval
from byol_main.config import load_config
from byol_main.utilities import log_examples
from dataloading.datamodules import datasets
from paths import Path_Handler, create_path

from byol_rr import BYOL_RR, count_masks, Count_Masks
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


def run_contrastive_pretraining(config, wandb_logger, trainer_settings):

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

    experiment_dir = config["files"] / config["run_id"]
    create_path(experiment_dir)

    pretrain_checkpoint = pl.callbacks.ModelCheckpoint(
        **checkpoint_mode[config["checkpoint_mode"]],
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        auto_insert_metric_name=False,
        verbose=True,
        dirpath=experiment_dir / 'checkpoints',  # e.g. byol/files/(run_id)/checkpoints/12-344-18.134.ckpt. 
        filename="{epoch}-{step}-{loss_to_monitor:.4f}",  # filename may not work here TODO
        save_weights_only=True,
        save_top_k=3
    )


    pretrain_data = datasets[config["dataset"]]["pretrain"](config)
    # pretrain_data.prepare_data()
    # pretrain_data.setup()

        pretrain_data = datasets[config["dataset"]]["pretrain"](config)

    # from torch.profiler import tensorboard_trace_handler
    # import torch
    # import glob

    # # default scheduler
    # profiler = torch.profiler.profile(on_trace_ready=tensorboard_trace_handler(str(experiment_dir)), with_stack=True)

    # with profiler:
    #     profiler_callback = TorchTensorboardProfilerCallback(profiler)


    # List of callbacks
    callbacks = [pretrain_checkpoint]
    if wandb_logger is not None:
        # only supported with a logger
        callbacks += [LearningRateMonitor(logging_interval='step')]  # change to step, may be slow
        # if config['profiler'] == 'kineto':

        # callbacks += [profiler_callback]
            

    if config['profiler'] == 'advanced':
        logging.info('Using advanced profiler')
        profiler = AdvancedProfiler(dirpath=experiment_dir, filename='advanced_profile')  # .txt
    elif config['profiler'] == 'pytorch':
        logging.info('Using pytorch profiler')
        profiler = PyTorchProfiler(dirpath=experiment_dir, filename='pytorch_profile', row_limit=-1)  # .txt
    else:
        logging.info('No profiler used')
        profiler=None

    import torch
    logging.info('Threads: {}'.format(torch.get_num_threads()))

    pre_trainer = pl.Trainer(
        # gpus=1,
        **trainer_settings[config["compute"]],
        fast_dev_run=config["debug"],
        max_epochs=config["model"]["n_epochs"],  # note that this will affect momentum of BYOL ensemble!
        logger=wandb_logger,
        deterministic=True,
        callbacks=callbacks,
        precision=config["precision"],
        #    check_val_every_n_epoch=3,
        log_every_n_steps=200,
        profiler=profiler,
        # max_steps = 200  # TODO temp
    )

    trainer_settings = {
        "slurm": {"gpus": 1, "num_nodes": 1},
        "gpu": {"devices": 1, "accelerator": "gpu"},
    }
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

    config = load_config()

    wandb.init(project=config["project_name"])
    config["run_id"] = str(wandb.run.id)

    # TODO could probably be directly included in config rather than config['compute'] indexing this
    trainer_settings = {
        "slurm": {"gpus": 1, "num_nodes": 1},
        "gpu": {"devices": 1, "accelerator": "gpu"},
    }

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

    pretrain_checkpoint, model = run_contrastive_pretraining(config, wandb_logger, trainer_settings)

    # run_linear_evaluation_protocol(config, wandb_logger, pretrain_checkpoint, trainer_settings, model)

    wandb_logger.experiment.finish()


if __name__ == "__main__":

    main()

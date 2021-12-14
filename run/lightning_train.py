from pathlib import Path
from typing import Dict

import wandb.util
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from src.config import Config
from src.model import get_model
from src.ray import initialize_ray


def train():
    wandb.login()
    Config.load_default()
    model = get_model()
    trainer = get_trainer()
    trainer.fit(model)


def get_trainer():
    train_config: Dict = Config["train"]
    steps = train_config["steps"]
    run_id = f"{train_config['bitrate']}-vfs_net-{wandb.util.generate_id()}".strip()
    dashified_characters = ":' {}[]()/,=\""
    for c in dashified_characters:
        run_id = run_id.replace(c, "-")
    Path(train_config["project_name"]).mkdir(exist_ok=True)
    wandb_logger = WandbLogger(save_dir=train_config["save_dir"], project=train_config["project_name"], id=run_id, log_model=True)
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_top_k=-1, save_last=True)
    learning_rate_monitor = LearningRateMonitor(logging_interval='step')
    if "ray" in Config.value:
        initialize_ray()
        from ray_lightning import RayPlugin
        plugins = [RayPlugin(
            use_gpu=True,
            num_workers=Config["ray"]["num_workers"],
            num_cpus_per_worker=Config["ray"]["num_cpus_per_worker"],
        )]
    else:
        plugins = []
    callbacks = [checkpoint_callback, learning_rate_monitor]
    val_check_interval = Config["train"]["val_check_interval"]
    return pl.Trainer(
        max_steps=steps, progress_bar_refresh_rate=20, gpus=1, logger=wandb_logger, callbacks=callbacks, plugins=plugins, sync_batchnorm=True, val_check_interval=val_check_interval
    )


if __name__ == "__main__":
    train()

from typing import Callable

import pytorch_lightning as pl
import torchvision
import wandb


class WandbImageSaver:
    def __init__(self):
        self.buffer = []
        self.limit = 40
        self.nrow = 4

    def add(self, images_getter: Callable):
        if len(self.buffer) < self.limit:
            self.buffer.extend(images_getter())

    def flush(self, model: pl.LightningModule, key="val_images"):
        if not self.buffer:
            return

        grid = torchvision.utils.make_grid(self.buffer, nrow=self.nrow).clip_(-1, 1)
        model.logger.experiment.log(
            {
                key: wandb.Image(grid.cpu(), caption="input - output - gt")
            },
        )
        self.buffer.clear()

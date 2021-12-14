from abc import ABC

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_msssim import ms_ssim
from torchmetrics.functional import ssim, psnr

from src.config import Config
from src.dataset import get_dataset
from src.model.util.wandb_image_saver import WandbImageSaver
from src.optimizer import get_optimizer


class StreamingModelABC(pl.LightningModule, ABC):
    def __init__(self):
        super().__init__()
        self.val_images_saver = WandbImageSaver()
        self.train_images_saver = WandbImageSaver()

        self.loss_alpha = float(Config["loss_alpha"])

        self.train_dataset = get_dataset(Config["train_dataset"])
        self.valid_dataset = get_dataset(Config["valid_dataset"])

    def criterion(self, pred, target):
        l1_loss = F.l1_loss(pred, target)
        ms_ssim_loss = 1.0 - ms_ssim(pred, target, data_range=2.0)
        return (
            ms_ssim_loss * self.loss_alpha
            + l1_loss * (1.0 - self.loss_alpha)
        )

    def forward(self, rgb: torch.Tensor):
        return self.model(rgb)

    def training_step(self, batch, batch_idx):
        compressed_rgb = batch["compressed"]
        raw_rgb = batch["raw"]
        deblocked_rgb = batch["deblocked"]

        pred_rgb = self.forward(compressed_rgb)

        loss = self.criterion(pred_rgb, raw_rgb)

        pred_rgb = pred_rgb.detach().clip(-1, 1)
        ms_ssim_value = ms_ssim(pred_rgb, raw_rgb, data_range=2.0)
        psnr_value = psnr(pred_rgb, raw_rgb, data_range=2.0, dim=(1, 2, 3))

        raw_ms_ssim_value = ms_ssim(compressed_rgb, raw_rgb, data_range=2.0)
        raw_psnr_value = psnr(compressed_rgb, raw_rgb, data_range=2.0, dim=(1, 2, 3))

        deblocked_ms_ssim_value = ms_ssim(deblocked_rgb, raw_rgb, data_range=2.0)
        deblocked_psnr_value = psnr(deblocked_rgb, raw_rgb, data_range=2.0, dim=(1, 2, 3))

        self.log(
            "train", {
                "loss": loss,
                "ms_ssim": ms_ssim_value,
                "psnr": psnr_value,
                "raw_ms_ssim": raw_ms_ssim_value,
                "raw_psnr": raw_psnr_value,
                "deblocked_ms_ssim": deblocked_ms_ssim_value,
                "deblocked_psnr": deblocked_psnr_value
            },
        )

        self.train_images_saver.add(
            lambda:
            [
                compressed_rgb[0].cpu(),
                pred_rgb[0].detach().cpu(),
                deblocked_rgb[0].detach().cpu(),
                raw_rgb[0].cpu()
            ]
        )

        return loss

    def validation_step(self, batch, batch_idx):
        compressed_rgb = batch["compressed"]
        raw_rgb = batch["raw"]
        deblocked_rgb = batch["deblocked"]

        pred_rgb = self.forward(compressed_rgb)

        loss = self.criterion(pred_rgb, raw_rgb)

        pred_rgb = pred_rgb.detach().clip(-1, 1)
        ms_ssim_value = ms_ssim(pred_rgb, raw_rgb, data_range=2.0)
        psnr_value = psnr(pred_rgb, raw_rgb, data_range=2.0, dim=(1, 2, 3))

        raw_ms_ssim_value = ms_ssim(compressed_rgb, raw_rgb, data_range=2.0)
        raw_psnr_value = psnr(compressed_rgb, raw_rgb, data_range=2.0, dim=(1, 2, 3))

        deblocked_ms_ssim_value = ms_ssim(deblocked_rgb, raw_rgb, data_range=2.0)
        deblocked_psnr_value = psnr(deblocked_rgb, raw_rgb, data_range=2.0, dim=(1, 2, 3))

        self.val_images_saver.add(
            lambda:
            [
                compressed_rgb[0].cpu(),
                pred_rgb[0].cpu(),
                deblocked_rgb[0].detach().cpu(),
                raw_rgb[0].cpu()
            ]
        )
        return {
            "val_loss": loss,
            "ms_ssim": ms_ssim_value,
            "psnr": psnr_value,
            "raw_ms_ssim": raw_ms_ssim_value,
            "raw_psnr": raw_psnr_value,
            "deblocked_ms_ssim": deblocked_ms_ssim_value,
            "deblocked_psnr": deblocked_psnr_value
        }

    def validation_epoch_end(self, outputs) -> None:
        super().validation_epoch_end(outputs)
        keys = list(outputs[0].keys())
        keys.remove("val_loss")

        self.log("val_loss", torch.stack([x["val_loss"] for x in outputs]).mean(), sync_dist=True)
        self.log(
            "valid", {
                key: torch.stack([x[key] for x in outputs]).mean()
                for key in keys
            },
            sync_dist=True,
        )

        self.val_images_saver.flush(self, key="val_images")
        self.train_images_saver.flush(self, key="train_images")

    def on_fit_start(self) -> None:
        self.logger.experiment.config.update(Config.value)

    def configure_optimizers(self):
        generator_opt = get_optimizer(
            self.parameters(), Config["optimizer_generator"]
        )
        return generator_opt

    def train_dataloader(self):
        return self.get_dataloader(self.train_dataset, Config["train_dataset"])

    def val_dataloader(self):
        return self.get_dataloader(self.valid_dataset, Config["valid_dataset"])

    @staticmethod
    def get_dataloader(dataset, dataset_config):
        return DataLoader(
            dataset,
            batch_size=dataset_config["batch_size"],
            shuffle=dataset_config["shuffle"],
            num_workers=dataset_config["num_workers"],
            prefetch_factor=dataset_config["prefetch_factor"],
            pin_memory=True,
            drop_last=True
        )

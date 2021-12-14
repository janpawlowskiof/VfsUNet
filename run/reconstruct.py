from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image

from src.config import Config
from src.dataset.mp4_clip import Mp4Clip, ProcessedImages
from src.model.util.transforms import StreamingTransformations


def reconstruct_clip(raw_clip: Mp4Clip, blocky_clip: Mp4Clip, deblocked_clip: Mp4Clip, model, func):
    transforms = StreamingTransformations.from_config(Config["valid_dataset"]["transformations"])
    raw_clip.generate_residue(blocky_clip, deblocked_clip, model=model, transforms=transforms, func=func)


class ImagesGridSaver:
    def __init__(self, output_path: Path):
        output_path.mkdir(parents=True, exist_ok=True)
        self.output_path = output_path
        self.index = 0

    def save_images_in_grid(self, processed_images: ProcessedImages):
        processed_images = list(processed_images)
        grid = torchvision.utils.make_grid(processed_images, nrow=3, padding=10)
        grid = grid.cpu().numpy().astype(np.uint8)
        grid = np.transpose(grid, (1, 2, 0))

        image_path = str(self.output_path / f"{self.index}.jpg")
        Image.fromarray(grid).save(image_path, quality=95)
        print(image_path)
        self.index += 1


class ReconstructedImageSaver:
    def __init__(self, output_path: Path):
        output_path.mkdir(parents=True, exist_ok=True)
        self.output_path = output_path
        self.index = 0

    def save_images_in_grid(self, processed_images: ProcessedImages):
        blocky = processed_images.blocky_image.cpu().numpy().astype(np.uint8)
        blocky = np.transpose(blocky, (1, 2, 0))
        image_path = str(self.output_path / f"{self.index}_blocky.png")
        Image.fromarray(blocky).save(image_path)

        final = processed_images.final_image.cpu().numpy().astype(np.uint8)
        final = np.transpose(final, (1, 2, 0))
        image_path = str(self.output_path / f"{self.index}_pred.png")
        Image.fromarray(final).save(image_path)

        deblocked = processed_images.deblocked_image.cpu().numpy().astype(np.uint8)
        deblocked = np.transpose(deblocked, (1, 2, 0))
        image_path = str(self.output_path / f"{self.index}_deblocked.png")
        Image.fromarray(deblocked).save(image_path)

        raw = processed_images.raw_image.cpu().numpy().astype(np.uint8)
        raw = np.transpose(raw, (1, 2, 0))
        image_path = str(self.output_path / f"{self.index}_raw.png")
        Image.fromarray(raw).save(image_path)

        print(image_path)
        self.index += 1


if __name__ == "__main__":
    Config.load_default()
    Config.value["train_dataset"]["name"] = "Empty"
    Config.value["valid_dataset"]["name"] = "Empty"

    # version = 'jan-pawlowski/vfs_unet_katana/model-katana-qp-39-vfs_net-1jhmi8xp:v30'
    # # version = 'jan-pawlowski/vfs_unet_trackmania/model-trackmania-qp-39-vfs_net-3df3mnhn:v30'
    #
    # # version = 'jan-pawlowski/vfs_unet_gta_v/model-gta-qp-39-vfs_net-2sgi7lcs:v1'
    # # version = 'jan-pawlowski/katana/model-katana-new-qp-35--11--3--3--sum-True-SingleConv-qa8cnksc:v40'
    #
    # artifact_dir = wandb.init().use_artifact(version, type="model").download()
    # model = VfsUNetWrapper.load_from_checkpoint(f"{artifact_dir}/model.ckpt", strict=False).eval().cuda()
    model = torch.load("../trained_models/trackmania-qp37").eval().cuda()

    # raw_path = Path(r"/mnt/nfs_svtai09-nvme1n1p1/jpawlowski/trackmania/valid/raw/Trackmania2021.11.27-20.55.14.04.mp4")
    raw_path = Path(r"/mnt/nfs_svtai08-nvme1n1p1/katana_zero/valid/raw/KatanaZero2021.10.24-19.35.22.02.mp4")
    # raw_path = Path(r"/mnt/nfs_svtai10-nvme1n1p1/jpawlowski/gta_v/valid/raw/GrandTheftAutoV2021.11.07-11.29.04.11.mp4")
    # raw_path = Path(r"/mnt/nfs_svtai10-nvme1n1p1/jpawlowski/gta_v/valid/raw/GrandTheftAutoV2021.11.07-10.43.30.07.mp4")
    raw_clip = Mp4Clip(raw_path)
    bitrate = "qp=37"
    print(f"bitrate: {bitrate}")

    blocky_clip = raw_clip.change_bitrate(f"{bitrate}", Path(f"{raw_path.stem}_{bitrate}_blocky.mp4"), deblocking=False, skip_if_exists=True)
    deblocked_clip = raw_clip.change_bitrate(f"{bitrate}", Path(f"{raw_path.stem}_{bitrate}_deblocked.mp4"), deblocking=True, skip_if_exists=True)

    images_path: Path = Path(f"{raw_path.stem}_{bitrate}_images")
    func = ReconstructedImageSaver(images_path).save_images_in_grid
    reconstruct_clip(raw_clip, blocky_clip, deblocked_clip, model=model, func=func)

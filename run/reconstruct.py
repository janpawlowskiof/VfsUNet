import argparse
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
        self._save_frame(processed_images.blocky_image, "no_postprocess")
        self._save_frame(processed_images.final_image, "vfs_unet")
        self._save_frame(processed_images.deblocked_image, "dbf_sao")
        self._save_frame(processed_images.raw_image, "raw")
        self.index += 1

    def _save_frame(self, tensor_frame: torch.Tensor, name: str):
        frame = tensor_frame.cpu().numpy().astype(np.uint8)
        deblocked = np.transpose(frame, (1, 2, 0))
        image_path = str(self.output_path / f"{self.index}_{name}.png")
        Image.fromarray(deblocked).save(image_path)


if __name__ == "__main__":
    Config.load_default()
    Config.value["train_dataset"]["name"] = "Empty"
    Config.value["valid_dataset"]["name"] = "Empty"

    parser = argparse.ArgumentParser(
        description='Arg parsing for reconstructing clip with model',
    )

    parser.add_argument('--model', help='path pre trained model', required=True)
    parser.add_argument('--clip', help='path clip', required=True)
    parser.add_argument('--output_dir', help='output directory', required=True)
    parser.add_argument('--qp', help='value of qp', required=True)
    args = parser.parse_args()

    model_path = args.model
    clip_path = args.clip
    qp = args.qp
    output_path = args.output_dir

    model = torch.load(model_path).eval().cuda()

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    raw_path = Path(clip_path)
    raw_clip = Mp4Clip(raw_path)
    bitrate = f"qp={qp}"
    print(f"reconstructing clip {clip_path} with a quality setting {bitrate} with model loaded from {model_path}")

    blocky_clip = raw_clip.change_bitrate(f"{bitrate}", output_path / f"{raw_path.stem}_{bitrate}_no_postprocessing.mp4", deblocking=False, skip_if_exists=True)
    deblocked_clip = raw_clip.change_bitrate(f"{bitrate}", output_path / f"{raw_path.stem}_{bitrate}_dbf_sao.mp4", deblocking=True, skip_if_exists=True)

    images_path: Path = output_path / f"{raw_path.stem}_{bitrate}_reconstructed"
    func = ReconstructedImageSaver(images_path).save_images_in_grid
    reconstruct_clip(raw_clip, blocky_clip, deblocked_clip, model=model, func=func)

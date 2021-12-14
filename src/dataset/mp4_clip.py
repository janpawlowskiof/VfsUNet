import subprocess
from collections import namedtuple
from typing import Callable

import cv2
import ffmpeg
from pathlib import Path

import torch
from PIL import Image
from torchmetrics.functional import psnr
from tqdm import tqdm

from src.dataset.util.png_frame_saver import PngFrameSaver
from src.model.util.transforms import StreamingTransformations

ProcessedImages = namedtuple("ProcessedImages", ["raw_image", "blocky_image", "deblocked_image", "final_image", "decoded_residue", "true_residue"])


class Mp4Clip:
    def __init__(self, path: Path):
        self.path = path

    def split(self, output_path: Path, frame_saver: PngFrameSaver):
        for index, frame in enumerate(self.as_numpy_arrays()):
            save_path = self._get_clip_path(output_path, index)
            frame_saver.save_frame(frame, save_path)

    def split_ffmpeg(self, output_path: Path):
        output_file_pattern = output_path / "%06d.png"
        print(f"output pattern is: {output_file_pattern.absolute()}")
        subprocess.run(['ffmpeg', "-y", "-i", str(self.path.absolute()), str(output_file_pattern.absolute())])

    def change_bitrate(self, bitrate, output_path: Path, skip_if_exists=True, deblocking=False, intra=True):
        output_path = output_path.with_suffix(".mp4")

        if output_path.exists() and skip_if_exists:
            return Mp4Clip(output_path)

        output_path.parent.mkdir(exist_ok=True, parents=True)
        x265_params = f'{bitrate}'
        if not deblocking:
            x265_params += ':--deblock=false:no-sao=1'
        if intra:
            x265_params += ":frame-threads=4:keyint=1:ref=1:no-open-gop=1:weightp=0:weightb=0:cutree=0:rc-lookahead=0:bframes=0:scenecut=0:b-adapt=0:repeat-headers=1"

        subprocess.run(
            ['ffmpeg', "-y", "-i", str(self.path.absolute()), "-c:v", "libx265", "-f", "mp4", "-x265-params", x265_params, str(output_path.absolute())]
            # ['ffmpeg', "-y", "-i", str(self.path.absolute()), "-c:v", "libx265", "-b:v", bitrate, "-f", "mp4", "-x265-params", x265_params, str(output_path.absolute())]
        )

        return Mp4Clip(output_path)

    # def compress_intra(self, output_path: Path):
    #     output_path.parent.mkdir(exist_ok=True, parents=True)
    #     ffmpeg_args = {'c:v': 'libx265', 'f': 'mp4', "deblock": "false"}
    #
    #     ffmpeg \
    #         .input(str(self.path)) \
    #         .output(str(output_path), **ffmpeg_args) \
    #          \
    #         .overwrite_output() \
    #         .run()
    #
    #     return Mp4Clip(output_path)

    def remove_spaces(self):
        new_name = self.path.name.replace(" ", "")
        new_path = self.path.parent / new_name
        self.path = self.path.rename(new_path)

    def __len__(self):
        cap = cv2.VideoCapture(str(self.path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return frame_count

    def as_numpy_arrays(self):
        cap = cv2.VideoCapture(str(self.path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for _ in range(frame_count):
            _, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield frame

    def as_pil_images(self):
        for frame in self.as_numpy_arrays():
            yield Image.fromarray(frame)

    def _get_clip_path(self, output_path: Path, index: int) -> Path:
        return output_path / f"{index:06}"

    def generate_residue(self, blocky_clip: "Mp4Clip", deblocked_clip: "Mp4Clip", model, transforms, func: Callable[[ProcessedImages], None], residue_multiplier=3):
        with torch.no_grad():
            for raw_frame, blocky_frame, deblocked_frame in tqdm(zip(self.as_pil_images(), blocky_clip.as_pil_images(), deblocked_clip.as_pil_images()), total=len(self)):
                raw_frame = transforms(raw_frame).cuda()
                blocky_frame = transforms(blocky_frame).cuda()
                deblocked_frame = transforms(deblocked_frame).cuda()

                prediction = model(blocky_frame.unsqueeze(0)).squeeze().clip(-1, 1)

                predicted_residue = StreamingTransformations.unnormalize((prediction - blocky_frame) * residue_multiplier) * 255.0
                true_residue = StreamingTransformations.unnormalize(raw_frame - blocky_frame) * 255.0

                prediction = StreamingTransformations.unnormalize(prediction) * 255.0
                blocky_frame = StreamingTransformations.unnormalize(blocky_frame) * 255.0
                deblocked_frame = StreamingTransformations.unnormalize(deblocked_frame) * 255.0
                raw_image = StreamingTransformations.unnormalize(raw_frame) * 255.0

                processed_images = ProcessedImages(
                    raw_image=raw_image, final_image=prediction, blocky_image=blocky_frame, deblocked_image=deblocked_frame, true_residue=true_residue, decoded_residue=predicted_residue
                )
                func(processed_images)

    def calculate_metric(self, metric_func: Callable, raw_clip: "Mp4Clip" = None):
        if raw_clip is None:
            raw_clip = self.get_raw()
        total = torch.tensor(0, dtype=torch.float64)
        for frame, raw_frame in tqdm(zip(self.as_numpy_arrays(), raw_clip.as_numpy_arrays()), total=len(self)):
            frame = torch.from_numpy(frame).cuda()
            raw_frame = torch.from_numpy(raw_frame).cuda()
            frame = torch.permute(frame, [2, 0, 1]) / 255 * 2 - 1
            raw_frame = torch.permute(raw_frame, [2, 0, 1]) / 255 * 2 - 1
            val = metric_func(frame.unsqueeze(0), raw_frame.unsqueeze(0), data_range=2).cpu()
            if torch.isinf(val):
                print(frame.min(), frame.max(), raw_frame.min(), raw_frame.max(), (raw_frame - frame).sum())
            else:
                total += val
            assert not torch.isinf(total)
        total /= len(self)
        return total

    def calculate_metric_with_model(self, model, metric_func: Callable):
        with torch.no_grad():
            raw_clip = self.get_raw()
            total = torch.tensor(0, dtype=torch.float64)
            for frame, raw_frame in tqdm(zip(self.as_numpy_arrays(), raw_clip.as_numpy_arrays()), total=len(self)):
                frame = torch.from_numpy(frame).cuda()
                raw_frame = torch.from_numpy(raw_frame).cuda()
                frame = torch.permute(frame, [2, 0, 1]) / 255 * 2 - 1
                raw_frame = torch.permute(raw_frame, [2, 0, 1]) / 255 * 2 - 1
                pred_frame = model(frame.unsqueeze(0))
                val = metric_func(pred_frame, raw_frame.unsqueeze(0), data_range=2).cpu()
                if torch.isinf(val):
                    print(frame.min(), frame.max(), raw_frame.min(), raw_frame.max(), (raw_frame - frame).sum())
                else:
                    total += val
                assert not torch.isinf(total)
            total /= len(self)
            return total

    def get_raw(self):
        root_path = self.path.parent.parent
        raw_path = root_path / "raw" / self.path.name
        return Mp4Clip(raw_path)

    def get_bitrate(self):
        from videoprops import get_video_properties
        props = get_video_properties(str(self.path))
        return int(props['bit_rate'])

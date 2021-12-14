import glob
import os
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchmetrics.functional import psnr
from tqdm import tqdm

from src.model.util.transforms import StreamingTransformations


@dataclass
class CompressionDatasetEntry:
    raw_path: Path
    compressed_path: Path
    deblocked_path: Path


class CompressionDataset(Dataset):

    @staticmethod
    def from_config(config):
        raw_directory = Path(config["raw_directory"])
        compressed_directory = Path(config["compressed_directory"])
        deblocked_directory = Path(config["deblocked_directory"])
        tfms = StreamingTransformations.from_config(config["transformations"])
        dataset = CompressionDataset(raw_directory=raw_directory, compressed_directory=compressed_directory, deblocked_directory=deblocked_directory, transform=tfms)
        if "subset" in config:
            return Subset(dataset, range(config["subset"]))
        else:
            return dataset

    def __init__(self, raw_directory: Path, compressed_directory: Path, deblocked_directory: Path, transform):
        print(f"loading dataset from {raw_directory} on {socket.gethostname()}")
        self.transform = transform
        self.entries: List[CompressionDatasetEntry] = []

        for raw_image_path in glob.glob(f"{raw_directory}/*/*.png"):
            raw_image_path = Path(raw_image_path)
            compressed_image_path = compressed_directory / raw_image_path.relative_to(raw_directory)
            deblocked_image_path = deblocked_directory / raw_image_path.relative_to(raw_directory)
            if raw_image_path.exists() and compressed_image_path.exists() and deblocked_image_path.exists():
                self.entries.append(CompressionDatasetEntry(raw_image_path, compressed_image_path, deblocked_image_path))
            else:
                print(f"path {raw_image_path} has no compressed version under {compressed_image_path} or {deblocked_image_path}")

    def __getitem__(self, n):
        raw_image = self.get_image_from_path(self.entries[n].raw_path)
        compressed_image = self.get_image_from_path(self.entries[n].compressed_path)
        deblocked_image = self.get_image_from_path(self.entries[n].deblocked_path)
        residue_image = raw_image - compressed_image

        data = {
            "raw": raw_image,
            "compressed": compressed_image,
            "residue": residue_image,
            "deblocked": deblocked_image
        }

        return data

    def get_image_from_path(self, path: Path):
        image = Image.open(path)
        image = self.transform(image)
        return image

    def calculate_psnr(self):
        total = torch.tensor(0, dtype=torch.float32)
        for item in tqdm(self, total=len(self)):
            raw = item["raw"]
            compressed = item["compressed"]
            total += psnr(compressed, raw, data_range=2.0)
        total /= len(self)
        return total

    def __len__(self):
        return len(self.entries)

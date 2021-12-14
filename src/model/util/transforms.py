import numpy as np
import torch
from torchvision import transforms

from src.config import Config


class StreamingTransformations:
    mean = np.array([0.5] * 3)
    std = np.array([0.5] * 3)

    @staticmethod
    def from_config(config):
        return transforms.Compose([
            StreamingTransformations.from_name(name)
            for name in config
        ])

    @staticmethod
    def from_name(name):
        if name == "normalize":
            return transforms.Normalize(StreamingTransformations.mean.tolist(), StreamingTransformations.std.tolist())
        elif name == "totensor":
            return transforms.ToTensor()
        elif name == "resize":
            return transforms.Resize((Config["model"]["height"], (Config["model"]["width"])))
        raise RuntimeError(f"unknown transform name {name}")

    @staticmethod
    def normalize(images: torch.Tensor):
        normalize_transform = transforms.Normalize(StreamingTransformations.mean.tolist(), StreamingTransformations.std.tolist())
        return normalize_transform(images)

    @staticmethod
    def unnormalize(images: torch.Tensor):
        unnormalize_transform = transforms.Normalize((-StreamingTransformations.mean / StreamingTransformations.std).tolist(), (1.0 / StreamingTransformations.std).tolist())
        return unnormalize_transform(images)

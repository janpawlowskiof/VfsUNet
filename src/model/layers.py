import torch.nn as nn

from src.config import Config


def create_activation():
    relu_config = Config["relu"]
    if relu_config["name"] == "ReLU":
        return nn.ReLU(inplace=True)
    elif relu_config["name"] == "LeakyReLU":
        slope = relu_config["slope"]
        return nn.LeakyReLU(slope, inplace=True)
    raise RuntimeError(f"No relu named {relu_config['name']}")


def create_BatchNorm2d(out_channels):
    return nn.BatchNorm2d(out_channels, eps=0.01, momentum=0.001)

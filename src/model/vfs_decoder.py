from typing import List

import torch
import torch.nn as nn

from src.model.layers import create_BatchNorm2d, create_activation
from src.model.vfs_encoder import VfsEncoder


class VariableFilterSizeUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        out_channels_per_conv = out_channels // 2
        self.conv3x3 = nn.Conv2d(in_channels, out_channels_per_conv * 4, kernel_size=(3, 3), padding=1, bias=False, padding_mode='reflect')
        self.conv5x5 = nn.Conv2d(in_channels, out_channels_per_conv * 4, kernel_size=(5, 5), padding=2, bias=False, padding_mode='reflect')
        self.ps = nn.PixelShuffle(2)
        self.bn = create_BatchNorm2d(out_channels)
        self.relu = create_activation()

    def forward(self, x):
        x0 = self.conv3x3(x)
        x1 = self.conv5x5(x)

        x0 = self.ps(x0)
        x1 = self.ps(x1)

        x = torch.cat([x0, x1], dim=1)
        x = self.bn(x)
        x = self.relu(x)

        return x


class VfsDecoder(nn.Module):
    def __init__(self, out_channels_before_cat=8):
        super().__init__()
        encoder_channels = list(reversed(VfsEncoder.channels))

        self.out_channels = [24, 16, 12, out_channels_before_cat]
        self.in_channels = (
            [encoder_channels[0]]
            + [
                out_channels + encoder_channels
                for out_channels, encoder_channels in zip(self.out_channels, encoder_channels[1:])
            ]
        )

        self.stages = nn.ModuleList(
            [
                VariableFilterSizeUpBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                )
                for in_channels, out_channels in zip(self.in_channels, self.out_channels)
            ]
        )

    def forward(self, encoder_output: List[torch.Tensor]):
        encoder_output = list(reversed(encoder_output))
        x = encoder_output[0]
        for stage, memory_block in zip(self.stages, encoder_output[1:]):
            x = stage(x)
            x = torch.cat([x, memory_block], dim=1)
        return x

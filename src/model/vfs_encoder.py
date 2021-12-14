import torch
from torch import nn

from src.model.layers import create_BatchNorm2d, create_activation


class VariableFilterSizeDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        out_channels_per_conv = out_channels // 2
        self.conv3x3 = nn.Conv2d(in_channels, out_channels_per_conv, kernel_size=(3, 3), padding=1, stride=2, bias=False, padding_mode='reflect')
        self.conv5x5 = nn.Conv2d(in_channels, out_channels_per_conv, kernel_size=(5, 5), padding=2, stride=2, bias=False, padding_mode='reflect')
        self.bn = create_BatchNorm2d(out_channels)
        self.relu = create_activation()

    def forward(self, x):
        x0 = self.conv3x3(x)
        x1 = self.conv5x5(x)
        x = torch.cat([x0, x1], dim=1)
        x = self.bn(x)
        x = self.relu(x)
        return x


class VfsEncoder(nn.Module):
    channels = [3, 12, 16, 24, 48]

    def __init__(self, depth=4):
        super().__init__()

        self.stages = nn.ModuleList(
            [
                VariableFilterSizeDownBlock(self.channels[i - 1], self.channels[i])
                for i in range(1, depth + 1)
            ]
        )

    def forward(self, x):
        outputs = [x]
        for stage in self.stages:
            layer_output = stage(outputs[-1])
            outputs.append(layer_output)
        return outputs

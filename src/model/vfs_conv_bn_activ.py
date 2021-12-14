import torch
import torch.nn as nn

from src.model.layers import create_activation


class VfsConvBnActiv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv5x5 = nn.Conv2d(in_channels, out_channels//2, kernel_size=(5, 5), padding=2, padding_mode='reflect', bias=False)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels//2, kernel_size=(3, 3), padding=1, padding_mode='reflect', bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activ = create_activation()

    def forward(self, x):
        x = torch.cat([self.conv5x5(x), self.conv3x3(x)], dim=1)
        x = self.bn(x)
        x = self.activ(x)
        return x

import torch
import torch.nn as nn


class VrCnnBn(nn.Module):
    def __init__(self):
        super().__init__()

        def block(**kwargs):
            return nn.Sequential(
                nn.Conv2d(**kwargs),
                nn.BatchNorm2d(kwargs.get("out_channels")),
                nn.ReLU()
            )

        self.conv1 = block(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv2_1 = block(in_channels=64, out_channels=16, kernel_size=5, padding=2)
        self.conv2_2 = block(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.conv3_1 = block(in_channels=32 + 16, out_channels=16, kernel_size=3, padding=1)
        self.conv3_2 = block(in_channels=32 + 16, out_channels=32, kernel_size=1, padding=0)
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32 + 16, out_channels=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1)
        )

    def forward(self, x):
        return x + self.calculate_residue(x)

    def calculate_residue(self, x):
        x = self.conv1(x)
        x = torch.cat([self.conv2_1(x), self.conv2_2(x)], dim=1)
        x = torch.cat([self.conv3_1(x), self.conv3_2(x)], dim=1)
        x = self.conv4(x)
        return x

import torch.nn as nn

from src.model.vfs_conv_bn_activ import VfsConvBnActiv
from src.model.vfs_decoder import VfsDecoder
from src.model.vfs_encoder import VfsEncoder


class VfsUNet(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()

        self.encoder = VfsEncoder()
        decoder_output_channels_before_cat = 8
        decoder_output_channels = decoder_output_channels_before_cat + in_channels
        self.decoder = VfsDecoder(decoder_output_channels_before_cat)

        self.out = nn.Sequential(
            VfsConvBnActiv(decoder_output_channels, out_channels * 2),
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=(1, 1)),
            nn.Tanh()
        )

    def forward(self, x):
        return x + self.calculate_residue(x)

    def calculate_residue(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.out(x)
        return x

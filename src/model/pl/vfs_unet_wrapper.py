from src.model.pl.streaming_model_abc import StreamingModelABC
from src.model.vfs_unet import VfsUNet


class VfsUNetWrapper(StreamingModelABC):
    def __init__(self):
        super().__init__()
        self.model = VfsUNet(
            in_channels=3,
            out_channels=3,
        )

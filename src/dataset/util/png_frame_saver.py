from pathlib import Path
import numpy as np
from PIL import Image


class PngFrameSaver:
    def __init__(self):
        self._format = ".png"

    def save_frame(self, frame: np.array, path: Path):
        image = Image.fromarray(frame)
        path = path.with_suffix(self._format)
        path.parent.mkdir(exist_ok=True, parents=True)
        image.save(str(path))

        
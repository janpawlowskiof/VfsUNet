from pathlib import Path
from typing import Dict

import yaml

import configs


class _Config:
    value: Dict = None
    path: Path = None

    @staticmethod
    def __getitem__(item):
        if Config.value is None:
            Config.load_default()
        return Config.value[item]

    @staticmethod
    def load_default():
        configs_root_path = Path(configs.__file__).parent
        Config.load(configs_root_path / "train_config.yaml")

    @staticmethod
    def load(path: Path):
        with path.open() as config_file:
            Config.path = path
            Config.value = yaml.load(config_file, Loader=yaml.FullLoader)


Config = _Config()

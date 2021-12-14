import wandb

from src.config import Config


def get_model():
    from src.model.pl.vfs_unet_wrapper import VfsUNetWrapper

    model = VfsUNetWrapper()
    if "load_model" in Config.value:
        load_id = Config["load_model"]
        artifact_dir = wandb.init().use_artifact(load_id, type="model").download()
        model = model.load_from_checkpoint(f"{artifact_dir}/model.ckpt", strict=False)
    return model

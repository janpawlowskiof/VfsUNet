import ray
import glob
from pathlib import Path

from src.dataset.mp4_clip import Mp4Clip
from src.ray import initialize_ray


def get_compress_clips_in_path_remotes(path: Path, bitrate: str, deblock: bool):
    deblock_name = "_deblocked_" if deblock else "_"
    return get_compress_clips_remote(path / "raw", path / f"ai_hevc{deblock_name}{bitrate}", bitrate=bitrate, deblock=deblock)


def get_compress_clips_remote(input_path: Path, output_path: Path, bitrate, deblock):
    print(f"{input_path}")
    remotes = []
    clip_paths = list(glob.glob(f"{input_path}/*.mp4")) + list(glob.glob(f"{input_path}/*.mkv"))
    for mp4_path in clip_paths:
        print(f"compressing clips in {mp4_path}")
        mp4_path = Path(mp4_path)
        clip = Mp4Clip(mp4_path)
        remotes.append(change_bitrate.remote(clip, bitrate, output_path / mp4_path.relative_to(input_path), deblock))
    return remotes


@ray.remote(num_cpus=4)
def change_bitrate(clip, bitrate, output_path, deblocking):
    clip.change_bitrate(bitrate, output_path, skip_if_exists=False, deblocking=deblocking, intra=True)


if __name__ == "__main__":
    initialize_ray()
    # ray.init()

    paths = [
        # Path("/mnt/nfs_svtai10-nvme1n1p1/jpawlowski/gta_v/train"),
        # Path("/mnt/nfs_svtai10-nvme1n1p1/jpawlowski/gta_v/valid")
        Path("/mnt/nfs_svtai08-nvme1n1p1/katana_zero/valid/"),
        Path("/mnt/nfs_svtai08-nvme1n1p1/katana_zero/train/"),
        Path("/mnt/nfs_svtai09-nvme1n1p1/jpawlowski/trackmania/valid"),
        Path("/mnt/nfs_svtai09-nvme1n1p1/jpawlowski/trackmania/train"),
    ]

    remotes = []
    for path in paths:
        for bitrate in [
            "qp=32",
            "qp=35",
            "qp=37",
            "qp=39",
        ]:
            for deblock in [
                True,
                False
            ]:
                remotes.extend(get_compress_clips_in_path_remotes(path, bitrate, deblock))

    ray.get(remotes)

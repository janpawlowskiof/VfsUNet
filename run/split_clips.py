import glob
from functools import partial
from pathlib import Path

import ray
from pqdm.threads import pqdm

from src.dataset.util.png_frame_saver import PngFrameSaver
from src.dataset.mp4_clip import Mp4Clip
from src.ray import initialize_ray

frame_saver = PngFrameSaver()


def split_gta_remote():
    # initialize_ray()
    ray.init()

    paths = [
        Path("/mnt/nfs_svtai10-nvme1n1p1/jpawlowski/gta_v/train"),
        Path("/mnt/nfs_svtai10-nvme1n1p1/jpawlowski/gta_v/valid")
    ]

    folders = [
        # "raw",
        # "ai_hevc_qp=32",
        # "ai_hevc_qp=35",
        # "ai_hevc_qp=37",
        "ai_hevc_qp=37",
        # "ai_hevc_deblocked_qp=32",
        # "ai_hevc_deblocked_qp=35",
        # "ai_hevc_deblocked_qp=37"
        "ai_hevc_deblocked_qp=37"
    ]

    remote_split_clips = ray.remote(split_clip)
    remotes = []
    for path in paths:
        for folder in folders:
            input_path = path / folder
            output_path = path / (folder + "_split_png")
            for mp4_path in glob.glob(f"{input_path}/*.mp4"):
                mp4_path = Path(mp4_path)
                split_path = output_path / mp4_path.stem
                split_path.mkdir(exist_ok=True, parents=True)
                print(split_path)
                remotes.append(remote_split_clips.remote(mp4_path, split_path))
    ray.get(remotes)


def split_katana():
    # initialize_ray()
    ray.init()

    paths = [
        Path("/mnt/nfs_svtai08-nvme1n1p1/katana_zero/train"),
        Path("/mnt/nfs_svtai08-nvme1n1p1/katana_zero/valid")
    ]

    folders = [
        # "raw",
        # "ai_hevc_qp=32",
        # "ai_hevc_qp=35",
        # "ai_hevc_qp=37",
        "ai_hevc_qp=39",
        # "ai_hevc_deblocked_qp=32",
        # "ai_hevc_deblocked_qp=35",
        # "ai_hevc_deblocked_qp=37",
        "ai_hevc_deblocked_qp=39"
    ]

    remote_split_clips = ray.remote(num_cpus=1)(split_clip)
    remotes = []
    for path in paths:
        for folder in folders:
            input_path = path / folder
            output_path = path / (folder + "_split_png")
            for mp4_path in glob.glob(f"{input_path}/*.mp4"):
                mp4_path = Path(mp4_path)
                split_path = output_path / mp4_path.stem
                split_path.mkdir(exist_ok=True, parents=True)
                print(split_path)
                remotes.append(remote_split_clips.remote(mp4_path, split_path))
    ray.get(remotes)


def split_trackmania():
    ray.init()

    paths = [
        Path("/mnt/nfs_svtai09-nvme1n1p1/jpawlowski/trackmania/train"),
        Path("/mnt/nfs_svtai09-nvme1n1p1/jpawlowski/trackmania/valid")
    ]

    folders = [
        # "raw",
        # "ai_hevc_qp=32",
        # "ai_hevc_qp=35",
        # "ai_hevc_qp=37",
        "ai_hevc_qp=37",
        "ai_hevc_qp=39",
        # "ai_hevc_deblocked_qp=32",
        # "ai_hevc_deblocked_qp=35",
        "ai_hevc_deblocked_qp=37",
        "ai_hevc_deblocked_qp=39",
    ]

    remote_split_clips = ray.remote(split_clip)
    remotes = []
    for path in paths:
        for folder in folders:
            input_path = path / folder
            output_path = path / (folder + "_split_png")
            for mp4_path in glob.glob(f"{input_path}/*.mp4"):
                mp4_path = Path(mp4_path)
                split_path = output_path / mp4_path.stem
                split_path.mkdir(exist_ok=True, parents=True)
                print(split_path)
                remotes.append(remote_split_clips.remote(mp4_path, split_path))
    ray.get(remotes)


# def split_clips_in_directory(clips_path, split_path):
#     pqdm(glob.glob(f"{clips_path}/*.mp4"), partial(split_clip, split_path=split_path), n_jobs=16)


def split_clip(mp4_path, split_path):
    print(f"splitting clip {mp4_path}")
    mp4_path = Path(mp4_path)
    clip = Mp4Clip(mp4_path)
    clip.remove_spaces()
    # clip.split_ffmpeg(split_path)
    clip.split(split_path, frame_saver)


if __name__ == "__main__":
    # split_trackmania()
    split_katana()
    # split_gta_remote()

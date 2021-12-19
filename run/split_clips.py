import glob
from pathlib import Path
import argparse

import ray

from src.dataset.util.png_frame_saver import PngFrameSaver
from src.dataset.mp4_clip import Mp4Clip

frame_saver = PngFrameSaver()


def split_clips_in_path(path: Path, folders):
    ray.init()

    remote_split_clips = ray.remote(split_clip)
    remotes = []
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


def split_clip(mp4_path, split_path):
    print(f"splitting clip {mp4_path}")
    mp4_path = Path(mp4_path)
    clip = Mp4Clip(mp4_path)
    clip.remove_spaces()
    clip.split(split_path, frame_saver)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Arg parsing for splitting clips',
    )

    parser.add_argument('--root', help='path in which clips will be split', required=True)
    parser.add_argument('--folders', nargs='+', help='Folders in root directory that will be split', required=True)
    args = parser.parse_args()

    root = Path(args.root)
    folders = args.folders
    print(f"splitting folders {folders} in root directory of {root}")

    split_clips_in_path(root, folders)

from pathlib import Path
from typing import Callable

from src.dataset.mp4_clip import Mp4Clip


def calculate_average_bitrate_in_directory(path: Path):
    clips = [
        Mp4Clip(clip_path)
        for clip_path in
        path.glob("*.mp4")
    ]

    total_len = sum(len(clip) for clip in clips)
    summed_weighted_bitrates = sum(clip.get_bitrate() * len(clip) for clip in clips)
    average_bitrate = summed_weighted_bitrates / total_len
    return average_bitrate


def calculate_average_metric_in_directory(path: Path, metric_func: Callable):
    clips = [
        Mp4Clip(clip_path)
        for clip_path in
        path.glob("*.mp4")
    ]

    total_len = sum(len(clip) for clip in clips)
    summed_weighted_metrics = sum(clip.calculate_metric(metric_func=metric_func) * len(clip) for clip in clips)
    average_psnr = summed_weighted_metrics / total_len
    return average_psnr


def calculate_average_metric_in_directory_for_model(path: Path, model, metric_func: Callable):
    clips = [
        Mp4Clip(clip_path)
        for clip_path in
        path.glob("*.mp4")
    ]

    total_len = sum(len(clip) for clip in clips)
    summed_weighted_metrics = sum(clip.calculate_metric_with_model(model, metric_func) * len(clip) for clip in clips)
    average_psnr = summed_weighted_metrics / total_len
    return average_psnr

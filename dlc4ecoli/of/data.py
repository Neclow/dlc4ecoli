from glob import glob
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import supervision as sv
import torch
import torchvision.transforms.functional as vF

from torchvision import transforms

from .features import STATS
from ..config import CAMERA_LABELS
from ..utils.signal import smooth_signal


def load_n_frames(video_path, start=0, end=None):
    # Output shape: T C W H
    return torch.stack(
        [
            vF.to_tensor(cv2.cvtColor(_, cv2.COLOR_BGR2RGB))
            for _ in sv.get_video_frames_generator(video_path, start=start, end=end)
        ],
        dim=0,
    )


def transform(batch, size):
    transform_list = [
        transforms.Resize(size=size, antialias=False),
        transforms.Normalize(mean=0.5, std=0.5),
    ]

    the_transforms = transforms.Compose(transform_list)

    batch = the_transforms(batch)

    return batch


def build_summary(data_folder, agg="mean"):
    if agg == "mean":
        agg_func = np.mean
    elif agg == "median":
        agg_func = np.median
    else:
        raise ValueError("agg must be mean or median")

    files = glob(f"{data_folder}/*/*.pt")

    data = {}

    for f in files:
        file_data = torch.load(f, weights_only=False).cpu().numpy()

        key = Path(f).stem

        data[key] = {
            stat: smooth_signal(file_data[:, i], mode="medfilt")
            for i, stat in enumerate(STATS)
        }

        data[key]["cv"] = data[key]["std"] / (data[key]["mean"] + np.finfo(float).eps)

    of_agg = {}

    for key in sorted(data.keys()):
        of_agg[key] = {}

        for stat in data[key].keys():
            # NOTE: std was used in older versions
            if stat == "std":
                of_agg[key][f"{agg}_var"] = agg_func(data[key][stat]) ** 2
            else:
                of_agg[key][f"{agg}_{stat}"] = agg_func(data[key][stat])

    of_agg = pd.DataFrame.from_dict(of_agg, orient="index").reset_index(names="video")

    of_agg["camera"] = of_agg.video.apply(lambda x: x.split("_")[1])

    of_agg["group"] = of_agg.camera.map(
        dict(map(lambda x: tuple(x.split(": ")), CAMERA_LABELS))
    )

    of_agg["video_number"] = of_agg.video.apply(
        lambda x: int(x.split("_")[0].replace("GX", ""))
    )

    of_agg["Day"] = (
        of_agg.video_number.sub(of_agg.groupby("camera").video_number.transform("min"))
        .mul(0.5)
        .add(1)
    )

    of_agg.sort_values(by="camera", inplace=True)

    return of_agg

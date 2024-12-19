import cv2
import supervision as sv
import torch
import torchvision.transforms.functional as vF

from torchvision import transforms


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

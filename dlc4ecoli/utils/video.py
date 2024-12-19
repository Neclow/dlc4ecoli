from collections import namedtuple

import cv2


VideoInfo = namedtuple("VideoInfo", ["height", "width", "fps", "total_frames"])


# Adapted from https://github.com/roboflow/supervision/blob/a9aad6727c9a0302199830c700fa169c5674d7a2/supervision/utils/video.py#L13
def get_video_info(video_path):
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        raise OSError(f"Could not open video at {video_path}")

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # removed int for fps
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    video.release()

    return VideoInfo(height=height, width=width, fps=fps, total_frames=total_frames)

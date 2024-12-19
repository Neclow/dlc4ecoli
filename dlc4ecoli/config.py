"""Constants."""

from typing import Final

import numpy as np

# Camera labels
CAMERA_LABELS = ["1: E. coli", "2: Control"]

# Food source coordinates in the videos
FOOD_AREAS: Final = {
    1: np.array([[550.0, 400.0], [425.0, 360.0], [700.0, 60.0], [825.0, 100.0]]),
    2: np.array([[550.0, 220.0], [650.0, 90.0], [1100.0, 200.0], [1000.0, 350.0]]),
}

# Video sample rate
FPS: Final = 30  # Hz

# Max video duration
MAX_DURATION: Final = 360  # s

# Minimum DLC acceptable likelihood
# Recommended: 0.8 from https://deeplabcut.github.io/DeepLabCut/docs/maDLC_UserGuide.html#plotting-results
MIN_LIKELIHOOD: Final = 0.9

# Threshold for number of NaNs
NAN_PER_VIDEO_THRESHOLD = 0.25

# P-value plot annotations
PVALUE_MAP = [[1e-3, "***"], [1e-2, "**"], [0.05, "*"], [1, "ns"]]

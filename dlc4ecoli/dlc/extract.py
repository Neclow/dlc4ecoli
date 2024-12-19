"""
Extract features from DeepLabCut outputs
"""

import os

from argparse import ArgumentParser
from glob import glob
from pathlib import Path

import pandas as pd

from .data import build_summary, prepare_data
from .features import get_body_area_change, get_length, get_time_near_source, get_travel
from ..config import FOOD_AREAS
from ..utils.signal import filter_outliers

pd.options.mode.copy_on_write = True


def parse_args():
    """Parse options for feature extraction."""
    parser = ArgumentParser(
        description="Arguments for DLC feature extraction",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to data (.csv outputs from DeepLabCut)",
    )
    return parser.parse_args()


def extract_features(dlc_files):
    all_travel = {}

    all_head_to_tail = {}

    all_near_food = {}

    all_delta_areas = {}

    for i, file in enumerate(dlc_files):
        print(f"Current file ({i+1:02d}/{len(dlc_files)}): {os.path.basename(file)}")

        # Get the code name for the current file
        code = Path(file).stem.split("DLC")[0]

        if code in all_travel:
            continue

        # Load DLC predicitons
        df_raw = pd.read_csv(file, header=[1, 2, 3], index_col=0)

        # Remove low-likelihood points and interpolate
        pos_df, _ = prepare_data(df_raw)

        # Head-to-saddle distnace
        head_to_saddle = get_length(pos_df, "head", "saddle")

        all_head_to_tail[code] = head_to_saddle.mean()

        # Distance travelled
        travel = get_travel(pos_df)

        all_travel[code] = filter_outliers(
            travel.div(head_to_saddle), max_zscore=3
        ).mean()

        # % time near food
        near_food = get_time_near_source(pos_df, FOOD_AREAS[int(code[-1])])

        all_near_food[code] = near_food.mean()

        # Delta area
        delta_areas = get_body_area_change(pos_df)

        all_delta_areas[code] = filter_outliers(
            delta_areas.div(head_to_saddle), max_zscore=3
        ).mean()

    return all_near_food, all_travel, all_near_food, all_delta_areas


def main():
    args = parse_args()

    dlc_files = sorted(glob(f"{args.data_path}/*.csv"))

    print(f"Found {len(dlc_files)} files.")

    feature_dicts = extract_features(dlc_files)

    feature_keys = ["h2t", "travel", "near_food", "delta_area"]

    # Make summaries
    for feature_key, feature_dict in zip(feature_keys, feature_dicts):
        build_summary(feature_dict, feature_key).to_csv(
            f"data/summary_{feature_key}.csv", index=False
        )


if __name__ == "__main__":
    main()

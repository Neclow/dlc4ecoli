# pylint: disable=invalid-name, line-too-long, unsubscriptable-object
"""
Data cleaning functions: data filtering, smoothing and aligning.

Some functions are inspired by DLC2Kinematics (https://github.com/AdaptiveMotorControlLab/DLC2Kinematics)
"""
import pandas as pd

from ..config import FPS, MIN_LIKELIHOOD


def prepare_data(df, dropna=False, limit_direction="forward", limit=FPS // 2):
    """Clean data:
     - separate (x, y) from likelihood data
     - remove low-likelihood predictions
     - apply some interpolation for missing values

    Parameters
    ----------
    df : pandas.DataFrame
        Raw data

    Returns
    -------
    pos_df : pandas.DataFrame
        Bodypart positions at each frame from DLC
    lik_df : pandas.DataFrame
        Likelihood of DLC predictions for each bodypart and frame
    """
    # if smoothed:
    #     df = df.apply(smooth_signal, **kwargs)

    pos_df, lik_df = [
        x
        for _, x in df.groupby(
            df.columns.get_level_values("coords").str.contains("likelihood"), axis=1
        )
    ]

    # Replace low-likelihood data using linear interpolation
    low_lik_mask = pos_df.apply(
        lambda x: x.mask(
            lik_df.iloc[:, pos_df.columns.get_loc(x.name) // 2] < MIN_LIKELIHOOD
        )
    )

    if dropna:
        low_lik_mask.dropna(inplace=True)

    return (
        low_lik_mask.interpolate(limit_direction=limit_direction, limit=limit),
        lik_df,
    )


def build_summary(data, value_name):
    summary_df = (
        pd.concat(data, axis=1)
        .reset_index(names="individuals")
        .melt(id_vars="individuals", value_name=value_name, var_name="video")
    )

    summary_df["camera"] = summary_df["video"].apply(lambda x: x.split("_")[-1])

    summary_df["video_number"] = summary_df.video.apply(
        lambda x: int(x.split("_")[0].replace("GX", ""))
    )

    summary_df["Day"] = (
        summary_df.video_number.sub(
            summary_df.groupby("camera").video_number.transform("min")
        )
        .mul(0.5)
        .add(1)
    )

    return summary_df

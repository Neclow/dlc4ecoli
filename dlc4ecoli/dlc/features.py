import numpy as np

from ..utils.geom import is_point_inside_convex_quadrilateral, scale_rectangle, shoelace
from ..utils.signal import derive


def get_length(pos_df, a="head", b="saddle"):
    # Calculate the distance between two body parts
    a_and_b = pos_df.loc[:, pos_df.columns.get_level_values("bodyparts").isin((a, b))]

    # x = -x, y = -y for b coordinate
    a_and_b.iloc[:, 2::4] = -a_and_b.iloc[:, 2::4]
    a_and_b.iloc[:, 3::4] = -a_and_b.iloc[:, 3::4]

    a_to_b = (
        a_and_b.groupby(level=[0, 2], axis=1)
        .sum(min_count=2)  # head_x - saddle_x, head_y - saddle_y
        .pow(2)  # dx^2, dy^2
        .groupby(level=0, axis=1)
        .sum(min_count=2)  # dx^2 + dy^2
        .apply(np.sqrt)  # qrt
    )

    return a_to_b


def get_travel(pos_df):
    # Calculate center position
    pos_centre = (
        pos_df.apply(derive, order=0)  # smoothing
        .drop("tail", level=1, axis=1)  # remove tail
        .groupby(level=[0, 2], axis=1)
        .mean()  # get mean x and mean y
    )

    # Calculate frame-wise travelled distance
    travel = (
        pos_centre.diff(axis=0)  # dx, dy
        .pow(2)  # dx^2, dy^2
        .groupby(level=0, axis=1)
        .sum(min_count=2)  # dx^2 + dy^2
        .apply(np.sqrt)  # sqrt
    )

    return travel


def get_body_area_change(pos_df):
    # Calculate rate of change of body area
    delta_areas = (
        pos_df.groupby(level=0, axis=1)
        .apply(
            lambda x: x.iloc[:, :8].agg(
                lambda y: shoelace(y.values.reshape(-1, 2)), axis=1
            )
        )
        .apply(derive, order=1)
        .abs()
    )

    return delta_areas


def get_time_near_source(pos_df, source, factor=1.05):
    # Calculate % of time spent near source
    near_source = (
        pos_df.loc[:, pos_df.columns.get_level_values("bodyparts").isin(("head",))]
        .apply(derive, order=0)  # smoothing
        .groupby(level=[0, 1], axis=1)
        .agg(tuple)  # (x, y) tuple
        .applymap(
            lambda _: is_point_inside_convex_quadrilateral(
                _,
                scale_rectangle(source, factor),
            )
        )
    ).droplevel(1, axis=1)

    return near_source

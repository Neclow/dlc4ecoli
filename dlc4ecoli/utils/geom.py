"""
Miscellaneous geometric utility functions
"""

import numpy as np


def shoelace(coords):
    """Calculate polygon areas using the Shoelace algorithm
    Source: https://rosettacode.org/wiki/Shoelace_formula_for_polygonal_area#Python

    Parameters
    ----------
    coords : list of arrays
        List of (x, y) coordinates

    Returns
    -------
    area : numpy.ndarray
        Polygon at each frame
    """
    x_pos, y_pos = coords.copy().T

    # Shoelace can be imprecise if x,y have big offsets
    # So demean the coords
    x_pos -= x_pos.mean()
    y_pos -= y_pos.mean()

    i = np.arange(coords.shape[0])
    area = np.abs(0.5 * np.sum(x_pos[i - 1] * y_pos[i] - x_pos[i] * y_pos[i - 1]))

    return area


def is_point_inside_convex_quadrilateral(p, quad_points):
    """Check if a point is inside a convex quadrilateral

    Parameters
    ----------
    p : tuple
        (x, y) coordinates of a point
    quad_points : list of tuples of size 4
        (x, y) coordinates of a quadrilateral

    Returns
    -------
    inside_convex_quadrilateral : bool
        True if p is inside quad_points
    """
    assert len(quad_points) == 4
    # Area of quadrilateral
    quad_area = shoelace(quad_points)

    # Area of all triangles
    point_areas = np.zeros((4,))
    for i in range(4):
        point_areas[i] = shoelace(np.array([p, *quad_points[(i, (i + 1) % 4), :]]))

    inside_convex_quadrilateral = abs(quad_area - point_areas.sum()) < 1e-6
    return inside_convex_quadrilateral


def scale_rectangle(rectangle, factor):
    """Calculate the coordinates of a rectangle
    after enlargement by a certain factor

    Parameters
    ----------
    rectangle : numpy.ndarray of size (4, 2)
        (x, y) coordinates of the rectangle
    factor : float
        Enlargement factor

    Returns
    -------
    enlarged_rectangle : numpy.ndarray of size (4, 2)
        Enlarged rectangle
    """
    # Calculate the difference between each vertex and the center
    center = rectangle.mean(0)
    diff = rectangle - center

    # Enlarge the rectangle by scaling the differences
    enlarged_rectangle = center + diff * factor

    return enlarged_rectangle

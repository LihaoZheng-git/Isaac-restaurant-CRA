"""Pure NumPy geometry operations for LiDAR point clouds.

This module must not import Isaac Sim, Omni, or USD APIs.
"""

import numpy as np


def spherical_to_cartesian(
    azimuth_degrees: np.ndarray,
    elevation_degrees: np.ndarray,
    distance_meters: np.ndarray,
) -> np.ndarray:
    """Convert ISO 8855 spherical coordinates to XYZ coordinates.

    The output sensor coordinate frame is:

    - +X forward
    - +Y left
    - +Z up
    """

    azimuth_values = np.asarray(azimuth_degrees).reshape(-1)
    elevation_values = np.asarray(elevation_degrees).reshape(-1)
    distance_values = np.asarray(distance_meters).reshape(-1)

    if not (
        azimuth_values.shape
        == elevation_values.shape
        == distance_values.shape
    ):
        raise ValueError(
            "Azimuth, elevation, and distance arrays must have "
            "matching shapes."
        )

    azimuth_radians = np.deg2rad(azimuth_values)
    elevation_radians = np.deg2rad(elevation_values)

    horizontal_distance = (
        distance_values * np.cos(elevation_radians)
    )

    x_values = horizontal_distance * np.cos(azimuth_radians)
    y_values = horizontal_distance * np.sin(azimuth_radians)
    z_values = distance_values * np.sin(elevation_radians)

    return np.column_stack(
        (x_values, y_values, z_values)
    ).astype(np.float32, copy=False)


def filter_valid_points(points: np.ndarray) -> np.ndarray:
    """Remove non-finite and sensor-origin points."""

    point_array = np.asarray(points, dtype=np.float32)

    if point_array.ndim != 2 or point_array.shape[1] != 3:
        raise ValueError(
            "LiDAR points must have shape (N, 3), "
            f"received {point_array.shape}."
        )

    finite_mask = np.isfinite(point_array).all(axis=1)

    nonzero_mask = (
        np.linalg.norm(point_array, axis=1) > 0.0
    )

    return np.ascontiguousarray(
        point_array[finite_mask & nonzero_mask],
        dtype=np.float32,
    )
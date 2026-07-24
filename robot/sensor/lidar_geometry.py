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


def transform_points(
    points_source_frame: np.ndarray,
    target_from_source_transform: np.ndarray,
) -> np.ndarray:
    """Transform an XYZ point cloud between coordinate frames.

    The transform follows the column-vector convention:

        point_target = target_from_source_transform @ point_source

    For the LiDAR integration, this will be used as:

        points_robot = transform_points(
            points_sensor,
            robot_from_sensor_transform,
        )

    Args:
        points_source_frame:
            Point cloud with shape ``(N, 3)``.
        target_from_source_transform:
            Rigid homogeneous transformation matrix with shape ``(4, 4)``.

    Returns:
        A contiguous ``float32`` point cloud with shape ``(N, 3)``
        expressed in the target coordinate frame.
    """

    points = np.asarray(
        points_source_frame,
        dtype=np.float64,
    )

    transform = np.asarray(
        target_from_source_transform,
        dtype=np.float64,
    )

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            "Source points must have shape (N, 3), "
            f"received {points.shape}."
        )

    if transform.shape != (4, 4):
        raise ValueError(
            "The coordinate transform must have shape (4, 4), "
            f"received {transform.shape}."
        )

    if not np.isfinite(points).all():
        raise ValueError(
            "Source points must contain only finite values."
        )

    if not np.isfinite(transform).all():
        raise ValueError(
            "The coordinate transform must contain only finite values."
        )

    expected_last_row = np.array(
        [0.0, 0.0, 0.0, 1.0],
        dtype=np.float64,
    )

    if not np.allclose(
        transform[3],
        expected_last_row,
        atol=1e-8,
        rtol=0.0,
    ):
        raise ValueError(
            "The coordinate transform must be a homogeneous matrix "
            "with final row [0, 0, 0, 1]."
        )

    rotation = transform[:3, :3]
    translation = transform[:3, 3]

    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3),
        atol=1e-6,
        rtol=1e-6,
    ):
        raise ValueError(
            "The transform rotation must be orthonormal."
        )

    rotation_determinant = float(
        np.linalg.det(rotation)
    )

    if not np.isclose(
        rotation_determinant,
        1.0,
        atol=1e-6,
        rtol=1e-6,
    ):
        raise ValueError(
            "The transform rotation must have determinant +1."
        )

    transformed_points = (
        points @ rotation.T
        + translation
    )

    return np.ascontiguousarray(
        transformed_points,
        dtype=np.float32,
    )
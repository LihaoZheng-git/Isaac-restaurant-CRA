"""Pure NumPy obstacle extraction from robot-frame point clouds.

Input points must be expressed in the robot base frame:

    +X forward
    +Y left
    +Z upward

The output contains two-dimensional obstacle positions compatible with
the CBF safety filter.
"""

from dataclasses import dataclass
import math

import numpy as np

from robot.sensor.lidar_geometry import filter_valid_points


@dataclass(frozen=True)
class ObstacleDetectionConfig:
    """Configuration for geometric obstacle-point extraction."""

    minimum_height: float
    maximum_height: float
    minimum_range: float = 0.05
    maximum_range: float = 8.0
    forward_only: bool = True
    angular_bin_size_degrees: float | None = 2.0

    def __post_init__(self) -> None:
        finite_values = {
            "minimum_height": self.minimum_height,
            "maximum_height": self.maximum_height,
            "minimum_range": self.minimum_range,
            "maximum_range": self.maximum_range,
        }

        for name, value in finite_values.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")

        if self.minimum_height >= self.maximum_height:
            raise ValueError(
                "minimum_height must be smaller than maximum_height."
            )

        if self.minimum_range < 0.0:
            raise ValueError("minimum_range cannot be negative.")

        if self.maximum_range <= self.minimum_range:
            raise ValueError(
                "maximum_range must be greater than minimum_range."
            )

        if not isinstance(self.forward_only, bool):
            raise TypeError("forward_only must be a boolean.")

        if self.angular_bin_size_degrees is not None:
            bin_size = self.angular_bin_size_degrees

            if not math.isfinite(bin_size):
                raise ValueError(
                    "angular_bin_size_degrees must be finite."
                )

            if bin_size <= 0.0 or bin_size > 360.0:
                raise ValueError(
                    "angular_bin_size_degrees must be greater than "
                    "zero and no greater than 360."
                )


def detect_obstacle_points(
    points_robot_frame: np.ndarray,
    config: ObstacleDetectionConfig,
) -> np.ndarray:
    """Extract CBF-relevant obstacle points from a point cloud.

    Processing steps:

    1. Remove non-finite and sensor-origin points.
    2. Retain points inside the configured height interval.
    3. Retain points inside the configured horizontal range.
    4. Optionally remove points behind the robot.
    5. Optionally retain the nearest point in each angular bin.

    Args:
        points_robot_frame:
            Point cloud with shape ``(N, 3)`` in the robot base frame.
        config:
            Geometric filtering and angular-reduction parameters.

    Returns:
        A contiguous ``float32`` array with shape ``(M, 2)`` containing
        obstacle ``[x, y]`` positions in the robot base frame.
    """

    points = filter_valid_points(points_robot_frame)

    if points.shape[0] == 0:
        return _empty_obstacle_points()

    height_mask = (
        (points[:, 2] >= config.minimum_height)
        & (points[:, 2] <= config.maximum_height)
    )

    points = points[height_mask]

    if points.shape[0] == 0:
        return _empty_obstacle_points()

    points_xy = points[:, :2]

    horizontal_distances = np.linalg.norm(
        points_xy,
        axis=1,
    )

    range_mask = (
        (horizontal_distances >= config.minimum_range)
        & (horizontal_distances <= config.maximum_range)
    )

    if config.forward_only:
        range_mask &= points_xy[:, 0] >= 0.0

    points_xy = points_xy[range_mask]

    if points_xy.shape[0] == 0:
        return _empty_obstacle_points()

    if config.angular_bin_size_degrees is not None:
        points_xy = _keep_nearest_per_angular_bin(
            points_xy,
            bin_size_degrees=(
                config.angular_bin_size_degrees
            ),
        )

    return np.ascontiguousarray(
        points_xy,
        dtype=np.float32,
    )


def _keep_nearest_per_angular_bin(
    points_xy: np.ndarray,
    *,
    bin_size_degrees: float,
) -> np.ndarray:
    """Keep the nearest point in each horizontal angular sector."""

    if points_xy.shape[0] <= 1:
        return np.ascontiguousarray(
            points_xy,
            dtype=np.float32,
        )

    angles = np.arctan2(
        points_xy[:, 1],
        points_xy[:, 0],
    )

    bin_width_radians = math.radians(
        bin_size_degrees
    )

    bin_indices = np.floor(
        (angles + math.pi) / bin_width_radians
    ).astype(np.int64)

    distance_squared = np.einsum(
        "ij,ij->i",
        points_xy,
        points_xy,
    )

    # Sort primarily by angular bin and secondarily by distance.
    sorted_indices = np.lexsort(
        (distance_squared, bin_indices)
    )

    sorted_bins = bin_indices[sorted_indices]

    nearest_mask = np.empty(
        sorted_indices.shape[0],
        dtype=bool,
    )
    nearest_mask[0] = True
    nearest_mask[1:] = (
        sorted_bins[1:] != sorted_bins[:-1]
    )

    nearest_indices = sorted_indices[nearest_mask]

    return np.ascontiguousarray(
        points_xy[nearest_indices],
        dtype=np.float32,
    )


def _empty_obstacle_points() -> np.ndarray:
    """Create a correctly shaped empty CBF obstacle array."""

    return np.empty(
        (0, 2),
        dtype=np.float32,
    )
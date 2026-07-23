import numpy as np
import pytest

from robot.sensor.lidar_geometry import (
    filter_valid_points,
    spherical_to_cartesian,
)


def test_spherical_to_cartesian_horizontal_directions() -> None:
    points = spherical_to_cartesian(
        azimuth_degrees=np.array([0.0, 90.0, -90.0, 180.0]),
        elevation_degrees=np.zeros(4),
        distance_meters=np.ones(4),
    )

    expected = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(points, expected, atol=1e-6)


def test_spherical_to_cartesian_upward_direction() -> None:
    points = spherical_to_cartesian(
        azimuth_degrees=np.array([0.0]),
        elevation_degrees=np.array([90.0]),
        distance_meters=np.array([2.0]),
    )

    np.testing.assert_allclose(
        points,
        np.array([[0.0, 0.0, 2.0]], dtype=np.float32),
        atol=1e-6,
    )


def test_filter_valid_points() -> None:
    points = np.array(
        [
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0],
            [np.nan, 1.0, 2.0],
            [1.0, np.inf, 2.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    filtered = filter_valid_points(points)

    expected = np.array(
        [
            [1.0, 2.0, 3.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    np.testing.assert_array_equal(filtered, expected)
    assert filtered.flags.c_contiguous


def test_filter_valid_points_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError):
        filter_valid_points(np.zeros((4, 2)))
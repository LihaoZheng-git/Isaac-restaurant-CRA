import numpy as np
import pytest

from robot.sensor.lidar_geometry import (
    filter_valid_points,
    spherical_to_cartesian,
    transform_points,
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



def test_transform_points_identity() -> None:
    points = np.array(
        [
            [1.0, 2.0, 3.0],
            [-1.0, 0.5, 2.0],
        ],
        dtype=np.float32,
    )

    transformed = transform_points(
        points,
        np.eye(4),
    )

    np.testing.assert_allclose(
        transformed,
        points,
        atol=1e-6,
    )

    assert transformed.dtype == np.float32
    assert transformed.flags.c_contiguous


def test_transform_points_translation() -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
        ],
        dtype=np.float32,
    )

    transform = np.eye(4)
    transform[:3, 3] = [1.0, -2.0, 0.5]

    transformed = transform_points(
        points,
        transform,
    )

    expected = np.array(
        [
            [1.0, -2.0, 0.5],
            [2.0, 0.0, 3.5],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        transformed,
        expected,
        atol=1e-6,
    )


def test_transform_points_rotation_about_z() -> None:
    points = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    transform = np.array(
        [
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    transformed = transform_points(
        points,
        transform,
    )

    expected = np.array(
        [
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        transformed,
        expected,
        atol=1e-6,
    )


def test_transform_points_rotation_then_translation() -> None:
    points = np.array(
        [
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    transform = np.array(
        [
            [0.0, -1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    transformed = transform_points(
        points,
        transform,
    )

    expected = np.array(
        [
            [1.0, 3.0, 3.0],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        transformed,
        expected,
        atol=1e-6,
    )


def test_transform_points_empty_cloud() -> None:
    transformed = transform_points(
        np.empty((0, 3)),
        np.eye(4),
    )

    assert transformed.shape == (0, 3)
    assert transformed.dtype == np.float32
    assert transformed.flags.c_contiguous


def test_transform_points_rejects_wrong_point_shape() -> None:
    with pytest.raises(ValueError, match=r"shape \(N, 3\)"):
        transform_points(
            np.zeros((4, 2)),
            np.eye(4),
        )


def test_transform_points_rejects_wrong_matrix_shape() -> None:
    with pytest.raises(ValueError, match=r"shape \(4, 4\)"):
        transform_points(
            np.zeros((4, 3)),
            np.eye(3),
        )


def test_transform_points_rejects_nonfinite_values() -> None:
    points = np.array(
        [
            [np.nan, 0.0, 0.0],
        ]
    )

    with pytest.raises(ValueError, match="finite"):
        transform_points(
            points,
            np.eye(4),
        )


def test_transform_points_rejects_scaling() -> None:
    transform = np.eye(4)
    transform[0, 0] = 2.0

    with pytest.raises(ValueError, match="orthonormal"):
        transform_points(
            np.zeros((1, 3)),
            transform,
        )


def test_transform_points_rejects_invalid_last_row() -> None:
    transform = np.eye(4)
    transform[3, 0] = 1.0

    with pytest.raises(ValueError, match="final row"):
        transform_points(
            np.zeros((1, 3)),
            transform,
        )
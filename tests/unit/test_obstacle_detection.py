import numpy as np
import pytest

from robot.safety.obstacle_detection import (
    ObstacleDetectionConfig,
    detect_obstacle_points,
    detect_obstacle_points_3d,
)


def make_config(
    *,
    forward_only: bool = True,
    angular_bin_size_degrees: float | None = None,
) -> ObstacleDetectionConfig:
    return ObstacleDetectionConfig(
        minimum_height=0.1,
        maximum_height=1.0,
        minimum_range=0.1,
        maximum_range=5.0,
        forward_only=forward_only,
        angular_bin_size_degrees=(
            angular_bin_size_degrees
        ),
    )


def test_detects_point_inside_configured_region() -> None:
    points = np.array(
        [
            [1.0, 0.2, 0.5],
        ],
        dtype=np.float32,
    )

    obstacles = detect_obstacle_points(
        points,
        make_config(),
    )

    np.testing.assert_array_equal(
        obstacles,
        np.array([[1.0, 0.2]], dtype=np.float32),
    )

    assert obstacles.shape == (1, 2)
    assert obstacles.dtype == np.float32
    assert obstacles.flags.c_contiguous


def test_removes_nonfinite_and_origin_points() -> None:
    points = np.array(
        [
            [1.0, 0.0, 0.5],
            [np.nan, 0.0, 0.5],
            [1.0, np.inf, 0.5],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    obstacles = detect_obstacle_points(
        points,
        make_config(),
    )

    np.testing.assert_array_equal(
        obstacles,
        np.array([[1.0, 0.0]], dtype=np.float32),
    )


def test_filters_points_by_height() -> None:
    points = np.array(
        [
            [1.0, 0.0, 0.1],
            [1.0, 0.2, 1.0],
            [1.0, 0.4, 0.09],
            [1.0, 0.6, 1.01],
        ],
        dtype=np.float32,
    )

    obstacles = detect_obstacle_points(
        points,
        make_config(),
    )

    expected = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.2],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        obstacles,
        expected,
    )


def test_filters_points_by_horizontal_range() -> None:
    points = np.array(
        [
            [0.1, 0.0, 0.5],
            [1.0, 0.0, 0.5],
            [3.0, 4.0, 0.5],
            [5.01, 0.0, 0.5],
        ],
        dtype=np.float32,
    )

    obstacles = detect_obstacle_points(
        points,
        make_config(),
    )

    expected = np.array(
        [
            [0.1, 0.0],
            [1.0, 0.0],
            [3.0, 4.0],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        obstacles,
        expected,
    )


def test_forward_only_removes_points_behind_robot() -> None:
    points = np.array(
        [
            [1.0, 0.0, 0.5],
            [-1.0, 0.0, 0.5],
            [0.0, 1.0, 0.5],
        ],
        dtype=np.float32,
    )

    obstacles = detect_obstacle_points(
        points,
        make_config(forward_only=True),
    )

    expected = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        obstacles,
        expected,
    )


def test_can_retain_points_behind_robot() -> None:
    points = np.array(
        [
            [1.0, 0.0, 0.5],
            [-1.0, 0.0, 0.5],
        ],
        dtype=np.float32,
    )

    obstacles = detect_obstacle_points(
        points,
        make_config(forward_only=False),
    )

    np.testing.assert_allclose(
        obstacles,
        points[:, :2],
    )


def test_angular_bins_keep_nearest_point() -> None:
    points = np.array(
        [
            [2.0, 0.0, 0.5],
            [1.0, 0.0, 0.5],
            [2.0, 2.0, 0.5],
            [1.0, 1.0, 0.5],
        ],
        dtype=np.float32,
    )

    obstacles = detect_obstacle_points(
        points,
        make_config(
            angular_bin_size_degrees=10.0,
        ),
    )

    expected = np.array(
        [
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        obstacles,
        expected,
    )


def test_three_dimensional_detection_preserves_height_and_alignment() -> None:
    points = np.array(
        [
            [2.0, 0.0, 0.8],
            [1.0, 0.0, 0.4],
            [1.0, 1.0, 0.7],
        ],
        dtype=np.float32,
    )

    obstacles_3d = detect_obstacle_points_3d(
        points,
        make_config(angular_bin_size_degrees=10.0),
    )
    obstacles_2d = detect_obstacle_points(
        points,
        make_config(angular_bin_size_degrees=10.0),
    )

    np.testing.assert_allclose(
        obstacles_3d,
        np.array(
            [
                [1.0, 0.0, 0.4],
                [1.0, 1.0, 0.7],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(obstacles_2d, obstacles_3d[:, :2])
    assert obstacles_3d.flags.c_contiguous


def test_three_dimensional_detection_returns_shaped_empty_array() -> None:
    obstacles = detect_obstacle_points_3d(
        np.array([[-1.0, 0.0, 0.5]], dtype=np.float32),
        make_config(),
    )

    assert obstacles.shape == (0, 3)
    assert obstacles.dtype == np.float32
    assert obstacles.flags.c_contiguous


def test_returns_shaped_empty_array_when_no_points_remain() -> None:
    points = np.array(
        [
            [-1.0, 0.0, 0.5],
            [1.0, 0.0, 2.0],
        ],
        dtype=np.float32,
    )

    obstacles = detect_obstacle_points(
        points,
        make_config(),
    )

    assert obstacles.shape == (0, 2)
    assert obstacles.dtype == np.float32
    assert obstacles.flags.c_contiguous


def test_invalid_point_cloud_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"shape \(N, 3\)"):
        detect_obstacle_points(
            np.zeros((4, 2)),
            make_config(),
        )


@pytest.mark.parametrize(
    "config_values",
    [
        {
            "minimum_height": 1.0,
            "maximum_height": 1.0,
        },
        {
            "minimum_height": 1.0,
            "maximum_height": 0.5,
        },
        {
            "minimum_height": 0.1,
            "maximum_height": 1.0,
            "minimum_range": -0.1,
        },
        {
            "minimum_height": 0.1,
            "maximum_height": 1.0,
            "minimum_range": 2.0,
            "maximum_range": 1.0,
        },
        {
            "minimum_height": 0.1,
            "maximum_height": 1.0,
            "angular_bin_size_degrees": 0.0,
        },
    ],
)
def test_invalid_configuration_is_rejected(
    config_values: dict[str, float],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        ObstacleDetectionConfig(**config_values)

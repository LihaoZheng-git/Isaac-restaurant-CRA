import math

import numpy as np
import pytest

from robot.safety.obstacle_tracking import (
    ObstacleTracker,
    ObstacleTrackerConfig,
)


CLUSTER_OFFSETS = np.array(
    [
        [-0.05, -0.05],
        [-0.05, 0.05],
        [0.05, -0.05],
        [0.05, 0.05],
    ],
    dtype=np.float64,
)


def make_cluster(
    center_xy: tuple[float, float],
) -> np.ndarray:
    """Return four connected points centered at ``center_xy``."""

    return (
        np.asarray(center_xy, dtype=np.float64)
        + CLUSTER_OFFSETS
    )


def world_points_to_robot(
    points_world_xy: np.ndarray,
    *,
    robot_position_world_xy: tuple[float, float],
    robot_yaw: float,
) -> np.ndarray:
    """Transform world-frame test points into robot-frame coordinates."""

    robot_position = np.asarray(
        robot_position_world_xy,
        dtype=np.float64,
    )
    translated_points = (
        points_world_xy - robot_position
    )

    cosine_yaw = math.cos(robot_yaw)
    sine_yaw = math.sin(robot_yaw)

    world_from_robot_rotation = np.array(
        [
            [cosine_yaw, -sine_yaw],
            [sine_yaw, cosine_yaw],
        ],
        dtype=np.float64,
    )

    return (
        translated_points
        @ world_from_robot_rotation
    )


def test_first_frame_creates_stationary_track() -> None:
    tracker = ObstacleTracker()
    points = make_cluster((2.0, 0.5))

    result = tracker.update(
        points,
        robot_position_world_xy=np.array([0.0, 0.0]),
        robot_yaw=0.0,
        timestamp_seconds=0.0,
    )

    assert result.cluster_count == 1
    assert result.new_track_count == 1
    assert result.matched_track_count == 0
    assert len(result.tracked_obstacles) == 1
    assert result.tracked_obstacles[0].track_id == 0
    assert result.tracked_obstacles[0].observation_count == 1
    assert np.all(result.point_track_ids == 0)
    np.testing.assert_allclose(
        result.point_velocities_robot_xy,
        np.zeros_like(points),
    )


def test_moving_cluster_velocity_is_estimated() -> None:
    tracker = ObstacleTracker()

    tracker.update(
        make_cluster((2.0, 0.0)),
        robot_position_world_xy=np.array([0.0, 0.0]),
        robot_yaw=0.0,
        timestamp_seconds=0.0,
    )

    result = tracker.update(
        make_cluster((2.0, -0.06)),
        robot_position_world_xy=np.array([0.0, 0.0]),
        robot_yaw=0.0,
        timestamp_seconds=0.1,
    )

    obstacle = result.tracked_obstacles[0]

    assert obstacle.track_id == 0
    assert obstacle.observation_count == 2
    assert obstacle.matched_previous_track
    assert result.matched_track_count == 1
    assert result.new_track_count == 0
    np.testing.assert_allclose(
        obstacle.velocity_world_xy,
        [0.0, -0.6],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.point_velocities_robot_xy,
        np.tile([0.0, -0.6], (4, 1)),
        atol=1e-12,
    )


def test_robot_translation_is_not_obstacle_velocity() -> None:
    tracker = ObstacleTracker()
    stationary_world_points = make_cluster((3.0, 0.5))

    first_robot_points = world_points_to_robot(
        stationary_world_points,
        robot_position_world_xy=(0.0, 0.0),
        robot_yaw=0.0,
    )
    second_robot_points = world_points_to_robot(
        stationary_world_points,
        robot_position_world_xy=(0.1, 0.0),
        robot_yaw=0.0,
    )

    tracker.update(
        first_robot_points,
        robot_position_world_xy=np.array([0.0, 0.0]),
        robot_yaw=0.0,
        timestamp_seconds=0.0,
    )
    result = tracker.update(
        second_robot_points,
        robot_position_world_xy=np.array([0.1, 0.0]),
        robot_yaw=0.0,
        timestamp_seconds=0.1,
    )

    np.testing.assert_allclose(
        result.tracked_obstacles[0].velocity_world_xy,
        [0.0, 0.0],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.point_velocities_robot_xy,
        np.zeros_like(second_robot_points),
        atol=1e-12,
    )


def test_robot_rotation_is_not_obstacle_velocity() -> None:
    tracker = ObstacleTracker()
    stationary_world_points = make_cluster((2.0, 1.0))

    first_robot_points = world_points_to_robot(
        stationary_world_points,
        robot_position_world_xy=(0.0, 0.0),
        robot_yaw=0.0,
    )
    second_robot_points = world_points_to_robot(
        stationary_world_points,
        robot_position_world_xy=(0.0, 0.0),
        robot_yaw=math.pi / 2.0,
    )

    tracker.update(
        first_robot_points,
        robot_position_world_xy=np.array([0.0, 0.0]),
        robot_yaw=0.0,
        timestamp_seconds=0.0,
    )
    result = tracker.update(
        second_robot_points,
        robot_position_world_xy=np.array([0.0, 0.0]),
        robot_yaw=math.pi / 2.0,
        timestamp_seconds=0.1,
    )

    np.testing.assert_allclose(
        result.tracked_obstacles[0].velocity_world_xy,
        [0.0, 0.0],
        atol=1e-12,
    )


def test_world_velocity_is_resolved_into_robot_frame() -> None:
    tracker = ObstacleTracker()
    robot_yaw = math.pi / 2.0

    first_world_points = make_cluster((2.0, 0.0))
    second_world_points = make_cluster((2.0, 0.1))

    tracker.update(
        world_points_to_robot(
            first_world_points,
            robot_position_world_xy=(0.0, 0.0),
            robot_yaw=robot_yaw,
        ),
        robot_position_world_xy=np.array([0.0, 0.0]),
        robot_yaw=robot_yaw,
        timestamp_seconds=0.0,
    )
    result = tracker.update(
        world_points_to_robot(
            second_world_points,
            robot_position_world_xy=(0.0, 0.0),
            robot_yaw=robot_yaw,
        ),
        robot_position_world_xy=np.array([0.0, 0.0]),
        robot_yaw=robot_yaw,
        timestamp_seconds=0.1,
    )

    np.testing.assert_allclose(
        result.tracked_obstacles[0].velocity_world_xy,
        [0.0, 1.0],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.tracked_obstacles[0].velocity_robot_xy,
        [1.0, 0.0],
        atol=1e-12,
    )


def test_velocity_measurements_are_smoothed() -> None:
    tracker = ObstacleTracker(
        ObstacleTrackerConfig(
            velocity_smoothing_factor=0.5
        )
    )

    tracker.update(
        make_cluster((1.0, 0.0)),
        robot_position_world_xy=np.array([0.0, 0.0]),
        robot_yaw=0.0,
        timestamp_seconds=0.0,
    )
    second_result = tracker.update(
        make_cluster((1.1, 0.0)),
        robot_position_world_xy=np.array([0.0, 0.0]),
        robot_yaw=0.0,
        timestamp_seconds=0.1,
    )
    third_result = tracker.update(
        make_cluster((1.3, 0.0)),
        robot_position_world_xy=np.array([0.0, 0.0]),
        robot_yaw=0.0,
        timestamp_seconds=0.2,
    )

    np.testing.assert_allclose(
        second_result.tracked_obstacles[0].velocity_world_xy,
        [1.0, 0.0],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        third_result.tracked_obstacles[0].velocity_world_xy,
        [1.5, 0.0],
        atol=1e-12,
    )


def test_multiple_clusters_keep_identity_when_input_order_changes() -> None:
    tracker = ObstacleTracker()

    first_result = tracker.update(
        np.vstack(
            [
                make_cluster((2.0, -1.0)),
                make_cluster((2.0, 1.0)),
            ]
        ),
        robot_position_world_xy=np.array([0.0, 0.0]),
        robot_yaw=0.0,
        timestamp_seconds=0.0,
    )

    second_result = tracker.update(
        np.vstack(
            [
                make_cluster((2.05, 1.0)),
                make_cluster((2.05, -1.0)),
            ]
        ),
        robot_position_world_xy=np.array([0.0, 0.0]),
        robot_yaw=0.0,
        timestamp_seconds=0.1,
    )

    first_ids_by_side = {
        int(math.copysign(1, obstacle.centroid_world_xy[1])): (
            obstacle.track_id
        )
        for obstacle in first_result.tracked_obstacles
    }
    second_ids_by_side = {
        int(math.copysign(1, obstacle.centroid_world_xy[1])): (
            obstacle.track_id
        )
        for obstacle in second_result.tracked_obstacles
    }

    assert second_ids_by_side == first_ids_by_side
    assert second_result.matched_track_count == 2
    assert second_result.new_track_count == 0


def test_isolated_points_receive_zero_velocity_and_no_track() -> None:
    tracker = ObstacleTracker(
        ObstacleTrackerConfig(
            minimum_cluster_points=2
        )
    )

    points = np.array(
        [
            [1.0, -1.0],
            [1.0, 1.0],
        ]
    )

    result = tracker.update(
        points,
        robot_position_world_xy=np.array([0.0, 0.0]),
        robot_yaw=0.0,
        timestamp_seconds=0.0,
    )

    assert result.cluster_count == 0
    assert len(result.tracked_obstacles) == 0
    assert np.all(result.point_track_ids == -1)
    np.testing.assert_allclose(
        result.point_velocities_robot_xy,
        np.zeros_like(points),
    )


def test_repeated_timestamp_is_rejected() -> None:
    tracker = ObstacleTracker()
    points = make_cluster((1.0, 0.0))

    tracker.update(
        points,
        robot_position_world_xy=np.array([0.0, 0.0]),
        robot_yaw=0.0,
        timestamp_seconds=1.0,
    )

    with pytest.raises(ValueError, match="must increase"):
        tracker.update(
            points,
            robot_position_world_xy=np.array([0.0, 0.0]),
            robot_yaw=0.0,
            timestamp_seconds=1.0,
        )


def test_reset_removes_history_and_restarts_track_ids() -> None:
    tracker = ObstacleTracker()
    points = make_cluster((1.0, 0.0))

    tracker.update(
        points,
        robot_position_world_xy=np.array([0.0, 0.0]),
        robot_yaw=0.0,
        timestamp_seconds=0.0,
    )

    tracker.reset()

    result = tracker.update(
        points,
        robot_position_world_xy=np.array([0.0, 0.0]),
        robot_yaw=0.0,
        timestamp_seconds=0.0,
    )

    assert tracker.active_track_count == 1
    assert result.tracked_obstacles[0].track_id == 0


@pytest.mark.parametrize(
    "points",
    [
        np.zeros((2, 3)),
        np.array([[np.nan, 0.0]]),
    ],
)
def test_invalid_obstacle_points_are_rejected(
    points: np.ndarray,
) -> None:
    tracker = ObstacleTracker()

    with pytest.raises(ValueError):
        tracker.update(
            points,
            robot_position_world_xy=np.array([0.0, 0.0]),
            robot_yaw=0.0,
            timestamp_seconds=0.0,
        )

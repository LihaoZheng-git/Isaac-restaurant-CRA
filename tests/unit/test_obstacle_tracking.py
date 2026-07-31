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


def test_stationary_speed_threshold_suppresses_small_jitter() -> None:
    tracker = ObstacleTracker(
        ObstacleTrackerConfig(
            stationary_speed_threshold=0.20
        )
    )

    tracker.update(
        make_cluster((1.0, 0.0)),
        robot_position_world_xy=np.array([0.0, 0.0]),
        robot_yaw=0.0,
        timestamp_seconds=0.0,
    )
    result = tracker.update(
        make_cluster((1.01, 0.0)),
        robot_position_world_xy=np.array([0.0, 0.0]),
        robot_yaw=0.0,
        timestamp_seconds=0.1,
    )

    # The measured centroid speed is 0.10 m/s, so the configured
    # 0.20 m/s deadband should classify it as stationary.
    np.testing.assert_allclose(
        result.tracked_obstacles[0].velocity_world_xy,
        [0.0, 0.0],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.point_velocities_robot_xy,
        np.zeros((4, 2)),
        atol=1e-12,
    )


def test_stationary_speed_threshold_preserves_clear_motion() -> None:
    tracker = ObstacleTracker(
        ObstacleTrackerConfig(
            stationary_speed_threshold=0.20
        )
    )

    tracker.update(
        make_cluster((1.0, 0.0)),
        robot_position_world_xy=np.array([0.0, 0.0]),
        robot_yaw=0.0,
        timestamp_seconds=0.0,
    )
    result = tracker.update(
        make_cluster((1.0, -0.06)),
        robot_position_world_xy=np.array([0.0, 0.0]),
        robot_yaw=0.0,
        timestamp_seconds=0.1,
    )

    # The proxy-like motion is 0.60 m/s and must remain available to
    # the velocity-aware CBF.
    np.testing.assert_allclose(
        result.tracked_obstacles[0].velocity_world_xy,
        [0.0, -0.6],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.point_velocities_robot_xy,
        np.tile([0.0, -0.6], (4, 1)),
        atol=1e-12,
    )


def test_motion_requires_configured_consecutive_updates() -> None:
    tracker = ObstacleTracker(
        ObstacleTrackerConfig(
            velocity_smoothing_factor=1.0,
            stationary_speed_threshold=0.20,
            minimum_consecutive_motion_updates=3,
        )
    )

    tracker.update(
        make_cluster((1.0, 0.0)),
        robot_position_world_xy=np.array([0.0, 0.0]),
        robot_yaw=0.0,
        timestamp_seconds=0.0,
    )

    for update_index in range(1, 3):
        result = tracker.update(
            make_cluster(
                (1.0, -0.06 * update_index)
            ),
            robot_position_world_xy=np.array([0.0, 0.0]),
            robot_yaw=0.0,
            timestamp_seconds=0.1 * update_index,
        )
        np.testing.assert_allclose(
            result.tracked_obstacles[0].velocity_world_xy,
            [0.0, 0.0],
            atol=1e-12,
        )

    confirmed_result = tracker.update(
        make_cluster((1.0, -0.18)),
        robot_position_world_xy=np.array([0.0, 0.0]),
        robot_yaw=0.0,
        timestamp_seconds=0.3,
    )

    np.testing.assert_allclose(
        (
            confirmed_result.tracked_obstacles[0]
            .velocity_world_xy
        ),
        [0.0, -0.6],
        atol=1e-12,
    )


def test_reversing_centroid_outliers_are_not_confirmed() -> None:
    tracker = ObstacleTracker(
        ObstacleTrackerConfig(
            velocity_smoothing_factor=1.0,
            stationary_speed_threshold=0.20,
            minimum_consecutive_motion_updates=3,
        )
    )

    tracker.update(
        make_cluster((1.0, 0.0)),
        robot_position_world_xy=np.array([0.0, 0.0]),
        robot_yaw=0.0,
        timestamp_seconds=0.0,
    )

    alternating_centers = (
        (1.0, 0.06),
        (1.0, 0.0),
        (1.0, 0.06),
        (1.0, 0.0),
    )

    for update_index, center in enumerate(
        alternating_centers,
        start=1,
    ):
        result = tracker.update(
            make_cluster(center),
            robot_position_world_xy=np.array([0.0, 0.0]),
            robot_yaw=0.0,
            timestamp_seconds=0.1 * update_index,
        )
        np.testing.assert_allclose(
            result.tracked_obstacles[0].velocity_world_xy,
            [0.0, 0.0],
            atol=1e-12,
        )


def test_single_centroid_jump_does_not_leave_velocity_tail() -> None:
    tracker = ObstacleTracker(
        ObstacleTrackerConfig(
            velocity_smoothing_factor=0.35,
            stationary_speed_threshold=0.20,
            minimum_consecutive_motion_updates=3,
        )
    )

    tracker.update(
        make_cluster((1.0, 0.0)),
        robot_position_world_xy=np.array([0.0, 0.0]),
        robot_yaw=0.0,
        timestamp_seconds=0.0,
    )

    jump_result = tracker.update(
        make_cluster((1.3, 0.0)),
        robot_position_world_xy=np.array([0.0, 0.0]),
        robot_yaw=0.0,
        timestamp_seconds=0.1,
    )
    np.testing.assert_allclose(
        jump_result.tracked_obstacles[0].velocity_world_xy,
        [0.0, 0.0],
        atol=1e-12,
    )

    for update_index in range(2, 6):
        result = tracker.update(
            make_cluster((1.3, 0.0)),
            robot_position_world_xy=np.array([0.0, 0.0]),
            robot_yaw=0.0,
            timestamp_seconds=0.1 * update_index,
        )
        np.testing.assert_allclose(
            result.tracked_obstacles[0].velocity_world_xy,
            [0.0, 0.0],
            atol=1e-12,
        )


def test_missed_observation_resets_motion_confirmation() -> None:
    tracker = ObstacleTracker(
        ObstacleTrackerConfig(
            velocity_smoothing_factor=1.0,
            stationary_speed_threshold=0.20,
            minimum_consecutive_motion_updates=3,
        )
    )

    tracker.update(
        make_cluster((1.0, 0.0)),
        robot_position_world_xy=np.array([0.0, 0.0]),
        robot_yaw=0.0,
        timestamp_seconds=0.0,
    )
    tracker.update(
        make_cluster((1.06, 0.0)),
        robot_position_world_xy=np.array([0.0, 0.0]),
        robot_yaw=0.0,
        timestamp_seconds=0.1,
    )
    tracker.update(
        np.empty((0, 2)),
        robot_position_world_xy=np.array([0.0, 0.0]),
        robot_yaw=0.0,
        timestamp_seconds=0.2,
    )

    for update_index, x_position in (
        (3, 1.18),
        (4, 1.24),
    ):
        result = tracker.update(
            make_cluster((x_position, 0.0)),
            robot_position_world_xy=np.array([0.0, 0.0]),
            robot_yaw=0.0,
            timestamp_seconds=0.1 * update_index,
        )
        np.testing.assert_allclose(
            result.tracked_obstacles[0].velocity_world_xy,
            [0.0, 0.0],
            atol=1e-12,
        )

    confirmed_result = tracker.update(
        make_cluster((1.30, 0.0)),
        robot_position_world_xy=np.array([0.0, 0.0]),
        robot_yaw=0.0,
        timestamp_seconds=0.5,
    )
    np.testing.assert_allclose(
        (
            confirmed_result.tracked_obstacles[0]
            .velocity_world_xy
        ),
        [0.6, 0.0],
        atol=1e-12,
    )


def test_negative_stationary_speed_threshold_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        ObstacleTrackerConfig(
            stationary_speed_threshold=-0.01
        )


@pytest.mark.parametrize(
    "invalid_value",
    [0, -1],
)
def test_invalid_minimum_motion_updates_are_rejected(
    invalid_value: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="must be greater than zero",
    ):
        ObstacleTrackerConfig(
            minimum_consecutive_motion_updates=(
                invalid_value
            )
        )


def test_boolean_minimum_motion_updates_are_rejected() -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        ObstacleTrackerConfig(
            minimum_consecutive_motion_updates=True
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

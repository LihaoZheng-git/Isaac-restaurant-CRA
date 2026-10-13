from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robot.perception.physical_attributes import EstimateStatus
from robot.perception.sensor_epochs import (
    EgoPoseHistory,
    EgoPoseSample,
    align_camera_lidar_epoch,
)
from robot.safety.obstacle_tracking import ObstacleTracker, ObstacleTrackerConfig


def pose(t, x=0.0, yaw=0.0, **changes):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
    matrix[0, 3] = x
    return EgoPoseSample(
        t,
        matrix,
        changes.get("status", EstimateStatus.ORACLE),
        changes.get("source", "test_pose_oracle"),
    )


def history():
    result = EgoPoseHistory(maximum_interpolation_gap_seconds=0.11)
    for i in range(4):
        result.append(pose(i * 0.1, i * 0.02, i * 0.03))
    return result


def test_interpolated_pose_and_camera_alignment_use_both_acquisition_epochs():
    h = history()
    sample = h.sample(0.15)
    assert sample.world_from_robot[0, 3] == pytest.approx(0.03)
    assert Rotation.from_matrix(sample.world_from_robot[:3, :3]).as_euler("xyz")[
        2
    ] == pytest.approx(0.045)
    assert sample.status is EstimateStatus.ORACLE
    aligned = align_camera_lidar_epoch(
        h,
        lidar_timestamp_seconds=0.1,
        camera_timestamp_seconds=0.2,
        camera_from_robot=np.eye(4),
    )
    scan_point = np.array([1.0, 0.1, 0.3, 1.0])
    world_point = h.sample(0.1).world_from_robot @ scan_point
    expected = np.linalg.inv(h.sample(0.2).world_from_robot) @ world_point
    np.testing.assert_allclose(aligned.camera_from_scan_robot @ scan_point, expected)
    assert not np.allclose(expected, scan_point)
    assert aligned.sensor_age_seconds == pytest.approx(0.1)


def test_velocity_is_derived_from_pose_history_without_retiming():
    assert history().velocity_world_xy(0.2) == pytest.approx((0.2, 0.0))


@pytest.mark.parametrize("at", [-0.1, 0.4])
def test_unavailable_pose_epochs_are_not_extrapolated(at):
    with pytest.raises(ValueError):
        history().sample(at)


def test_large_pose_gaps_and_changed_provenance_are_rejected():
    h = EgoPoseHistory()
    h.append(pose(0))
    h.append(pose(0.1))
    with pytest.raises(ValueError, match="gap"):
        h.sample(0.05)
    h = EgoPoseHistory(maximum_interpolation_gap_seconds=0.11)
    h.append(pose(0))
    h.append(pose(0.1, status=EstimateStatus.MEASURED))
    with pytest.raises(ValueError, match="provenance"):
        h.sample(0.05)


@pytest.mark.parametrize("scan,camera", [(0.2, 0.1), (0.0, 0.3)])
def test_future_and_stale_sensor_pair_is_rejected(scan, camera):
    with pytest.raises(ValueError, match="future or stale"):
        align_camera_lidar_epoch(
            history(),
            lidar_timestamp_seconds=scan,
            camera_timestamp_seconds=camera,
            camera_from_robot=np.eye(4),
        )


def test_pose_arrays_are_owned_and_improper_transforms_cannot_enter_history():
    sample = pose(0)
    assert not sample.world_from_robot.flags.writeable
    matrix = np.eye(4)
    matrix[0, 0] = -1
    with pytest.raises(ValueError, match="proper rigid"):
        replace(sample, world_from_robot=matrix)
    with pytest.raises(ValueError, match="unavailable"):
        replace(sample, status=EstimateStatus.UNAVAILABLE)


def test_moving_robot_does_not_create_static_obstacle_velocity():
    tracker = ObstacleTracker(
        ObstacleTrackerConfig(
            cluster_radius=0.2,
            minimum_cluster_points=4,
            velocity_smoothing_factor=1.0,
            maximum_association_distance=0.5,
        )
    )
    world = np.array(
        [
            [2.0, 0.0, 0.2, 1.0],
            [2.0, 0.03, 0.2, 1.0],
            [2.03, 0.03, 0.2, 1.0],
            [2.03, 0.0, 0.2, 1.0],
        ]
    )
    for sample in history().samples:
        robot_points = (world @ np.linalg.inv(sample.world_from_robot).T)[:, :2]
        result = tracker.update(
            robot_points,
            robot_position_world_xy=sample.world_from_robot[:2, 3],
            robot_yaw=Rotation.from_matrix(sample.world_from_robot[:3, :3]).as_euler(
                "xyz"
            )[2],
            timestamp_seconds=sample.timestamp_seconds,
        )
    assert len(result.tracked_obstacles) == 1
    assert result.tracked_obstacles[0].velocity_world_xy == pytest.approx(
        (0.0, 0.0), abs=1e-12
    )

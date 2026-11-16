from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robot.perception.lidar_deskew import deskew_uncompensated_lidar
from robot.perception.physical_attributes import EstimateStatus
from robot.perception.sensor_epochs import EgoPoseHistory, EgoPoseSample
from robot.sensor.lidar_scan_metadata import LidarScanMetadata


def pose(t):
    value = np.eye(4)
    value[:3, :3] = Rotation.from_euler("z", 0.7 * t).as_matrix()
    value[:3, 3] = [0.3 * t, 0.1 * t, 0]
    return value


def fixture():
    history = EgoPoseHistory(maximum_interpolation_gap_seconds=0.05)
    for t in np.linspace(1, 1.15, 7):
        history.append(EgoPoseSample(t, pose(t), EstimateStatus.ORACLE, "fixture_pose"))
    offsets = np.array([0, 25_000_000, 50_000_000, 100_000_000], np.int64)
    metadata = LidarScanMetadata(
        1_000_000_000,
        "SENSOR",
        "NONCOMPENSATED",
        None,
        None,
        tuple(np.eye(4).reshape(-1)),
        offsets,
        (),
    )
    mount = np.eye(4)
    mount[:3, 3] = [0.1, 0, 0.5]
    world = np.array(
        [[3, y, z] for y, z in [(0, 0), (0.2, 0.1), (0.1, 0.2), (0.2, 0.3)]]
    )
    raw = []
    for point, offset in zip(world, offsets):
        transform = np.linalg.inv(pose(1 + offset * 1e-9) @ mount)
        raw.append(transform[:3, :3] @ point + transform[:3, 3])
    return history, metadata, mount, world, np.array(raw)


def test_translation_rotation_and_sensor_mount_recover_static_world_geometry():
    history, metadata, mount, world, raw = fixture()
    result, diagnostics = deskew_uncompensated_lidar(
        raw, metadata, history, robot_from_sensor=mount, reference_timestamp_seconds=1.0
    )
    recovered_world = result @ pose(1)[:3, :3].T + pose(1)[:3, 3]
    np.testing.assert_allclose(recovered_world, world, atol=2e-7)
    legacy = raw @ mount[:3, :3].T + mount[:3, 3]
    assert np.linalg.norm(legacy[-1] - result[-1]) > 0.15
    assert diagnostics["pose_status"] == "oracle"
    assert diagnostics["acquisition_end_seconds"] == pytest.approx(1.1)


@pytest.mark.parametrize(
    "changes",
    [
        {"motion_compensation_state": "COMPENSATED"},
        {"motion_compensation_state": "NOT_APPLICABLE"},
        {"frame_of_reference": "WORLD"},
        {"point_time_offsets_nanoseconds": None},
        {"point_time_offsets_nanoseconds": np.array([0, 1])},
        {"model_to_app_transform": None},
    ],
)
def test_unsupported_or_missing_contracts_are_rejected(changes):
    history, metadata, mount, _, raw = fixture()
    with pytest.raises(ValueError):
        deskew_uncompensated_lidar(
            raw,
            replace(metadata, **changes),
            history,
            robot_from_sensor=mount,
            reference_timestamp_seconds=1,
        )


def test_future_returns_never_use_extrapolated_poses():
    history, metadata, mount, _, raw = fixture()
    metadata = replace(
        metadata,
        point_time_offsets_nanoseconds=np.array(
            [0, 25_000_000, 50_000_000, 200_000_000]
        ),
    )
    with pytest.raises(ValueError, match="scan-duration|no extrapolation"):
        deskew_uncompensated_lidar(
            raw,
            metadata,
            history,
            robot_from_sensor=mount,
            reference_timestamp_seconds=1,
        )


def test_mixed_provenance_rejected():
    history, metadata, mount, _, raw = fixture()
    history.samples[2] = replace(history.samples[2], source="other_pose")
    with pytest.raises(ValueError, match="provenance"):
        deskew_uncompensated_lidar(
            raw,
            metadata,
            history,
            robot_from_sensor=mount,
            reference_timestamp_seconds=1,
        )


def test_missing_pose_for_an_in_scope_scan_cannot_be_extrapolated():
    history, metadata, mount, _, raw = fixture()
    history.samples = history.samples[:-3]
    with pytest.raises(ValueError, match="no extrapolation"):
        deskew_uncompensated_lidar(
            raw,
            metadata,
            history,
            robot_from_sensor=mount,
            reference_timestamp_seconds=1,
        )


def test_copied_scan_can_use_later_recorded_pose_without_shifting_its_epochs():
    history, metadata, mount, world, raw = fixture()
    retained = history.samples[:4]
    original_points = raw.copy()
    original_offsets = metadata.point_time_offsets_nanoseconds.copy()
    original_header = metadata.sensor_timestamp_nanoseconds
    history.samples = retained.copy()
    with pytest.raises(ValueError, match="no extrapolation"):
        deskew_uncompensated_lidar(
            raw,
            metadata,
            history,
            robot_from_sensor=mount,
            reference_timestamp_seconds=1.0,
        )
    history.append(EgoPoseSample(1.1, pose(1.1), EstimateStatus.ORACLE, "fixture_pose"))
    corrected, _ = deskew_uncompensated_lidar(
        raw, metadata, history, robot_from_sensor=mount, reference_timestamp_seconds=1.0
    )
    np.testing.assert_allclose(
        corrected @ pose(1)[:3, :3].T + pose(1)[:3, 3], world, atol=2e-7
    )
    np.testing.assert_array_equal(raw, original_points)
    np.testing.assert_array_equal(
        metadata.point_time_offsets_nanoseconds, original_offsets
    )
    assert metadata.sensor_timestamp_nanoseconds == original_header


def test_history_gap_rejected():
    history, metadata, mount, _, raw = fixture()
    history.samples.pop(2)
    history.maximum_gap = 0.03
    with pytest.raises(ValueError, match="gap"):
        deskew_uncompensated_lidar(
            raw,
            metadata,
            history,
            robot_from_sensor=mount,
            reference_timestamp_seconds=1,
        )


def test_stationary_deskew_is_the_existing_sensor_mount_transform():
    history, metadata, mount, _, raw = fixture()
    history.samples = [
        replace(item, world_from_robot=pose(1)) for item in history.samples
    ]
    result, _ = deskew_uncompensated_lidar(
        raw, metadata, history, robot_from_sensor=mount, reference_timestamp_seconds=1
    )
    np.testing.assert_allclose(result, raw @ mount[:3, :3].T + mount[:3, 3], atol=2e-7)

from types import SimpleNamespace

import numpy as np
import pytest

from robot.sensor.lidar_scan_metadata import copy_scan_metadata


def output():
    return SimpleNamespace(
        numElements=4,
        timestampNs=1_000_000_000,
        frameOfReference=SimpleNamespace(name="SENSOR"),
        motionCompensationState=SimpleNamespace(name="NONCOMPENSATED"),
        frameStart=SimpleNamespace(
            timestampNs=1_000_000_000, posM=[1, 2, 3], orientation=[0, 0, 0, 1]
        ),
        frameEnd=SimpleNamespace(
            timestampNs=1_100_000_000, posM=[1.1, 2, 3], orientation=[0, 0, 0, 1]
        ),
        modelToAppTransform=np.eye(4).reshape(-1),
        timeOffsetNs=np.array([0, 10_000_000, 50_000_000, 90_000_000], dtype=np.int32),
    )


def test_times_follow_exact_point_filter_and_own_sensor_buffers():
    source = output()
    result = copy_scan_metadata(source, np.array([True, False, False, True]))
    source.timeOffsetNs[:] = 0
    source.frameStart.posM[0] = 100
    source.modelToAppTransform[:] = 100
    np.testing.assert_array_equal(
        result.point_time_offsets_nanoseconds, [0, 90_000_000]
    )
    assert not result.point_time_offsets_nanoseconds.flags.writeable
    assert result.frame_start.position_m == (1, 2, 3)
    assert result.model_to_app_transform[0] == 1
    assert result.summary()["offset_max_ns"] == 90_000_000
    assert not result.unavailable_reasons


def test_missing_times_and_invalid_poses_remain_unavailable():
    source = output()
    source.timeOffsetNs = None
    source.frameEnd.orientation = [0, 0, 0, 0]
    source.modelToAppTransform = None
    result = copy_scan_metadata(source, np.ones(4, dtype=bool))
    assert result.point_time_offsets_nanoseconds is None
    assert result.frame_end is None
    assert result.model_to_app_transform is None
    assert len(result.unavailable_reasons) == 3


@pytest.mark.parametrize("offsets", [np.array([1, 2]), np.zeros(4, dtype=float)])
def test_short_or_nonnumeric_integer_time_contract_is_not_repaired(offsets):
    source = output()
    source.timeOffsetNs = offsets
    assert (
        copy_scan_metadata(
            source, np.ones(4, dtype=bool)
        ).point_time_offsets_nanoseconds
        is None
    )


@pytest.mark.parametrize("mask", [np.ones(3, dtype=bool), np.ones(4, dtype=int)])
def test_matching_xyz_mask_is_required(mask):
    with pytest.raises(ValueError):
        copy_scan_metadata(output(), mask)


def test_reference_and_compensation_are_preserved_without_guessing():
    source = output()
    source.frameOfReference = SimpleNamespace(name="WORLD")
    source.motionCompensationState = SimpleNamespace(name="COMPENSATED")
    result = copy_scan_metadata(source, np.ones(4, dtype=bool))
    assert result.frame_of_reference == "WORLD"
    assert result.motion_compensation_state == "COMPENSATED"

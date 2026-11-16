from types import SimpleNamespace

import numpy as np
import pytest
from test_crossing_maneuver import setup
from test_safe_sage_cbf_filter import make_fields

from robot.safety.checked_wait_transition import assess_checked_wait_transition


def context():
    mpc, _, field, stopped = setup()
    field.crossing_yield_track_ids = (7,)
    field.tracking_result = SimpleNamespace(
        timestamp_seconds=2.0,
        tracked_obstacles=[
            SimpleNamespace(
                track_id=7,
                point_count=80,
                matched_previous_track=True,
                observation_count=4,
            )
        ],
    )
    return mpc, field, stopped


def test_current_complete_stopped_horizon_is_accepted_without_changing_fields_or_clocks():
    mpc, field, stopped = context()
    before = field.pipeline.h_full.values.copy()
    result = assess_checked_wait_transition(mpc, np.zeros(3), field, lambda: stopped)
    assert result["accepted"] and result["exact_dense_accepted"]
    assert result["future_analytical_accepted"] and result["first_analytical_accepted"]
    assert result["source_timestamp_seconds"] == 2.0
    assert result["original_selection_timestamp_seconds"] == 1.0
    assert field.sensor_timestamp_seconds == 2.0
    np.testing.assert_array_equal(field.pipeline.h_full.values, before)


@pytest.mark.parametrize("failure", ["dynamic", "geometry", "turn", "emergency"])
def test_failed_exact_geometry_or_analytical_stop_cannot_accept_a_wait_field(failure):
    mpc, field, stopped = context()
    if failure == "dynamic":
        field.temporal_safety_derivative = -100.0
    elif failure == "geometry":
        cs, h, guidance = make_fields(safety_value=2.0, unsafe_yaw_indices=(0, 1, 2, 3))
        field.pipeline = SimpleNamespace(
            configuration_space=cs, h_full=h, guidance_fields=guidance
        )
    elif failure == "turn":
        stopped.safe_angular_velocity = 0.1
    else:
        stopped.emergency_stop_active = True
    result = assess_checked_wait_transition(mpc, np.zeros(3), field, lambda: stopped)
    assert not result["accepted"]


@pytest.mark.parametrize("failure", ["expired", "future", "pair", "unselected"])
def test_missing_or_misaligned_original_selection_and_tracker_sources_rejected(failure):
    mpc, field, stopped = context()
    if failure == "expired":
        field.motion_passing_reference_source_timestamp_seconds = -8.0
    elif failure == "future":
        field.motion_passing_reference_source_timestamp_seconds = 2.1
    elif failure == "pair":
        field.tracking_result.timestamp_seconds = 1.9
    else:
        field.motion_passing_reference_track_ids = ()
    with pytest.raises(ValueError):
        assess_checked_wait_transition(mpc, np.zeros(3), field, lambda: stopped)


@pytest.mark.parametrize("failure", ["absent", "sparse", "new", "few", "ambiguous"])
def test_unsupported_track_cannot_qualify_a_waiting_policy(failure):
    mpc, field, stopped = context()
    track = field.tracking_result.tracked_obstacles[0]
    if failure == "absent":
        field.tracking_result.tracked_obstacles = []
    elif failure == "sparse":
        track.point_count = 19
    elif failure == "new":
        track.matched_previous_track = False
    elif failure == "few":
        track.observation_count = 2
    else:
        field.tracking_result.tracked_obstacles.append(track)
    result = assess_checked_wait_transition(mpc, np.zeros(3), field, lambda: stopped)
    assert (
        not result["accepted"]
        and result["reason"] == "insufficient_current_track_support"
    )


def test_parallel_or_multiple_crossings_cannot_create_a_stopped_plan():
    mpc, field, _ = context()
    for ids in [(), (7, 8)]:
        field.crossing_yield_track_ids = ids
        result = assess_checked_wait_transition(
            mpc,
            np.zeros(3),
            field,
            lambda: pytest.fail("no filter call without unique current crossing"),
        )
        assert (
            not result["accepted"] and result["reason"] == "no_unique_current_crossing"
        )

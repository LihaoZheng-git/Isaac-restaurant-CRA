"""Fresh approach preferences cannot masquerade as a completed rear passage."""

from dataclasses import replace

import numpy as np
import pytest

from robot.safety.person_relative_rear import CurrentRearApproachReference, PersonRelativeRearMemory
from robot.safety.checked_rear_pass_reference import CheckedRearPassReferenceConfig, search_checked_rear_pass_reference
from test_person_relative_rear import inputs
from test_rear_delayed_commit import delayed_config
from test_checked_rear_pass_reference import search


def config():
    return replace(delayed_config(), enable_preparation=True)


@pytest.mark.parametrize('side', [-1, 1])
@pytest.mark.parametrize('yaw', [0., .9])
def test_fresh_preparation_mirrors_rotates_and_does_not_freeze_a_forecast(side, yaw):
    args = inputs(side=side, yaw=yaw, lead=1.37)
    memory = PersonRelativeRearMemory(config())
    ref = memory.update(**args)
    assert isinstance(ref, CurrentRearApproachReference)
    assert memory.reference is None and memory.active_reference is ref
    route = np.array(args['route_direction'])
    motion = np.array(args['velocity_xy'])
    assert (ref.target - args['initial'][:2]) @ motion < 0
    assert (ref.target - args['person_xy']) @ route < 0
    assert ref.context(args['initial'])['person_relative_phase'] == 'preparation'
    assert 'predicted_person_xy' not in ref.diagnostic()
    assert 'passage_timestamp_seconds' not in ref.diagnostic()


def test_each_preparation_needs_current_source_and_missing_motion_discards_it():
    args = inputs(lead=1.37)
    memory = PersonRelativeRearMemory(config())
    ref = memory.update(**args)
    current = dict(track_id=7, selection_timestamp=1., source_timestamp=1.,
        person_xy=args['person_xy'], velocity_xy=args['velocity_xy'], route_direction=args['route_direction'])
    ref.validate_current(**current)
    for change in [dict(source_timestamp=1.1), dict(track_id=8), dict(selection_timestamp=1.1),
                   dict(person_xy=(1.4, -.5)), dict(velocity_xy=(0., .6))]:
        with pytest.raises(ValueError, match='exact current'):
            ref.validate_current(**dict(current, **change))
    memory.update(**dict(args, track_id=None, source_timestamp=1.1))
    assert memory.active_reference is None
    new = memory.update(**dict(args, source_timestamp=1.6))
    assert new.source_timestamp == 1.6 and new.selection_timestamp == 1.
    # New observations can admit a complete curve, retaining original selection.
    full = memory.update(**dict(inputs(lead=.6), source_timestamp=1.7, person_xy=(.6, 1.1)))
    assert full is memory.reference and not isinstance(full, CurrentRearApproachReference)
    memory.update(**dict(args, track_id=None, source_timestamp=1.8))
    assert memory.invalidated
    assert memory.update(**dict(args, source_timestamp=1.9)) is None
    assert memory.active_reference is None


@pytest.mark.parametrize('failure', [None, 'analytical', 'future', 'temporal'])
def test_preparation_still_requires_complete_exact_future_and_first_cbf_checks(failure):
    mpc, _, options, check = search(failure=failure)
    args = inputs(lead=1.37)
    ref = PersonRelativeRearMemory(config()).update(**args)
    result = search_checked_rear_pass_reference(mpc, args['initial'], args['goal'],
        args['person_xy'], args['velocity_xy'], args['route_direction'], options, check,
        config=CheckedRearPassReferenceConfig(person_relative=config(), feasibility_led_start=True),
        person_relative_reference=ref)
    assert (result.trajectory is not None) == (failure is None)
    if failure is None:
        trajectory = result.trajectory
        assert mpc._trajectory_is_safe(trajectory)
        assert np.max((trajectory.chassis_states[:, :2] - args['person_xy']) @ args['route_direction']) < -.05
        assert not check(trajectory.commands[0]).emergency_stop_active
        assert trajectory.commands[0, 1] < 0


def test_preparation_horizon_cannot_cross_current_person_plane_even_in_clear_field():
    mpc, _, options, check = search()
    args = inputs(lead=.11)
    ref = PersonRelativeRearMemory(config()).update(**args)
    result = search_checked_rear_pass_reference(mpc, args['initial'], args['goal'],
        args['person_xy'], args['velocity_xy'], args['route_direction'], options, check,
        config=CheckedRearPassReferenceConfig(person_relative=config(), feasibility_led_start=True),
        person_relative_reference=ref)
    assert result.diagnostics['counts']['preparation_plane_rejections'] > 0
    if result.trajectory is not None:
        assert result.trajectory.chassis_states[:, 0].max() < .06


def test_original_expiry_and_disabled_mode_cannot_prepare():
    memory = PersonRelativeRearMemory(config())
    assert memory.update(**dict(inputs(), source_timestamp=9.01)) is None
    assert memory.invalidated and memory.active_reference is None
    assert PersonRelativeRearMemory(delayed_config()).update(**inputs(lead=1.37)) is None
    with pytest.raises(ValueError):
        replace(config(), enable_preparation=1)


def test_lateral_target_is_route_anchored_not_repeatedly_added_to_robot_position():
    memory = PersonRelativeRearMemory(config())
    first = memory.update(**inputs(lead=1.37))
    second = memory.update(**dict(inputs(lead=1.37), initial=(.1, -.2, -.2), source_timestamp=1.1))
    assert first.target[1] == second.target[1] == -.35


def test_live_worker_restores_normal_lambda_for_current_preparation_and_field_validates_source():
    from types import SimpleNamespace
    from test_encounter_guidance import inputs as policies
    from test_live_rear_pass import track
    from test_safe_sage_runtime import make_runtime, make_observation, make_provider
    from robot.evaluation.encounter_guidance import apply_encounter_guidance
    from robot.evaluation.live_rear_pass import adapt_live_rear_guidance

    normal, rows, classification, passing = policies()
    applied, rows, crossing = apply_encounter_guidance(normal, rows, classification, passing, source_time=2.)
    source = SimpleNamespace(sensor_timestamp_seconds=2., robot_position_world_xy=(0., 0.), robot_yaw=0.,
        waypoint_world_xy=(1., 0.), nominal_route_direction_world_xy=(1., 0.),
        nominal_route_terminal_goal_world_xy=(3.9, 0.))
    tracking = SimpleNamespace(timestamp_seconds=2., tracked_obstacles=[track(position=(1.37, -.5), velocity=(0., .5))])
    memory = PersonRelativeRearMemory(config())
    changed, audit, wait, rear, _ = adapt_live_rear_guidance(normal, applied, rows, tracking, source,
        anchor=np.zeros(2), anchor_yaw=0., crossing_ids=crossing,
        config=CheckedRearPassReferenceConfig(person_relative=config(), feasibility_led_start=True),
        person_relative_memory=memory)
    assert not wait and rear == (7,) and memory.reference is None
    assert isinstance(memory.active_reference, CurrentRearApproachReference)
    assert all(changed[key] == normal[key] for key in changed)
    assert any(row.get('live_rear_pass_behavior', {}).get('person_relative_phase') == 'preparation' for row in audit)

    runtime = make_runtime()
    runtime.initialize(make_observation(1, 2.), make_provider())
    f = runtime.field
    t = replace(f.tracking_result.tracked_obstacles[0], track_id=7, point_count=30,
        observation_count=3, matched_previous_track=True, centroid_world_xy=(1.37, -.5), velocity_world_xy=(0., .5))
    f = replace(f, anchor_position_world_xy=np.zeros(2), anchor_yaw=0.,
        tracking_result=replace(f.tracking_result, tracked_obstacles=(t,)),
        motion_passing_reference_track_ids=(7,), motion_passing_reference_sides=('right',),
        motion_passing_reference_source_timestamp_seconds=1., rear_pass_reference_track_ids=(7,),
        rear_pass_route_direction_grid_xy=(1., 0.), person_relative_rear_reference=memory.active_reference)
    assert f.person_relative_rear_reference is memory.active_reference
    with pytest.raises(ValueError, match='exact current'):
        replace(f, person_relative_rear_reference=replace(memory.active_reference, source_timestamp=1.9))

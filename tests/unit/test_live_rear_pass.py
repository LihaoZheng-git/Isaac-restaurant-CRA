import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from test_encounter_guidance import inputs
from test_live_checked_wait import configured
from test_live_object_perception import packet

from robot.evaluation.encounter_guidance import apply_encounter_guidance
from robot.evaluation.live_rear_pass import (
    adapt_live_rear_guidance, live_track_context, world_to_grid_rotation,
    LiveRearPassControlRuntime, summarize_rear_execution,
)
from robot.safety.checked_rear_pass_reference import CheckedRearPassReferenceConfig


def track(key=7, position=(10., 23.3), velocity=(-.5, 0.)):
    return SimpleNamespace(track_id=key, point_count=30, matched_previous_track=True,
        observation_count=3, centroid_world_xy=position, velocity_world_xy=velocity)


@pytest.mark.parametrize('sign', [-1, 1])
@pytest.mark.parametrize('lead,mode,lam', [(3.3, 'rear', 1.25), (1., 'wait', 0.)])
def test_live_policy_rotates_motion_and_changes_only_owned_lambda(sign, lead, mode, lam):
    normal, rows, classification, passing = inputs()
    applied, rows, crossing = apply_encounter_guidance(normal, rows, classification, passing, source_time=2.)
    saved = copy.deepcopy((normal, applied, rows))
    source = SimpleNamespace(sensor_timestamp_seconds=2., robot_position_world_xy=(10., 20.),
        robot_yaw=np.pi/2, waypoint_world_xy=(10., 23.9), nominal_route_direction_world_xy=(0., 1.))
    tracked = SimpleNamespace(timestamp_seconds=2., tracked_obstacles=[track(
        position=(10.+sign*.5, 20.+lead), velocity=(-sign*.5, 0.))])
    changed, audit, waiting, rear, route = adapt_live_rear_guidance(normal, applied, rows,
        tracked, source, anchor=np.array([10., 20.]), anchor_yaw=np.pi/2,
        crossing_ids=crossing, config=CheckedRearPassReferenceConfig())
    assert waiting == ((7,) if mode == 'wait' else ())
    assert rear == ((7,) if mode == 'rear' else ())
    np.testing.assert_allclose(route, (1., 0.), atol=1e-12)
    for row in audit:
        key = row['track_or_group_id']
        assert changed[key].risk == normal[key].risk
        assert changed[key].boundary_flux == normal[key].boundary_flux
        assert changed[key].passing_side == normal[key].passing_side
        assert changed[key].tangential_strength == (lam if key in (7, 8, 70) else 1.25)
        if key in (7, 8, 70):
            d = row['live_rear_pass_behavior']
            assert d['mode'] == mode and d['source_timestamp_seconds'] == 2.
            assert d['original_selection_timestamp_seconds'] == 1.
            if mode == 'rear':
                np.testing.assert_allclose(d['rear_direction'], (0., -sign), atol=1e-12)
    assert (normal, applied, rows) == saved


@pytest.mark.parametrize('sign', [-1,1])
def test_feasibility_start_admits_only_reference_search_and_retains_physical_policy(sign):
    normal,rows,classification,passing=inputs()
    applied,rows,crossing=apply_encounter_guidance(normal,rows,classification,passing,source_time=2.)
    source=SimpleNamespace(sensor_timestamp_seconds=2.,robot_position_world_xy=(10.,20.),robot_yaw=np.pi/2,
        waypoint_world_xy=(10.,23.9),nominal_route_direction_world_xy=(0.,1.))
    tracked=SimpleNamespace(timestamp_seconds=2.,tracked_obstacles=[track(position=(10.+sign*.5,21.),velocity=(-sign*.5,0.))])
    changed,audit,waiting,rear,_=adapt_live_rear_guidance(normal,applied,rows,tracked,source,
        anchor=np.array([10.,20.]),anchor_yaw=np.pi/2,crossing_ids=crossing,
        config=CheckedRearPassReferenceConfig(feasibility_led_start=True))
    assert not waiting and rear == (7,)
    for row in audit:
        key=row['track_or_group_id']
        assert changed[key] == normal[key]
        if 'live_rear_pass_behavior' in row:
            assert row['live_rear_pass_behavior']['lead_preference_overridden'] is True


@pytest.mark.parametrize('failure', ['old', 'expired', 'few', 'unmatched', 'short'])
def test_live_track_context_rejects_stale_or_weak_geometry(failure):
    t = track()
    f = SimpleNamespace(sensor_timestamp_seconds=2.,
        motion_passing_reference_source_timestamp_seconds=1.,
        motion_passing_reference_track_ids=(7,), motion_passing_reference_sides=('left',),
        tracking_result=SimpleNamespace(timestamp_seconds=2., tracked_obstacles=[t]),
        anchor_yaw=np.pi/2, anchor_position_world_xy=np.array([10., 20.]))
    if failure == 'old':
        f.tracking_result.timestamp_seconds = 1.9
    if failure == 'expired':
        f.motion_passing_reference_source_timestamp_seconds = -7.
    if failure == 'few':
        t.point_count = 19
    if failure == 'unmatched':
        t.matched_previous_track = False
    if failure == 'short':
        t.observation_count = 2
    with pytest.raises(ValueError):
        live_track_context(f, 7)


def test_native_field_no_current_crossing_does_not_borrow_old_reference(monkeypatch):
    worker = configured()
    worker.live_rear_pass_config = CheckedRearPassReferenceConfig()
    source = replace(packet(), nominal_route_direction_world_xy=(1., 0.))
    # First tracker frame cannot become an established rear reference.
    result = worker(source)
    assert not result.field.rear_pass_reference_track_ids
    assert not result.field.crossing_yield_track_ids
    assert all(r['lambda'] == 1.25 for r in result.diagnostics['applied_field_policies'])
    assert result.field.rear_pass_route_direction_grid_xy == (1., 0.)
    with pytest.raises(ValueError, match='one currently selected'):
        replace(result.field, rear_pass_reference_track_ids=(999,))


def test_encounter_fit_state_is_transactional_and_no_human_defaults_are_identical():
    from robot.evaluation.live_checked_wait import StagedLivePerception
    from robot.safety.encounter_motion import EncounterMotionClassifier
    from robot.safety.encounter_motion_estimate import EncounterMotionEstimate
    ordinary, experimental = configured(), configured()
    ordinary.encounter_motion, experimental.encounter_motion = EncounterMotionClassifier(), EncounterMotionClassifier()
    experimental.rear_encounter_motion_estimate = EncounterMotionEstimate()
    a, b = ordinary(packet()), experimental(packet())
    np.testing.assert_array_equal(a.field.pipeline.h_full.values, b.field.pipeline.h_full.values)
    assert a.objects[0].policy == b.objects[0].policy
    estimator = experimental.rear_encounter_motion_estimate
    staged = StagedLivePerception(experimental)
    proposed = staged(replace(packet(2, 1.1)))
    assert experimental.rear_encounter_motion_estimate is estimator
    assert estimator.last_timestamp == 1.
    staged.discard()
    proposed = staged(packet(3, 1.2))
    staged.accept(proposed)
    assert experimental.rear_encounter_motion_estimate is not estimator
    assert experimental.rear_encounter_motion_estimate.last_timestamp == 1.2


def test_native_runtime_primary_identity_without_qualified_crossing():
    from test_safe_sage_runtime import make_runtime, make_observation, make_provider
    from test_checked_rear_pass_reference import search
    from robot.safety.safe_sage_runtime import SafeSAGEControlRuntime
    from robot.safety.safe_sage_unicycle_cbf_filter import SafeSAGEUnicycleCBFFilter
    mpc, _, _, _ = search()
    backup = SafeSAGEUnicycleCBFFilter(mpc.paper_filter)
    primary = SafeSAGEControlRuntime(mpc=mpc, backup_filter=backup)
    experimental = LiveRearPassControlRuntime(mpc=primary.mpc, backup_filter=primary.backup_filter)
    perception = make_runtime()
    perception.initialize(make_observation(1, 1.), make_provider())
    field = perception.field
    initial, nominal = np.array([0., 0., 0.]), np.array([.1, 0.])
    a = primary.plan(initial, nominal, field)
    b = experimental.plan(initial, nominal, field, nominal_goal_grid_xy=(3.9, 0.))
    np.testing.assert_array_equal(a.command_sequence, b.command_sequence)
    assert not b.rear_pass_reference['selected']


def test_world_rotation_does_not_apply_translation_to_velocity():
    rot = world_to_grid_rotation(np.pi/2)
    np.testing.assert_allclose(rot @ [-.5, 0.], [0., .5], atol=1e-12)


@pytest.mark.parametrize('sign', [-1, 1])
def test_consistent_fit_prevents_mixed_motion_rear_context_without_changing_raw_tracking(sign):
    from robot.safety.encounter_motion_estimate import EncounterPolicyMotion, EncounterMotionEstimateConfig
    raw, fitted = np.array([sign*.26, -.38]), np.array([sign*.5, -.1])
    center = np.array([10.-sign*.5, 23.3])
    times = np.array([1.6,1.7,1.8,1.9,2.])
    positions = center+(times-2.)[:,None]*fitted
    estimate = EncounterPolicyMotion(7,tuple(times),tuple(map(tuple,positions)),tuple(raw),tuple(fitted),30,3,EncounterMotionEstimateConfig())
    tracked = SimpleNamespace(timestamp_seconds=2.,tracked_obstacles=[track(position=tuple(center),velocity=tuple(raw))])
    source = SimpleNamespace(sensor_timestamp_seconds=2.,robot_position_world_xy=(10.,20.),robot_yaw=np.pi/2,
        waypoint_world_xy=(10.,23.9),nominal_route_direction_world_xy=(0.,1.))
    normal, rows, classification, passing = inputs()
    applied, rows, crossing = apply_encounter_guidance(normal, rows, classification, passing, source_time=2.)
    kwargs = dict(anchor=np.array([10.,20.]),anchor_yaw=np.pi/2,crossing_ids=crossing,
        config=CheckedRearPassReferenceConfig())
    _, old_rows, _, old_rear, _ = adapt_live_rear_guidance(normal,applied,rows,tracked,source,**kwargs)
    _, new_rows, _, new_rear, _ = adapt_live_rear_guidance(normal,applied,rows,tracked,source,
        policy_motion=(estimate,),**kwargs)
    assert not old_rear and new_rear == (7,)
    assert next(r['live_rear_pass_behavior'] for r in old_rows if 'live_rear_pass_behavior' in r)['mode'] == 'none'
    decision = next(r['live_rear_pass_behavior'] for r in new_rows if 'live_rear_pass_behavior' in r)
    np.testing.assert_allclose(decision['rear_direction'],(0.,sign),atol=1e-12)
    assert decision['motion_policy_estimate']['status'] == 'inferred'
    np.testing.assert_array_equal(tracked.tracked_obstacles[0].velocity_world_xy,raw)


def test_runtime_field_rejects_policy_motion_with_unselected_or_changed_source():
    from test_safe_sage_runtime import make_runtime,make_observation,make_provider
    from robot.safety.encounter_motion_estimate import EncounterPolicyMotion,EncounterMotionEstimateConfig
    runtime=make_runtime()
    runtime.initialize(make_observation(1,2.),make_provider())
    f=runtime.field
    # Provide one established current track in this numeric metadata fixture.
    t=replace(f.tracking_result.tracked_obstacles[0],point_count=30,observation_count=3,matched_previous_track=True)
    tracking=replace(f.tracking_result,tracked_obstacles=(t,))
    times=(1.6,1.7,1.8,1.9,2.)
    estimate=EncounterPolicyMotion(t.track_id,times,tuple(t.centroid_world_xy for _ in times),
        t.velocity_world_xy,(0.,0.),30,3,EncounterMotionEstimateConfig())
    f=replace(f,tracking_result=tracking)
    with pytest.raises(ValueError,match='currently selected'):
        replace(f,rear_pass_policy_motion=(estimate,))
    accepted=replace(f,motion_passing_reference_track_ids=(t.track_id,),motion_passing_reference_sides=('left',),
        motion_passing_reference_source_timestamp_seconds=1.9,rear_pass_policy_motion=(estimate,))
    assert accepted.rear_pass_policy_motion == (estimate,)
    with pytest.raises(ValueError,match='original current'):
        replace(accepted,sensor_timestamp_seconds=2.1,tracking_result=replace(tracking,timestamp_seconds=2.1))


@pytest.mark.parametrize('sign', [-1, 1])
def test_native_numeric_search_selects_translating_checked_reference_with_rotated_tracks(monkeypatch, sign):
    from test_checked_wait_transition import context
    from robot.safety.safe_sage_unicycle_cbf_filter import SafeSAGEUnicycleCBFFilter
    from robot.safety.safe_sage_unicycle_mpc import SafeSAGEUnicycleMPC
    old_mpc, f, _ = context()
    mpc = SafeSAGEUnicycleMPC(old_mpc.paper_filter, old_mpc.config)
    f.anchor_yaw, f.anchor_position_world_xy = np.pi/2, np.array([10., 20.])
    f.tracking_result.tracked_obstacles = [track(
        position=(10.+sign*.5, 23.3), velocity=(-sign*.5, 0.))]
    f.rear_pass_reference_track_ids = (7,)
    f.rear_pass_route_direction_grid_xy = (1., 0.)
    f.crossing_yield_track_ids = ()
    control = LiveRearPassControlRuntime(mpc=mpc,
        backup_filter=SafeSAGEUnicycleCBFFilter(mpc.paper_filter))
    # The numeric helper fixture has a lightweight pipeline; production still
    # requires the actual validated SafeSAGERuntimeField type (tested above).
    monkeypatch.setattr(control, '_validate_field', lambda field: None)
    result = control.plan(np.zeros(3), (.2, 0.), f, nominal_goal_grid_xy=(3.9, 0.))
    assert result.success and result.rear_pass_reference['selected']
    assert result.status == 'live_experimental_checked_rear_reference'
    assert np.linalg.norm(result.predicted_chassis_states[-1, :2]) >= .01
    assert result.angular_velocity * sign < 0
    np.testing.assert_allclose(result.rear_pass_reference['observed_velocity_grid_xy'], (0., sign*.5), atol=1e-12)
    assert not result.command_sequence.flags.writeable
    checked = mpc._evaluate_trajectory(np.zeros(3), result.command_sequence, require_feasible=True,
        configuration_space=f.pipeline.configuration_space, h_full=f.pipeline.h_full,
        guidance_fields=f.pipeline.guidance_fields, temporal_safety_derivative=f.temporal_safety_derivative)
    assert checked is not None and mpc._trajectory_is_safe(checked)
    assert not control.filter_command(result.first_command, np.zeros(3), f).emergency_stop_active


@pytest.mark.parametrize('failure', [None, 'field_clock', 'stop', 'late', 'unaccepted'])
def test_rear_execution_requires_actual_request_clock_translation_during_walking(failure):
    truth = [dict(time=10.+i*.1, actor_xyz=[i*.05, 0., 0.], phase='move' if i < 8 else 'walk_away')
             for i in range(12)]
    plans = [dict(time=.2+i*.1, source_age_seconds=.05, accepted=failure != 'unaccepted',
        rear_pass_reference_ablation=dict(selected=True, source_timestamp_seconds=10.05+i*.1,
            sensor_frame_id=i, original_selection_timestamp_seconds=10.)) for i in range(3)]
    # Optimizer request10.15 is different from the field epoch10.05.
    age = .15 if failure == 'field_clock' else .05
    commands = [dict(time=.2+i*.01, optimizer_candidate_source_age=age+i*.01,
        horizon_validation_age=0., v=.1, stop_reason='stale_field' if failure == 'stop' else None)
        for i in range(3)]
    if failure == 'late':
        for c in commands:
            c['time'] += .8
            c['optimizer_candidate_source_age'] += .8
    result = summarize_rear_execution(plans, commands, truth, navigation_start_time=10.)
    assert result['passed'] == (failure is None)
    if failure is None:
        assert result['verified_translation_seconds'] == pytest.approx(.02)
        assert result['joined_commands'][0]['optimizer_request_timestamp_seconds'] == pytest.approx(10.15)

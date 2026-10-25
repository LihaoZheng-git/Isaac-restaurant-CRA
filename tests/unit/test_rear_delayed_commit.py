"""Delayed curve commitment changes preference timing, never safety acceptance."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from robot.safety.person_relative_rear import PersonRelativeRearMemory
from robot.scene.rear_pass_encounter import (
    OPEN_ACTOR_START_WORLD,
    OPEN_ROBOT_ROUTE_WORLD,
    OPEN_ROBOT_START_WORLD,
    open_robot_route_local_xy,
    validate_crossing_layout,
)
from test_person_relative_rear import configuration, inputs


def delayed_config():
    return replace(configuration(), minimum_current_rear_separation_m=1.05,
                   continue_frozen_until_deadline=True)


def test_delayed_commit_waits_for_current_observation_without_borrowing_future_motion():
    memory = PersonRelativeRearMemory(delayed_config())
    assert memory.update(**inputs(lead=.6)) is None
    assert memory.reason == "waiting_for_current_rear_commit_separation"
    current = dict(inputs(lead=.6), source_timestamp=1.1, person_xy=(.6, 1.1))
    ref = memory.update(**current)
    assert ref is not None and ref.source_timestamp == 1.1
    assert ref.selection_timestamp == 1.0 and ref.current_rear_separation_m == pytest.approx(1.1)
    assert ref.expires_at <= 4.1 and ref.passage_seconds <= 3.0
    assert ref.screening_seconds <= ref.passage_seconds
    assert ref.config.maximum_linear_velocity == .25
    with pytest.raises(ValueError, match="commit preference"):
        replace(ref, original_person_xy=(.6, 1.0))


@pytest.mark.parametrize("change", [dict(minimum_current_rear_separation_m=-.1),
    dict(minimum_current_rear_separation_m=True), dict(minimum_current_rear_separation_m=np.nan),
    dict(continue_frozen_until_deadline=1)])
def test_delayed_configuration_requires_explicit_finite_preferences(change):
    with pytest.raises(ValueError):
        replace(configuration(), **change)


def test_frozen_curve_can_outlast_legacy_lateral_release_but_never_original_expiry():
    from test_checked_rear_pass_reference import search
    from robot.safety.checked_rear_pass_reference import (
        CheckedRearPassReferenceConfig, search_checked_rear_pass_reference,
    )
    mpc, _, options, check = search()
    cfg = delayed_config()
    args = dict(inputs(lead=.6), person_xy=(.6, 1.1))
    ref = PersonRelativeRearMemory(cfg).update(**args)
    assert ref is not None
    current_person = np.array(args['person_xy']) + [0., .15]
    # A current compatible source beyond the old 1.2m release preference.
    ref.validate_current(track_id=7, selection_timestamp=1., source_timestamp=1.2,
        person_xy=current_person, velocity_xy=(0., .5), route_direction=(1., 0.))
    result = search_checked_rear_pass_reference(mpc, args['initial'], args['goal'],
        current_person, (0., .5), (1., 0.), options, check,
        config=CheckedRearPassReferenceConfig(person_relative=cfg, feasibility_led_start=True),
        person_relative_reference=ref)
    assert result.diagnostics['context']['mode'] == 'rear'
    assert result.trajectory is not None
    assert result.diagnostics['context']['person_relative']['original_source_timestamp_seconds'] == 1.
    with pytest.raises(ValueError, match="expired"):
        ref.validate_current(track_id=7, selection_timestamp=1., source_timestamp=4.1,
            person_xy=(.6, 2.65), velocity_xy=(0., .5), route_direction=(1., 0.))


def test_live_worker_continues_only_an_existing_fresh_curve_after_release():
    from robot.evaluation.live_rear_pass import adapt_live_rear_guidance
    from robot.evaluation.encounter_guidance import apply_encounter_guidance
    from robot.safety.checked_rear_pass_reference import CheckedRearPassReferenceConfig
    from test_encounter_guidance import inputs as policies
    from test_live_rear_pass import track

    normal, rows, classification, passing = policies()
    applied, rows, crossing = apply_encounter_guidance(normal, rows, classification, passing, source_time=2.)
    memory = PersonRelativeRearMemory(delayed_config())
    reference = memory.update(**dict(inputs(lead=.6), source_timestamp=1.9, person_xy=(.6, 1.2)))
    assert reference is not None
    source = SimpleNamespace(sensor_timestamp_seconds=2., robot_position_world_xy=(0., 0.), robot_yaw=0.,
        waypoint_world_xy=(1., 0.), nominal_route_direction_world_xy=(1., 0.),
        nominal_route_terminal_goal_world_xy=(3.9, 0.))
    tracking = SimpleNamespace(timestamp_seconds=2., tracked_obstacles=[track(position=(.6, 1.25), velocity=(0., .5))])
    _, _, wait, rear, _ = adapt_live_rear_guidance(normal, applied, rows, tracking, source,
        anchor=np.zeros(2), anchor_yaw=0., crossing_ids=crossing,
        config=CheckedRearPassReferenceConfig(person_relative=delayed_config(), feasibility_led_start=True),
        person_relative_memory=memory)
    assert rear == (7,) and not wait and memory.reference is reference

    # Delayed eligibility is not confined to the old 1.05..1.2m sliver.
    fresh = PersonRelativeRearMemory(delayed_config())
    tracking.tracked_obstacles = [track(position=(.6, 1.4), velocity=(0., .5))]
    _, _, wait, rear, _ = adapt_live_rear_guidance(normal, applied, rows, tracking, source,
        anchor=np.zeros(2), anchor_yaw=0., crossing_ids=crossing,
        config=CheckedRearPassReferenceConfig(person_relative=delayed_config(), feasibility_led_start=True),
        person_relative_memory=fresh)
    assert rear == (7,) and not wait and fresh.reference is not None
    assert fresh.reference.source_timestamp == 2. and fresh.reference.selection_timestamp == 1.


def test_open_encounter_has_separate_authored_poses_and_obstacle_unaware_waypoints():
    from robot.scene.restaurant_departing_pedestrian import DepartingPedestrianRoute
    assert validate_crossing_layout('lower-aisle', direction='left-to-right', extension=0., crossing_y=8.9)
    route = DepartingPedestrianRoute(crossing_direction='left-to-right', crossing_layout='lower-aisle')
    assert OPEN_ACTOR_START_WORLD == (25.8, 7., 0.)
    assert route.route == ((29.8, 7., 0.), (29.8, 12.5, 0.))
    assert route.advance((29.56, 7., 0.), 10.)
    assert route.goal == (29.8, 12.5, 0.)
    local = np.array(open_robot_route_local_xy())
    np.testing.assert_allclose(local @ [[0., 1.], [-1., 0.]] + OPEN_ROBOT_START_WORLD[:2], OPEN_ROBOT_ROUTE_WORLD)
    assert np.max(np.abs(local[:, 1])) == 0.
    assert DepartingPedestrianRoute().route == ((29.8, 8.9, 0.), (29.8, 12.5, 0.))


@pytest.mark.parametrize('change', [dict(layout='unknown'), dict(direction='right-to-left'),
    dict(extension=.1), dict(crossing_y=7.)])
def test_open_layout_does_not_silently_override_legacy_geometry_or_enable_mirror(change):
    args = dict(layout='lower-aisle', direction='left-to-right', extension=0., crossing_y=8.9)
    args.update(change)
    with pytest.raises(ValueError):
        validate_crossing_layout(**args)


def test_open_actor_acceptance_retains_actual_motion_and_geometry_thresholds():
    from robot.evaluation.departing_pedestrian_validation import summarize_departing_pedestrian
    from test_departing_pedestrian import trace
    rows = trace()
    for row in rows:
        t = row['time']
        row['actor_xyz'][0] = 25.8 + min(4., max(0., t - 1.) * .5)
        row['actor_xyz'][1] = 7. + max(0., t - 9.) * .5
        row['phase'] = 'idle_before' if t < 1 else 'move' if t < 9 else 'walk_away'
    # Extend the actual trace through the unchanged upper-aisle criterion.
    for t in np.arange(13., 20., .05):
        row = dict(rows[-1], time=float(t), actor_xyz=[29.8, 7. + (t - 9.) * .5, 0.])
        rows.append(row)
    assert summarize_departing_pedestrian(rows, parked_robot=True, crossing_layout='lower-aisle')['passed']
    assert not summarize_departing_pedestrian(rows, parked_robot=True)['passed']
    rows[80]['actor_layout_clearance_m'] = .01
    assert not summarize_departing_pedestrian(rows, parked_robot=True, crossing_layout='lower-aisle')['passed']


def test_relocated_setup_clears_initial_velocities_once_without_pose_lock_or_ongoing_commands():
    from robot.scene.rear_pass_encounter import reset_open_encounter_velocity

    events = []

    class Robot:
        def get_dof_positions(self):
            return SimpleNamespace(numpy=lambda: np.ones((1, 5)))

        def set_velocities(self, **kwargs):
            events.append(('base', kwargs))

        def set_dof_velocities(self, values):
            events.append(('dofs', values))

    reset_open_encounter_velocity(SimpleNamespace(robot=Robot(), stop=lambda: events.append(('stop',))))
    assert [x[0] for x in events] == ['stop', 'base', 'dofs']
    assert events[1][1] == dict(linear_velocities=[[0., 0., 0.]], angular_velocities=[[0., 0., 0.]])
    np.testing.assert_array_equal(events[2][1], np.zeros((1, 5)))


def test_separate_stationary_mount_keeps_original_layers_and_binds_actual_chassis():
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    from robot.scene.rear_pass_encounter import mount_stationary_perception_bench

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, '/World')
    UsdGeom.Xform.Define(stage, '/World/Robot')
    chassis = UsdGeom.Xform.Define(stage, '/World/Robot/Chassis')
    UsdPhysics.RigidBodyAPI.Apply(chassis.GetPrim())
    chassis.AddTranslateOp().Set(Gf.Vec3d(.1, .2, .3))
    original = stage.GetRootLayer().ExportToString()
    metadata = mount_stationary_perception_bench(stage, '/World/Robot', '/World/Robot/Chassis')
    assert stage.GetRootLayer().ExportToString() == original
    joint = UsdPhysics.FixedJoint.Get(stage, metadata['joint_path'])
    assert joint.GetBody0Rel().GetTargets() == []
    assert [str(x) for x in joint.GetBody1Rel().GetTargets()] == ['/World/Robot/Chassis']
    np.testing.assert_allclose(joint.GetLocalPos0Attr().Get(), [26.8, 5.2, .37382], atol=1e-6)
    assert metadata['status'] == 'oracle' and metadata['enabled']
    assert stage.GetSessionLayer().GetPrimAtPath(metadata['joint_path']) is not None

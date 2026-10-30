import copy

import numpy as np
import pytest

from robot.evaluation.parallel_walking_validation import validate_parallel_actor, validate_parallel_curve
from robot.scene.parallel_walk_encounter import (
    PARALLEL_ROBOT_START, PARALLEL_ROBOT_ROUTE, validate_parallel_scope,
)


def example():
    truth, records = [], []
    for t in np.linspace(0, 10, 201):
        phase = 'idle_before' if t < 1 else 'move' if t < 8 else 'away_endpoint'
        person = [29.8, 10.5-min(3.5, max(0., t-1)*.5), 0.]
        robot = [28.8-.35*np.sin(np.pi*t/10)**2, 7+.35*t]
        truth.append(dict(time=float(t), phase=phase, actor_xyz=person, robot_xy=robot,
            left_foot_xyz=[0., .15*np.sin(8*t), 0.], right_foot_xyz=[0., 0., 0.],
            actor_layout_clearance_m=.4, collider_alignment_error_m=0., nonzero_wheel_command_count=0))
        records.append(dict(sensor_timestamp=float(t), objects=[dict(
            diagnostics=dict(policy_return_count=30, centroid_world_xy=person[:2]),
            attributes=dict(motion={k: dict(value=value, status='inferred', timestamp_seconds=float(t))
                for k, value in [('velocity_x_mps', 0.), ('velocity_y_mps', -.5 if phase == 'move' else 0.)]}))],
            diagnostics=dict(applied_field_policies=[{'lambda': 1.25}])))
    return records, truth


def test_parallel_curve_requires_current_motion_normal_field_and_actual_moving_encounter():
    records, truth = example()
    result = validate_parallel_curve(records, truth, goal_world_xy=(28.8, 10.5))
    assert result['passed']
    assert result['nearby_walking_excursion_m'] == pytest.approx(.35)
    assert result['observed_parallel_rows'] > 3


@pytest.mark.parametrize('failure', ['late_curve', 'no_curve', 'lambda', 'stale', 'unknown',
    'wrong_identity', 'no_arrival', 'wrong_direction', 'geometry', 'no_limb_motion'])
def test_parallel_claim_rejects_invalid_physical_or_control_evidence(failure):
    records, truth = copy.deepcopy(example())
    if failure in {'late_curve', 'no_curve'}:
        for r in truth:
            r['robot_xy'][0] = 28.8-(.4 if failure == 'late_curve' and r['time'] > 8 else 0.)
        truth[-1]['robot_xy'][0] = 28.8
    elif failure == 'lambda':
        records[80]['diagnostics']['applied_field_policies'][0]['lambda'] = 0.
    elif failure in {'stale', 'unknown'}:
        for r in records:
            value = r['objects'][0]['attributes']['motion']['velocity_y_mps']
            value['timestamp_seconds' if failure == 'stale' else 'value'] = -1 if failure == 'stale' else None
    elif failure == 'wrong_identity':
        for r in records:
            r['objects'][0]['diagnostics']['centroid_world_xy'] = [0., 0.]
    elif failure == 'no_arrival':
        truth[-1]['robot_xy'][1] = 10.
    elif failure == 'wrong_direction':
        for r in truth:
            r['actor_xyz'][1] = 17.5-r['actor_xyz'][1]
    elif failure == 'geometry':
        truth[50]['actor_layout_clearance_m'] = .01
    else:
        for r in truth:
            r['left_foot_xyz'] = r['right_foot_xyz']
    assert not validate_parallel_curve(records, truth, goal_world_xy=(28.8, 10.5))['passed']


def test_parked_gate_rejects_drift_return_and_nonzero_wheels():
    _, truth = example()
    for r in truth:
        r['robot_xy'] = [28.8, 7.]
    assert validate_parallel_actor(truth, parked_robot=True)['passed']
    truth[40]['robot_xy'][0] += .03
    assert not validate_parallel_actor(truth, parked_robot=True)['passed']
    truth[40]['robot_xy'][0] -= .03
    truth[40]['nonzero_wheel_command_count'] = 1
    assert not validate_parallel_actor(truth, parked_robot=True)['passed']


def test_fixture_is_short_straight_nominal_and_not_a_crossing_override():
    assert PARALLEL_ROBOT_START[:2] == (28.8, 7.)
    assert PARALLEL_ROBOT_ROUTE == ((1., 0.), (2., 0.), (3.5, 0.))
    validate_parallel_scope(direction='right-to-left', extension=0., crossing_y=8.9, layout='legacy')
    with pytest.raises(ValueError):
        validate_parallel_scope(direction='left-to-right', extension=0., crossing_y=8.9, layout='legacy')
    assert not validate_parallel_curve([], [], goal_world_xy=(28.8, 10.5))['passed']


@pytest.mark.parametrize('mode', ['parallel', 'crossing'])
def test_native_mode_flag_is_explicit_before_any_asset_or_physics_authoring(monkeypatch, mode):
    from pxr import Usd
    from robot.scene import restaurant_human_episode as module

    class AtAuthoring(Exception):
        pass

    seen = []

    def capture(stage, **kwargs):
        seen.append(kwargs)
        raise AtAuthoring

    monkeypatch.setattr(module, 'prepare_native_restaurant_human', capture)
    with pytest.raises(AtAuthoring):
        module.RestaurantHumanEpisode(Usd.Stage.CreateInMemory(), mode=mode, output=None,
            speed_mps=.5, crossing_delay_seconds=3.)
    assert seen[0]['parallel_oncoming'] == (mode == 'parallel')


@pytest.mark.parametrize('change', [dict(crossing_delay_seconds=None), dict(speed_mps=.7),
    dict(carry_bowl=True)])
def test_parallel_cannot_silently_mix_actor_protocols(change):
    from robot.scene.restaurant_human_episode import RestaurantHumanEpisode
    args = dict(mode='parallel', output=None, speed_mps=.5, crossing_delay_seconds=3.)
    args.update(change)
    with pytest.raises(ValueError):
        RestaurantHumanEpisode(None, **args)


def test_polyline_curve_scoring_does_not_count_nominal_corner_as_avoidance():
    records, truth = example()
    route = [(7.8, 4.05), (28.8, 4.05), (28.8, 10.5)]
    accepted = validate_parallel_curve(records, truth, goal_world_xy=route[-1], nominal_route_world_xy=route)
    assert accepted['passed']
    assert accepted['nearby_walking_excursion_m'] == pytest.approx(.35)
    for r in truth:
        r['robot_xy'][0] = 28.8
    result = validate_parallel_curve(records, truth, goal_world_xy=route[-1], nominal_route_world_xy=route)
    assert not result['checks']['curved_during_nearby_parallel_walk']
    assert result['nearby_walking_excursion_m'] == pytest.approx(0., abs=1e-12)
    assert result['checks']['current_observed_oncoming_rows']


def test_service_actor_trigger_is_explicit_and_does_not_change_legacy_protocol(monkeypatch):
    from pxr import Usd
    from robot.scene import restaurant_human_episode as module
    class AtAuthoring(Exception):
        pass
    def stop(*args, **kwargs):
        assert kwargs['parallel_oncoming']
        raise AtAuthoring
    monkeypatch.setattr(module, 'prepare_native_restaurant_human', stop)
    args = dict(mode='parallel', output=None, speed_mps=.5, trigger_robot_y=9.)
    with pytest.raises(ValueError):
        module.RestaurantHumanEpisode(None, **args)
    with pytest.raises(AtAuthoring):
        module.RestaurantHumanEpisode(Usd.Stage.CreateInMemory(), **args, long_service_parallel=True)
    for change in (dict(trigger_robot_y=9.3), dict(crossing_delay_seconds=3.), dict(mode='crossing')):
        with pytest.raises(ValueError):
            module.RestaurantHumanEpisode(None, **(args | change), long_service_parallel=True)

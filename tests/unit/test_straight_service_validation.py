"""The authored v3 route changes evaluation bounds, never physical tolerances."""
import copy

import numpy as np
import pytest

from robot.evaluation.human_navigation_validation import summarize_human_navigation
from robot.evaluation.parallel_walking_validation import validate_parallel_actor, validate_parallel_curve


def straight_trace():
    return [dict(time=float(t), phase='move' if t < 33 else 'away_endpoint',
        actor_xyz=[29.8, 23.5-.5*t, 0.], robot_xy=[28.8, 19.5],
        left_foot_xyz=[0., .15*np.sin(8*t), 0.], right_foot_xyz=[0., 0., 0.],
        actor_layout_clearance_m=.4, collider_alignment_error_m=0.,
        nonzero_wheel_command_count=0, straight_service_walk=True)
        for t in np.linspace(0, 33, 331)]


def test_new_actor_bounds_require_explicit_variant_and_keep_legacy_failure():
    truth = straight_trace()
    legacy = validate_parallel_actor(truth, parked_robot=True)
    assert {k for k, v in legacy['checks'].items() if not v} == {'bounded_actor_region'}
    assert validate_parallel_actor(truth, parked_robot=True, straight_service_walk=True)['passed']
    for r in truth:
        r.pop('straight_service_walk')
    assert not validate_parallel_actor(truth, parked_robot=True, straight_service_walk=True)['passed']


@pytest.mark.parametrize('failure', ['lateral', 'region', 'reverse', 'collision', 'collider',
    'ground', 'limbs', 'robot_drift', 'wheels', 'short', 'speed'])
def test_explicit_one_way_keeps_physical_motion_and_stopping_gates(failure):
    truth = copy.deepcopy(straight_trace())
    if failure == 'lateral':
        truth[100]['actor_xyz'][0] += .3
    elif failure == 'region':
        truth[0]['actor_xyz'][1] = 24.1
    elif failure == 'reverse':
        truth[100]['actor_xyz'][1] += 1.
    elif failure == 'collision':
        truth[100]['actor_layout_clearance_m'] = .01
    elif failure == 'collider':
        truth[100]['collider_alignment_error_m'] = .04
    elif failure == 'ground':
        truth[100]['actor_xyz'][2] = .06
    elif failure == 'limbs':
        for r in truth:
            r['left_foot_xyz'] = r['right_foot_xyz']
    elif failure == 'robot_drift':
        truth[100]['robot_xy'][0] += .03
    elif failure == 'wheels':
        truth[100]['nonzero_wheel_command_count'] = 1
    elif failure == 'short':
        truth = truth[:200]
        truth[-1]['phase'] = 'away_endpoint'
    else:
        for r in truth:
            r['time'] *= 3
    assert not validate_parallel_actor(truth, parked_robot=True, straight_service_walk=True)['passed']


def test_curve_and_human_summary_use_the_declared_actor_protocol():
    truth = straight_trace()
    records = []
    for r in truth:
        t = r['time']
        r['robot_xy'] = [28.8-.35*np.exp(-(t-6)**2/9), 19.5+.2*t]
        records.append(dict(sensor_timestamp=t, objects=[dict(
            diagnostics=dict(policy_return_count=30, centroid_world_xy=r['actor_xyz'][:2]),
            attributes=dict(motion={k: dict(value=v, status='inferred', timestamp_seconds=t)
                for k, v in [('velocity_x_mps', 0.), ('velocity_y_mps', -.5)]}))],
            diagnostics=dict(applied_field_policies=[{'lambda': 1.25}])))
    goal = truth[-1]['robot_xy']
    assert not validate_parallel_curve(records, truth, goal_world_xy=goal)['passed']
    assert validate_parallel_curve(records, truth, goal_world_xy=goal, straight_service_walk=True)['passed']
    # Empty perception intentionally fails the other checks; only dispatch is tested.
    assert summarize_human_navigation([], truth, mode='parallel', straight_service_walk=True)['checks']['actual_parallel_walking']
    assert not summarize_human_navigation([], truth, mode='parallel')['checks']['actual_parallel_walking']
    with pytest.raises(ValueError):
        summarize_human_navigation([], truth, mode='standing', straight_service_walk=True)

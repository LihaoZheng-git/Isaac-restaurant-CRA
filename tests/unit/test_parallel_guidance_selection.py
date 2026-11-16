from types import SimpleNamespace as NS
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from robot.safety.motion_aware_passing import MotionAwarePassing, MotionAwarePassingConfig
from robot.safety.safe_sage_guidance_policy import PassingSide
from robot.evaluation.parallel_walking_validation import validate_parallel_selection


def step(policy, t, *, side=1., rotation=0., velocity=(0., -.5), ids=(7,), visual=True,
         point_count=100, matched=True):
    c, s = np.cos(rotation), np.sin(rotation)
    rot = np.array([[c, -s], [s, c]])
    tracks = tuple(NS(track_id=k, centroid_world_xy=rot @ np.array([side, 2.5-.5*t]),
        velocity_world_xy=rot @ np.array(velocity), point_count=point_count,
        observation_count=10, matched_previous_track=matched) for k in ids)
    return policy.update(NS(timestamp_seconds=t, tracked_obstacles=tracks), robot_xy=(0., 0.),
        robot_velocity_xy=rot @ np.array([0., .2]), goal_xy=rot @ np.array([0., 4.]),
        visual_person_track_ids=ids if visual else ())


def policy():
    return MotionAwarePassing(MotionAwarePassingConfig(encounter_geometry='oncoming_parallel',
        require_visual_person_selection=True))


@pytest.mark.parametrize('rotation', [0., .8, -2.])
@pytest.mark.parametrize('side,expected', [(1., PassingSide.LEFT), (-1., PassingSide.RIGHT)])
def test_observed_oncoming_tangent_is_forward_and_away_in_any_world_frame(side, expected, rotation):
    p = policy()
    assert not step(p, .1, side=side, rotation=rotation)[0]
    assert not step(p, .2, side=side, rotation=rotation)[0]
    selected, audit = step(p, .3, side=side, rotation=rotation)
    assert selected == {7: expected}
    assert audit['decision'] == 'selected_parallel_away_tangent'
    assert audit['original_selection_visual_person_verified']
    assert audit['encounter_selected_timestamp'] == .3
    # Same toward-obstacle tangent convention used by the field/reference.
    n = np.array([side, 2.35])
    tangent = np.array([-n[1], n[0]]) * (1 if expected == PassingSide.LEFT else -1)
    assert tangent[1] > 0 and tangent[0] * side < 0
    _, held = step(p, .4, side=side, rotation=rotation)
    assert held['encounter_selected_timestamp'] == .3


def test_primary_transverse_selector_still_rejects_parallel_motion():
    p = MotionAwarePassing(MotionAwarePassingConfig(require_visual_person_selection=True))
    for t in (.1, .2, .3):
        selected, audit = step(p, t)
    assert not selected and 'not_transverse_crossing' in audit['rejections'][7]


def test_configured_nominal_segment_keeps_parallel_definition_during_a_detour():
    fixed, bearing = policy(), policy()
    for t in (.1, .2, .3):
        track = NS(track_id=7, centroid_world_xy=(1., 2.5-.5*t), velocity_world_xy=(0., -.5),
            point_count=100, observation_count=10, matched_previous_track=True)
        args = dict(robot_xy=(0., 0.), robot_velocity_xy=(0., .2), goal_xy=(.01, 0.),
            visual_person_track_ids=(7,))
        source = NS(timestamp_seconds=t, tracked_obstacles=(track,))
        selected, audit = fixed.update(source, nominal_route_direction_world_xy=(0., 1.), **args)
        old, _ = bearing.update(source, **args)
    assert selected and not old
    assert audit['route_direction_basis'] == 'configured_obstacle_unaware_nominal_route_segment'


@pytest.mark.parametrize('change', [dict(visual=False), dict(velocity=(0., .5)),
    dict(velocity=(.5, 0.)), dict(velocity=(0., 0.)), dict(side=.05),
    dict(point_count=19), dict(matched=False), dict(ids=(7, 8))])
def test_parallel_selection_rejects_missing_or_ambiguous_current_evidence(change):
    p = policy()
    for t in (.1, .2, .3):
        selected, _ = step(p, t, **change)
    assert not selected and p.encounter is None


@pytest.mark.parametrize('change', [dict(velocity=(.5, 0.)), dict(velocity=(0., 0.)),
    dict(side=.05), dict(point_count=19), dict(matched=False), dict(ids=())])
def test_parallel_held_policy_releases_when_current_motion_or_identity_fails(change):
    p = policy()
    for t in (.1, .2, .3):
        step(p, t)
    assert p.encounter is not None
    assert not step(p, .4, **change)[0]
    assert p.encounter is None


def test_parallel_source_gap_does_not_renew_old_policy_or_transfer_identity():
    p = policy()
    for t in (.1, .2, .3):
        step(p, t)
    assert not step(p, .8)[0] and p.encounter is None
    with pytest.raises(ValueError, match='increase'):
        step(p, .8)
    p = policy()
    for t in (.1, .2, .3):
        step(p, t)
    assert not step(p, .4, ids=(8,))[0] and p.encounter is None


def test_parallel_mode_is_explicit_and_requires_visual_person_scope():
    assert MotionAwarePassingConfig(True).require_visual_person_selection
    assert MotionAwarePassingConfig(True).encounter_geometry == 'transverse_crossing'
    with pytest.raises(ValueError):
        MotionAwarePassingConfig(encounter_geometry='oncoming_parallel')
    with pytest.raises(ValueError):
        MotionAwarePassingConfig(encounter_geometry='anything')
    with pytest.raises(ValueError):
        MotionAwarePassingConfig(minimum_oncoming_alignment=1.1)


def selection_records():
    p = policy()
    records = []
    for t in (.1, .2, .3, .4, .5):
        _, audit = step(p, t)
        records.append(dict(time=t, motion_aware_passing=audit,
            applied_field_policies=[dict(track_or_group_id=7, applied_passing_side='left', **{'lambda': 1.25})], objects=[dict(
            attributes=dict(track_id=7, motion={k: dict(value=value, status='inferred', timestamp_seconds=t)
                for k, value in [('velocity_x_mps', 0.), ('velocity_y_mps', -.5)]}),
            diagnostics=dict(centroid_world_xy=[1., 2.5-.5*t]))]))
    return records


@pytest.mark.parametrize('failure', [None, 'stale', 'unverified', 'future_selection',
    'wrong_side', 'sparse', 'wrong_mode', 'duplicate_epoch', 'unapplied_field'])
def test_stationary_and_live_selection_audit_rejects_bad_epochs_and_support(failure):
    rows = selection_records()
    if failure == 'stale':
        rows[-1]['objects'][0]['attributes']['motion']['velocity_y_mps']['timestamp_seconds'] = 0.
    elif failure == 'unverified':
        rows[-1]['motion_aware_passing']['original_selection_visual_person_verified'] = False
    elif failure == 'future_selection':
        rows[-1]['motion_aware_passing']['encounter_selected_timestamp'] = .6
    elif failure == 'wrong_side':
        rows[-1]['motion_aware_passing']['preferences'][7] = 'right'
    elif failure == 'sparse':
        rows[-1]['motion_aware_passing']['qualified_candidates'][0]['positive_lidar_point_count'] = 1
    elif failure == 'wrong_mode':
        rows[-1]['motion_aware_passing']['encounter_geometry'] = 'transverse_crossing'
    elif failure == 'duplicate_epoch':
        rows = [rows[-1]] * 3
    elif failure == 'unapplied_field':
        rows[-1]['applied_field_policies'][0]['lambda'] = 0.
    result = validate_parallel_selection(rows)
    assert result['passed'] == (failure is None)
    if failure is None:
        assert result['distinct_selected_sources'] == 3
        live = [dict(sensor_timestamp=r['time'], diagnostics=dict(motion_aware_passing=r['motion_aware_passing'],
            applied_field_policies=r['applied_field_policies']), objects=r['objects']) for r in rows]
        assert validate_parallel_selection(live)['passed']
    assert not validate_parallel_selection([])['passed']


@pytest.mark.parametrize('option,value', [('--restaurant-human', 'crossing'),
    ('--guidance-tracking-weight', '0'), ('--guidance-tangent-velocity-fraction', '0'),
    ('--guidance-tangent-basis', 'planar_gradient'),
    ('--guidance-reference-mode', 'bounded_field_heading'),
    ('--experiment-baseline', 'geometric_velocity_cbf'),
    ('--motion-aware-passing', None), ('--human-observed-passing-target', None)])
def test_parallel_cli_rejects_unlabeled_or_incompatible_objectives_before_isaac(tmp_path, option, value):
    script = Path(__file__).resolve().parents[2] / 'scripts/demos/cra_live_object_isaac_demo.py'
    argv = ['--output', str(tmp_path / 'unused'), '--parallel-guidance-ablation',
        '--restaurant-human', 'parallel', '--motion-aware-passing', '--human-observed-passing-target',
        '--guidance-tracking-weight', '100', '--guidance-tangent-velocity-fraction', '1',
        '--guidance-tangent-basis', 'selected_observed_obstacle', '--guidance-reference-mode', 'planar_forecast_velocity']
    if value is None:
        argv.remove(option)
    elif option in argv:
        argv[argv.index(option)+1] = value
    else:
        argv.extend([option, value])
    result = subprocess.run([sys.executable, str(script), *argv], capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
    assert 'parallel guidance ablation requires' in result.stderr
    assert not (tmp_path / 'unused').exists()

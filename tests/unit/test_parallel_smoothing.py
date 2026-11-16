"""Actual-motion scoring cannot turn a stalled or failed trial into a pass."""

import copy
from dataclasses import replace

import numpy as np
import pytest
from test_parallel_route_guides import planner
from test_selected_tangent_reference import field

from robot.evaluation.parallel_smoothing import (
    activation_only_command_change, compare_parallel_smoothness, summarize_parallel_smoothness,
)


def trace(hold=False, *, robot_speed=.15):
    rows = []
    for t in np.arange(0., 8.01, .05):
        x = robot_speed * (min(t, 2.) + max(0., t-5.) if hold else t)
        rows.append(dict(time=float(t), phase='move', robot_xy=[x, 0.],
            actor_xyz=[1., 1.6-.4*t, 0.], robot_command=[.2, .8]))
    report = dict(passed=True, native_human_mode='parallel', fixed_lambda=1.25,
        human_navigation_validation=dict(parallel_curve=dict(passed=True)),
        minimum_clearance_m=.3, navigation_simulation_seconds=8.)
    return report, rows


def test_true_translation_hold_is_counted_despite_nonzero_motion_and_rotation_commands():
    result = summarize_parallel_smoothness(*trace(hold=True))
    assert result['eligible']
    assert result['longest_translation_hold']['duration_seconds'] == pytest.approx(2.9)
    assert result['longest_slow_translation']['duration_seconds'] >= 2.9
    assert summarize_parallel_smoothness(*trace())['longest_translation_hold']['duration_seconds'] == 0


def test_creeping_is_reported_separately_from_a_full_translation_hold():
    result = summarize_parallel_smoothness(*trace(robot_speed=.03))
    assert result['longest_translation_hold']['duration_seconds'] == 0
    assert result['longest_slow_translation']['duration_seconds'] > 3


@pytest.mark.parametrize('failure', ['native', 'curve', 'lambda', 'mode', 'no_nearby'])
def test_missing_or_failed_native_encounter_cannot_be_smooth_acceptance(failure):
    report, rows = trace()
    if failure == 'native':
        report['passed'] = False
    elif failure == 'curve':
        report['human_navigation_validation']['parallel_curve']['passed'] = False
    elif failure == 'lambda':
        report['fixed_lambda'] = 0.
    elif failure == 'mode':
        report['native_human_mode'] = 'crossing'
    else:
        for row in rows:
            row['actor_xyz'][0] = 20.
    candidate = summarize_parallel_smoothness(report, rows)
    assert not candidate['eligible']
    baseline = summarize_parallel_smoothness(*trace(hold=True))
    assert not compare_parallel_smoothness(baseline, candidate, matched_configuration=True)['passed']


@pytest.mark.parametrize('failure', ['short', 'duplicate', 'reversed', 'gap', 'nonfinite'])
def test_incomplete_source_timing_is_rejected(failure):
    report, rows = trace()
    if failure == 'short':
        rows = rows[:10]
    elif failure == 'duplicate':
        rows[5]['time'] = rows[4]['time']
    elif failure == 'reversed':
        rows.reverse()
    elif failure == 'gap':
        rows = rows[:50] + rows[70:]
    else:
        rows[5]['robot_xy'][0] = np.nan
    with pytest.raises(ValueError):
        summarize_parallel_smoothness(report, rows)


def test_rotation_and_translation_preserve_actual_hold_score():
    report, rows = trace(hold=True)
    original = summarize_parallel_smoothness(report, rows)
    angle = 1.3
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    for row in rows:
        row['robot_xy'] = rotation @ row['robot_xy'] + [28., 7.]
        row['actor_xyz'][:2] = rotation @ row['actor_xyz'][:2] + [28., 7.]
    rotated = summarize_parallel_smoothness(report, rows)
    assert rotated['longest_translation_hold'] == original['longest_translation_hold']


def test_smoothness_target_keeps_native_acceptance_and_matched_parameters_mandatory():
    baseline = summarize_parallel_smoothness(*trace(hold=True))
    candidate = summarize_parallel_smoothness(*trace())
    assert compare_parallel_smoothness(baseline, candidate, matched_configuration=True)['passed']
    assert not compare_parallel_smoothness(baseline, candidate, matched_configuration=False)['passed']
    assert not compare_parallel_smoothness(baseline, baseline, matched_configuration=True)['passed']
    candidate = copy.deepcopy(candidate)
    candidate['longest_translation_hold']['duration_seconds'] = .6
    assert not compare_parallel_smoothness(baseline, candidate, matched_configuration=True)['passed']


def command(activation='2.0'):
    return ['isaac-python', '-u', 'demo.py', '--output', 'old', '--guidance-activation-h', activation,
        '--fixed-lambda', '1.25', '--speed', '.2']


@pytest.mark.parametrize('change', ['output_only', 'earlier', 'lambda', 'speed', 'lower_activation'])
def test_activation_only_comparison_rejects_other_changed_parameters(change):
    a, b = command(), command('3.0')
    b[b.index('--output')+1] = 'new'
    if change == 'output_only':
        b[b.index('--guidance-activation-h')+1] = '2.0'
    elif change in {'lambda', 'speed'}:
        b[b.index('--fixed-lambda' if change == 'lambda' else '--speed')+1] = '9'
    elif change == 'lower_activation':
        b[b.index('--guidance-activation-h')+1] = '1.0'
    assert activation_only_command_change(a, b)['matched'] == (change == 'earlier')


@pytest.mark.parametrize('bad', ['nan', '-1', 'inf'])
def test_invalid_activation_rejected(bad):
    with pytest.raises(ValueError):
        activation_only_command_change(command(), command(bad))


@pytest.mark.parametrize('side,sign', [('left', 1.), ('right', -1.)])
def test_earlier_activation_bends_before_legacy_onset_with_unchanged_exact_checks(side, sign):
    kwargs = field(-sign, gradient=(0., 0., 0.), h=2.5)
    nominal = np.tile([.2, 0.], (6, 1))
    before = nominal.copy()
    results = []
    for activation in (2., 3.):
        m = planner()
        m.guidance_config = replace(m.guidance_config, activation_safety_value=activation)
        result = m.plan(np.zeros(3), nominal, nominal_goal_grid_xy=(2., 0.),
            selected_passing_tangents=((7, (1., -sign), side),), **kwargs)
        assert result.success
        checked = m._evaluate_trajectory(np.zeros(3), result.command_sequence,
            temporal_safety_derivative=0., **kwargs)
        assert checked is not None and m._trajectory_is_safe(checked)
        results.append(result)
    assert not results[0].guidance_tracking.active
    assert results[1].guidance_tracking.active
    assert sign*results[1].angular_velocity > .01
    np.testing.assert_array_equal(nominal, before)

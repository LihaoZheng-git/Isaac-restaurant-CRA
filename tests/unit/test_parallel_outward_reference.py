"""Current observed travel preference adds no safety authority or old evidence."""

from dataclasses import fields, replace
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from test_parallel_route_guides import planner
from test_selected_tangent_reference import field, runtime_field

from robot.evaluation.parallel_smoothing import outward_only_command_change
from robot.safety.safe_sage_runtime import SafeSAGEControlRuntime
from robot.safety.safe_sage_unicycle_cbf_filter import SafeSAGEUnicycleCBFFilter


def outward_planner(value=.35):
    m = planner()
    m.guidance_config = replace(m.guidance_config, outward_reference_fraction=value)
    return m


@pytest.mark.parametrize('side,sign', [('left', 1.), ('right', -1.)])
@pytest.mark.parametrize('yaw', [0., np.pi/2, -2.4])
def test_outward_preference_is_rotationally_consistent_and_speed_bounded(side, sign, yaw):
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    state = np.array([0., 0., yaw])
    goal, center = rotation @ [2., 0.], rotation @ [1., -sign]
    kwargs = dict(field(gradient=(0., 0., 0.)), selected_passing_tangents=((7, center, side),))
    directions = []
    for value in (0., .35):
        m = outward_planner(value)
        heading, speed = m._reference_preference(state, goal, .2, m._sample(state, kwargs), kwargs)
        assert 0 < speed <= .2
        directions.append(sign*((heading-yaw+np.pi) % (2*np.pi)-np.pi))
    assert directions[1] > directions[0] > 0


@pytest.mark.parametrize('side,sign', [('left', 1.), ('right', -1.)])
def test_outward_reference_and_result_both_keep_full_exact_checks_and_nominal(side, sign):
    m = outward_planner()
    nominal = np.tile([.2, 0.], (6, 1))
    original = nominal.copy()
    kwargs = field(-sign)
    result = m.plan(np.zeros(3), nominal, nominal_goal_grid_xy=(2., 0.),
        selected_passing_tangents=((7, (1., -sign), side),), **kwargs)
    assert result.success and result.guidance_tracking.active
    assert result.guidance_tracking.outward_reference_fraction == .35
    assert sign*result.angular_velocity > .01
    for commands in (result.command_sequence, result.guidance_tracking.reference_commands):
        checked = m._evaluate_trajectory(np.zeros(3), np.array(commands),
            temporal_safety_derivative=0., **kwargs)
        assert checked is not None and m._trajectory_is_safe(checked)
    np.testing.assert_array_equal(nominal, original)


def test_zero_outward_default_keeps_complete_result_identical():
    results = [m.plan(np.zeros(3), (.2, 0.), nominal_goal_grid_xy=(2., 0.),
        selected_passing_tangents=((7, (1., -1.), 'left'),), **field())
        for m in (planner(), outward_planner(0.))]
    for item in fields(results[0]):
        a, b = (getattr(r, item.name) for r in results)
        if isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        else:
            assert a == b


def test_no_current_selection_disables_outward_preference_locally():
    m, f = outward_planner(), runtime_field()
    runtime = SafeSAGEControlRuntime(mpc=m, backup_filter=SafeSAGEUnicycleCBFFilter(m.paper_filter))
    result = runtime.plan(np.zeros(3), (.2, 0.), f, nominal_goal_grid_xy=(1., 0.))
    assert result.guidance_tracking.outward_reference_fraction == 0.
    assert result.guidance_tracking.tangent_velocity_fraction == 0.
    assert m.guidance_config.outward_reference_fraction == .35
    assert m.guidance_config.tangent_velocity_fraction == 1.
    assert f.sensor_timestamp_seconds == 1. and f.motion_passing_reference_track_ids == ()


@pytest.mark.parametrize('kwargs', [dict(h=-.1), dict(h=.1)])
def test_outward_preference_cannot_rescue_unsafe_geometry_or_temporal_horizon(kwargs):
    m = outward_planner()
    inputs = field(**kwargs)
    if kwargs['h'] > 0:
        inputs['temporal_safety_derivative'] = -1.
    result = m.plan(np.zeros(3), (.2, 0.), nominal_goal_grid_xy=(2., 0.),
        selected_passing_tangents=((7, (1., -1.), 'left'),), **inputs)
    assert not result.success and result.first_command == (0., 0.)


@pytest.mark.parametrize('value', [-.1, 1.1, True, np.nan, np.inf])
def test_invalid_outward_fraction_rejected(value):
    with pytest.raises(ValueError):
        outward_planner(value)


@pytest.mark.parametrize('changes', [dict(tangent_blend_mode='additive'),
    dict(reference_tangent_basis='planar_gradient'), dict(reference_mode='filtered_goal_velocity', tangent_velocity_fraction=0.),
    dict(weight=0., tangent_velocity_fraction=0.)])
def test_outward_requires_explicit_selected_convex_forecast_ablation(changes):
    with pytest.raises(ValueError):
        replace(outward_planner().guidance_config, **changes)


@pytest.mark.parametrize('value', ['.35', '-.1', 'nan'])
def test_outward_cli_cannot_enable_unlabeled_reference_before_isaac(tmp_path, value):
    script = Path(__file__).resolve().parents[2]/'scripts/demos/cra_live_object_isaac_demo.py'
    result = subprocess.run([sys.executable, str(script), '--output', str(tmp_path/'unused'),
        '--guidance-outward-reference-fraction', value], capture_output=True, text=True, timeout=10)
    assert result.returncode == 2 and 'outward reference requires' in result.stderr
    assert not (tmp_path/'unused').exists()


@pytest.mark.parametrize('change', ['outward', 'activation', 'lambda', 'body', 'duplicate'])
def test_outward_comparison_rejects_hidden_control_or_safety_changes(change):
    a = ['isaac', '--output', 'old', '--guidance-activation-h', '2.0', '--lambda', '1.25', '--body', '.65']
    b = a.copy()
    b[2] = 'new'
    b += ['--guidance-outward-reference-fraction', '.35']
    if change == 'duplicate':
        b += ['--guidance-outward-reference-fraction', '.35']
        with pytest.raises(ValueError):
            outward_only_command_change(a, b)
        return
    if change != 'outward':
        key = {'activation':'--guidance-activation-h', 'lambda':'--lambda', 'body':'--body'}[change]
        b[b.index(key)+1] = '9'
    audit = outward_only_command_change(a, b)
    assert audit['matched'] == (change == 'outward')
    assert audit['baseline_outward_fraction'] == 0. and audit['candidate_outward_fraction'] == .35

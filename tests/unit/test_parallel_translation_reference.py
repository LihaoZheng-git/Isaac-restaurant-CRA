"""A reference preference never overrides projection, turn-only or exact safety."""

from dataclasses import fields, replace
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from test_parallel_route_guides import planner
from test_selected_tangent_reference import field, runtime_field

from robot.evaluation.parallel_smoothing import translation_only_command_change
from robot.safety.safe_sage_runtime import SafeSAGEControlRuntime
from robot.safety.safe_sage_unicycle_cbf_filter import SafeSAGEUnicycleCBFFilter


def preferred(value=.75):
    m = planner()
    m.guidance_config = replace(m.guidance_config, turn_translation_fraction=value)
    return m


@pytest.mark.parametrize('side,sign', [('left', 1.), ('right', -1.)])
def test_translation_preference_and_result_retain_exact_horizon_and_original_nominal(side, sign):
    m = preferred()
    nominal = np.tile([.2, 0.], (6, 1))
    before = nominal.copy()
    inputs = field(-sign)
    result = m.plan(np.zeros(3), nominal, nominal_goal_grid_xy=(2., 0.),
        selected_passing_tangents=((7, (1., -sign), side),), **inputs)
    assert result.success and result.guidance_tracking.active
    assert result.guidance_tracking.turn_translation_fraction == .75
    assert sign*result.angular_velocity > .01
    for commands in (result.command_sequence, result.guidance_tracking.reference_commands):
        commands = np.array(commands)
        assert (commands[:, 0] <= m.config.maximum_linear_velocity).all()
        assert (np.abs(commands[:, 1]) <= m.config.maximum_angular_velocity).all()
        evaluated = m._evaluate_trajectory(np.zeros(3), commands,
            temporal_safety_derivative=0., **inputs)
        assert evaluated is not None and m._trajectory_is_safe(evaluated)
    np.testing.assert_array_equal(nominal, before)


def test_turn_in_place_or_backward_heading_never_receives_a_forced_forward_floor():
    m = preferred(1.)
    args = dict(field(gradient=(0., 0., 0.)), selected_passing_tangents=((7, (1., -1.), 'left'),))
    turn = m._guidance_rollout(np.zeros(3), np.array([2., 0.]), .2, .8, True, args)
    backward = m._guidance_rollout(np.array([0., 0., np.pi]), np.array([2., 0.]), .2, .8, False, args)
    assert turn is not None and turn[0, 0] == 0.
    assert backward is not None and backward[0, 0] == 0.
    near = m._guidance_rollout(np.zeros(3), np.array([.03, 0.]), .2, .8, False, args)
    np.testing.assert_array_equal(near, 0.)


def test_affine_safety_projection_can_reduce_the_preference_below_its_requested_fraction(monkeypatch):
    import robot.safety.laplace_guidance_tracking_mpc as module
    original = module.project_weighted_box_halfspace

    def tighter_halfspace(preferred_command, **kwargs):
        # An independent control-space halfspace v<=.02 must win over the
        # .15m/s preference; exercise the real box/halfspace projector.
        kwargs.update(constraint_normal=np.array([-1., 0.]), constraint_rhs=-.02)
        return original(preferred_command, **kwargs)

    monkeypatch.setattr(module, 'project_weighted_box_halfspace', tighter_halfspace)
    m = preferred()
    args = dict(field(gradient=(0., 0., 0.)), selected_passing_tangents=((7, (1., -1.), 'left'),))
    commands = m._guidance_rollout(np.zeros(3), np.array([2., 0.]), .2, .8, False, args)
    assert commands is not None and np.max(commands[:, 0]) <= .02 + 1e-9


@pytest.mark.parametrize('inputs', [dict(h=-.1), dict(h=.1, temporal_safety_derivative=-1.)])
def test_translation_preference_cannot_rescue_unsafe_geometry_or_future_horizon(inputs):
    args = field(h=inputs['h'])
    args['temporal_safety_derivative'] = inputs.get('temporal_safety_derivative', 0.)
    result = preferred(1.).plan(np.zeros(3), (.2, 0.), nominal_goal_grid_xy=(2., 0.),
        selected_passing_tangents=((7, (1., -1.), 'left'),), **args)
    assert not result.success and result.first_command == (0., 0.)


def test_default_zero_retains_complete_old_result_and_current_selection_gates():
    results = [m.plan(np.zeros(3), (.2, 0.), nominal_goal_grid_xy=(2., 0.),
        selected_passing_tangents=((7, (1., -1.), 'left'),), **field()) for m in (planner(), preferred(0.))]
    for item in fields(results[0]):
        a, b = [getattr(r, item.name) for r in results]
        if isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        else:
            assert a == b
    m, f = preferred(), runtime_field()
    runtime = SafeSAGEControlRuntime(mpc=m, backup_filter=SafeSAGEUnicycleCBFFilter(m.paper_filter))
    result = runtime.plan(np.zeros(3), (.2, 0.), f, nominal_goal_grid_xy=(1., 0.))
    assert result.guidance_tracking.turn_translation_fraction == 0.
    assert m.guidance_config.turn_translation_fraction == .75
    assert f.sensor_timestamp_seconds == 1.


@pytest.mark.parametrize('value', [-.1, 1.1, True, np.nan])
def test_invalid_reference_fraction_rejected(value):
    with pytest.raises(ValueError):
        preferred(value)


@pytest.mark.parametrize('changes', [dict(tangent_blend_mode='additive'), dict(weight=0., tangent_velocity_fraction=0.),
    dict(reference_tangent_basis='planar_gradient')])
def test_translation_requires_explicit_selected_convex_forecast_scope(changes):
    with pytest.raises(ValueError):
        replace(preferred().guidance_config, **changes)


@pytest.mark.parametrize('value', ['.75', '-.1', 'nan'])
def test_cli_rejects_unlabeled_turn_preference_before_isaac(tmp_path, value):
    script = Path(__file__).resolve().parents[2]/'scripts/demos/cra_live_object_isaac_demo.py'
    result = subprocess.run([sys.executable, str(script), '--output', str(tmp_path/'unused'),
        '--guidance-turn-translation-fraction', value], capture_output=True, text=True, timeout=10)
    assert result.returncode == 2 and 'turn translation requires' in result.stderr
    assert not (tmp_path/'unused').exists()


@pytest.mark.parametrize('change', [None, 'identity', 'activation', 'lambda', 'body', 'duplicate'])
def test_comparison_only_allows_the_single_declared_turning_preference(change):
    a = ['isaac', '--output', 'old', '--human-passing-identity-prior-max-age', '0',
        '--guidance-activation-h', '2', '--lambda', '1.25', '--body', '.65']
    b = a.copy()
    b[2] = 'new'
    b += ['--guidance-turn-translation-fraction', '.75']
    if change == 'duplicate':
        b += ['--guidance-turn-translation-fraction', '.75']
        with pytest.raises(ValueError):
            translation_only_command_change(a, b)
    else:
        if change:
            flag = {'identity': '--human-passing-identity-prior-max-age', 'activation': '--guidance-activation-h',
                'lambda': '--lambda', 'body': '--body'}[change]
            b[b.index(flag)+1] = '9'
        assert translation_only_command_change(a, b)['matched'] == (change is None)

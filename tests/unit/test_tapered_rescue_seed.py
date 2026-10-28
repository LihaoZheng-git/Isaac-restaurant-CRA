"""Numerical seeds preserve exact acceptance and original solver budgets."""

from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytest
from test_safe_sage_cbf_filter import make_fields
from test_safe_sage_unicycle_mpc import make_mpc

from robot.evaluation.restaurant_comparison import planner_result_diagnostic
from robot.safety.safe_sage_unicycle_mpc import SafeSAGEUnicycleMPCConfig


def case(temporal=-.2):
    values = make_fields(safety_value=.8, gradient=(-1., 0., 0.), guidance=(-1., 0.))
    return dict(zip(('configuration_space', 'h_full', 'guidance_fields'), values),
        temporal_safety_derivative=temporal)


def mpc():
    return make_mpc(horizon_steps=6, enable_tapered_rescue_seed=True,
        maximum_nonlinear_rescue_iterations=20, safety_substeps_per_interval=4,
        integration_substeps_per_interval=4)


@pytest.mark.parametrize('value', [1, 'yes', None])
def test_seed_opt_in_requires_explicit_boolean(value):
    with pytest.raises(TypeError, match='enable_tapered_rescue_seed'):
        SafeSAGEUnicycleMPCConfig(enable_tapered_rescue_seed=value)


def test_seed_requires_three_intervals_without_changing_legacy_short_horizon():
    assert SafeSAGEUnicycleMPCConfig(horizon_steps=2).horizon_steps == 2
    with pytest.raises(ValueError, match='three intervals'):
        SafeSAGEUnicycleMPCConfig(horizon_steps=2, enable_tapered_rescue_seed=True)


def test_safe_reference_is_retained_without_candidate_search():
    controller = mpc()
    kwargs = case(0.)
    nominal = controller._nominal_sequence((.2, 0.))
    reference = controller._evaluate_trajectory(np.zeros(3), nominal, **kwargs)
    assert reference is not None and controller._trajectory_is_safe(reference)
    result, audit = controller._select_tapered_rescue_seed(reference, initial_state=np.zeros(3),
        nominal_sequence=nominal, target_chassis_states=None, **kwargs)
    assert result is reference and not audit['selected'] and audit['tested_candidates'] == 0


def test_tapered_seed_restores_real_temporal_feasibility_with_unchanged_geometry():
    controller = mpc()
    kwargs = case()
    original_values = kwargs['h_full'].values.copy()
    nominal = controller._nominal_sequence((.3, 0.))
    reference = controller._evaluate_trajectory(np.zeros(3), nominal, **kwargs)
    assert reference is not None and not controller._trajectory_is_safe(reference)
    result, audit = controller._select_tapered_rescue_seed(reference, initial_state=np.zeros(3),
        nominal_sequence=nominal, target_chassis_states=None, **kwargs)
    assert audit['selected'] and audit['tested_candidates'] == 1050
    assert controller._maximum_violation(result) < controller._maximum_violation(reference)
    assert controller._trajectory_is_safe(result)
    assert result.commands[0, 0] > .02
    assert np.all((result.commands[:, 0] >= 0.) & (result.commands[:, 0] <= .3))
    assert np.all(np.abs(result.commands[:, 1]) <= 1.)
    assert controller.revalidate_commands(np.zeros(3), result.commands, **kwargs)
    np.testing.assert_array_equal(kwargs['h_full'].values, original_values)
    np.testing.assert_array_equal(nominal, np.tile((.3, 0.), (6, 1)))


def test_failed_qp_uses_only_one_original_budget_rescue_and_exact_accepted_result(monkeypatch):
    controller = mpc()
    monkeypatch.setattr(controller, '_solve_sparse_subproblem',
        lambda *a, **k: SimpleNamespace(success=False, status='forced QP failure', iteration_count=0))
    original = controller._solve_exact_nonlinear_rescue
    calls = []
    def rescue(reference, **kwargs):
        calls.append(controller.config.maximum_nonlinear_rescue_iterations)
        return original(reference, **kwargs)
    monkeypatch.setattr(controller, '_solve_exact_nonlinear_rescue', rescue)
    result = controller.plan(np.zeros(3), (.3, 0.), **case())
    assert calls == [20] and result.success and not result.fallback_active
    assert result.tapered_rescue_seed['selected']
    assert result.minimum_dtcbf_margin >= -controller.config.constraint_tolerance
    assert controller.revalidate_commands(np.zeros(3), result.command_sequence, **case())
    assert planner_result_diagnostic(result)['tapered_rescue_seed'] == result.tapered_rescue_seed


def test_infeasible_seed_never_becomes_execution_permission(monkeypatch):
    controller = mpc()
    kwargs = case(-.4)
    monkeypatch.setattr(controller, '_solve_sparse_subproblem',
        lambda *a, **k: SimpleNamespace(success=False, status='forced QP failure', iteration_count=0))
    monkeypatch.setattr(controller, '_solve_exact_nonlinear_rescue', lambda *a, **k: (None, 20, 'forced failure'))
    monkeypatch.setattr(controller, '_recover_feasible_command_horizon', lambda *a, **k: (None, 'forced failure'))
    result = controller.plan(np.zeros(3), (.3, 0.), **kwargs)
    assert not result.success and result.fallback_active and result.first_command == (0., 0.)
    assert result.tapered_rescue_seed is not None


def test_unsafe_current_geometry_still_stops_before_rescue_search():
    controller = mpc()
    values = make_fields(safety_value=0., unsafe_yaw_indices=(2,))
    result = controller.plan(np.zeros(3), (.2, 0.),
        **dict(zip(('configuration_space', 'h_full', 'guidance_fields'), values)))
    assert not result.success and result.first_command == (0., 0.)
    assert result.status == 'unsafe_configuration_boundary'
    assert result.tapered_rescue_seed is None


def test_explicit_disabled_option_matches_entire_legacy_result():
    results = [controller.plan(np.zeros(3), (.2, 0.), **case(0.)) for controller in (
        make_mpc(horizon_steps=6), make_mpc(horizon_steps=6, enable_tapered_rescue_seed=False))]
    for key in fields(results[0]):
        a, b = getattr(results[0], key.name), getattr(results[1], key.name)
        if isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        else:
            assert a == b

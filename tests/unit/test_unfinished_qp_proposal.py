"""Iteration-limited QP iterates are proposals, never execution permission."""
from types import SimpleNamespace
import numpy as np
import pytest
from test_safe_sage_unicycle_mpc import make_mpc
from test_tapered_rescue_seed import case
from robot.safety.safe_sage_unicycle_mpc import SafeSAGEUnicycleMPCConfig
from robot.evaluation.restaurant_comparison import planner_result_diagnostic


def fake_qp(monkeypatch, *, command=(.1, 0.), status='maximum iterations reached', malformed=None):
    class Solver:
        def setup(self, **kwargs):
            self.size = len(kwargs['q'])
        def warm_start(self, **kwargs):
            pass
        def solve(self, **kwargs):
            horizon = self.size // 3
            vector = np.concatenate([np.tile(command, horizon), np.zeros(horizon)])
            if malformed == 'nan':
                vector[0] = np.nan
            if malformed == 'short':
                vector = vector[:-1]
            if malformed == 'none':
                vector = None
            return SimpleNamespace(x=vector, info=SimpleNamespace(status=status, iter=1000,
                prim_res=.01, dual_res=.02))
    monkeypatch.setattr('robot.safety.safe_sage_unicycle_mpc.osqp.OSQP', Solver)


def proposal(controller, audit):
    kwargs = case(0.)
    commands = controller._nominal_sequence((.2, 0.))
    reference = controller._evaluate_trajectory(np.zeros(3), commands, **kwargs)
    return controller._solve_sparse_subproblem(reference, nominal_sequence=commands,
        configuration_space=kwargs['configuration_space'], derivative_mode='gradient', proposal_audit=audit)


@pytest.mark.parametrize('value', [1, 'yes', None])
def test_unfinished_proposal_requires_explicit_boolean(value):
    with pytest.raises(TypeError, match='enable_unfinished_qp_proposal'):
        SafeSAGEUnicycleMPCConfig(enable_unfinished_qp_proposal=value)


def test_legacy_iteration_limit_stays_rejected(monkeypatch):
    fake_qp(monkeypatch)
    audit = []
    result = proposal(make_mpc(horizon_steps=6), audit)
    assert not result.success and audit == []


def test_enabled_finite_iterate_is_labeled_numerical_proposal(monkeypatch):
    fake_qp(monkeypatch)
    audit = []
    result = proposal(make_mpc(horizon_steps=6, enable_unfinished_qp_proposal=True), audit)
    assert result.success and result.status == 'unfinished_qp_numerical_proposal'
    assert audit[0]['solver_status'] == 'maximum iterations reached'
    assert audit[0]['basis'] == 'unfinished_QP_numerical_proposal_not_execution_permission'
    assert audit[0]['iterations'] == 1000 and audit[0]['primal_residual'] == .01


@pytest.mark.parametrize('status,malformed', [
    ('primal infeasible', None), ('dual infeasible', None), ('interrupted', None),
    ('problem non convex', None), ('maximum iterations reached', 'nan'),
    ('maximum iterations reached', 'short'), ('maximum iterations reached', 'none'),
])
def test_only_finite_correctly_shaped_iteration_limited_proposals_are_allowed(monkeypatch, status, malformed):
    fake_qp(monkeypatch, status=status, malformed=malformed)
    audit = []
    result = proposal(make_mpc(horizon_steps=6, enable_unfinished_qp_proposal=True), audit)
    assert not result.success and not audit


def test_unfinished_proposal_must_pass_full_original_horizon(monkeypatch):
    fake_qp(monkeypatch, command=(.1, 0.))
    controller = make_mpc(horizon_steps=6, enable_unfinished_qp_proposal=True,
        maximum_solver_iterations=1000, maximum_nonlinear_rescue_iterations=20)
    kwargs = case(-.2)
    result = controller.plan(np.zeros(3), (.3, 0.), **kwargs)
    assert result.success and result.unfinished_qp_proposals
    assert controller.revalidate_commands(np.zeros(3), result.command_sequence, **kwargs)
    assert result.minimum_dtcbf_margin >= -controller.config.constraint_tolerance
    assert result.command_sequence.shape == (6, 2)
    assert planner_result_diagnostic(result)['unfinished_qp_proposals'] == result.unfinished_qp_proposals
    assert controller.config.time_step * controller.config.horizon_steps == pytest.approx(1.2)


def test_unsafe_unfinished_proposal_cannot_bypass_exact_checks(monkeypatch):
    fake_qp(monkeypatch, command=(.3, 0.))
    controller = make_mpc(horizon_steps=6, enable_unfinished_qp_proposal=True,
        maximum_sqp_iterations=2, maximum_restoration_iterations=2)
    monkeypatch.setattr(controller, '_solve_exact_nonlinear_rescue', lambda *a, **k: (None, 20, 'forced failure'))
    monkeypatch.setattr(controller, '_recover_feasible_command_horizon', lambda *a, **k: (None, 'forced failure'))
    result = controller.plan(np.zeros(3), (.3, 0.), **case(-.4))
    assert not result.success and result.fallback_active and result.first_command == (0., 0.)
    assert result.unfinished_qp_proposals


def test_existing_safe_warm_start_is_preserved_when_unfinished_step_is_unsafe(monkeypatch):
    fake_qp(monkeypatch, command=(.3, 0.))
    controller = make_mpc(horizon_steps=6, enable_unfinished_qp_proposal=True)
    kwargs = case(-.2)
    warm = np.tile((.05, 0.), (6, 1))
    result = controller.plan(np.zeros(3), (.3, 0.), initial_command_guess=warm, **kwargs)
    assert result.success and result.unfinished_qp_proposals
    assert controller.revalidate_commands(np.zeros(3), result.command_sequence, **kwargs)
    assert result.first_command[0] < .3

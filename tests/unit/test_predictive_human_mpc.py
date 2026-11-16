from dataclasses import replace

import numpy as np
import pytest
from test_predictive_human_geometry import FOOTPRINT, geometry, prediction
from test_safe_sage_cbf_filter import make_fields, make_filter
from test_selected_tangent_reference import runtime_field

from robot.safety.predictive_human_geometry import HumanGeometryForecastConfig
from robot.safety.safe_sage_configuration_space import RectangularRobotFootprint
from robot.safety.safe_sage_runtime import SafeSAGEControlRuntime
from robot.safety.safe_sage_unicycle_cbf_filter import SafeSAGEUnicycleCBFFilter
from robot.safety.safe_sage_unicycle_mpc import SafeSAGEUnicycleMPC, SafeSAGEUnicycleMPCConfig


def inputs():
    cs, h, guide = make_fields()
    return dict(configuration_space=replace(cs, footprint=FOOTPRINT), h_full=h, guidance_fields=guide)


def mpc(prediction=None, **kwargs):
    return SafeSAGEUnicycleMPC(make_filter(), SafeSAGEUnicycleMPCConfig(
        horizon_steps=6, maximum_nonlinear_rescue_iterations=20, **kwargs), predictive_geometry=prediction)


def test_default_and_far_forecast_keep_old_command_but_report_separate_prior():
    old, far = mpc(), mpc(geometry(prediction(center=(4., 1.), velocity=(0., 0.))))
    a, b = [m.plan(np.zeros(3), (.2, 0.), **inputs()) for m in (old, far)]
    assert a.success and b.success and a.predictive_geometry is None
    np.testing.assert_allclose(a.command_sequence, b.command_sequence, atol=1e-8, rtol=0)
    assert b.predictive_geometry['status'] == 'prior'
    assert b.predictive_geometry['minimum_endpoint_margin_m'] > 0
    assert old.predictive_geometry is None


def test_additional_prediction_rejects_a_future_collision_that_current_empty_field_accepts():
    old, future = mpc(), mpc(geometry())
    nominal = np.tile([.2, 0.], (6, 1))
    assert old.revalidate_commands(np.zeros(3), nominal, **inputs())
    assert not future.revalidate_commands(np.zeros(3), nominal, **inputs())
    result = future.plan(np.zeros(3), nominal, **inputs())
    assert not result.success or future.revalidate_commands(np.zeros(3), result.command_sequence, **inputs())
    assert not np.array_equal(result.command_sequence, nominal)


@pytest.mark.parametrize('floor', [None, .3])
def test_infeasible_main_reference_is_retained_for_restoration_and_hard_qp_rows_have_no_slack(monkeypatch, floor):
    import robot.safety.safe_sage_unicycle_mpc as module
    cfg = HumanGeometryForecastConfig(velocity_uncertainty_prior_mps=.5 if floor else .25,
        velocity_uncertainty_floor_prior_mps=floor)
    m = mpc(geometry(prediction(config=cfg)))
    reference = m._evaluate_trajectory(np.zeros(3), np.tile([.2, 0.], (6, 1)),
        temporal_safety_derivative=0., **inputs())
    assert reference is not None and np.min(reference.predictive_margins) < 0
    assert not m._trajectory_is_safe(reference)
    assert np.isfinite(m._maximum_violation(reference))
    captured, original = {}, module.osqp.OSQP.setup

    def capture(self, **kwargs):
        captured.update(kwargs)
        return original(self, **kwargs)

    monkeypatch.setattr(module.osqp.OSQP, 'setup', capture)
    m._solve_sparse_subproblem(reference, nominal_sequence=reference.commands,
        configuration_space=inputs()['configuration_space'], derivative_mode='gradient')
    matrix = captured['A'].toarray()
    assert matrix.shape == (42, 18)
    np.testing.assert_array_equal(matrix[-6:, 12:], 0.)
    # Independently perturb commands and differentiate the true margins.
    sensitivity = m._rollout_state_sensitivities(reference.chassis_states, reference.commands)
    analytic = np.concatenate([g@sensitivity[k] for k, g in enumerate(reference.predictive_gradients, 1)])
    np.testing.assert_allclose(matrix[-6:, :12], analytic, atol=1e-12)
    eps = 1e-6
    numeric = np.empty_like(analytic)
    for j in range(12):
        change = np.zeros(12)
        change[j] = eps
        values = []
        for sign in (1, -1):
            states = m._rollout(np.zeros(3), reference.commands+sign*change.reshape(6, 2))
            values.append(m.predictive_geometry.constraints(states[1:], np.arange(1, 7)*.2, FOOTPRINT, .1)[0].ravel())
        numeric[:, j] = (values[0]-values[1])/(2*eps)
    np.testing.assert_allclose(matrix[-6:, :12], numeric, atol=1e-7)


def test_dense_future_crossing_cannot_pass_safe_endpoints():
    cfg = HumanGeometryForecastConfig(velocity_uncertainty_prior_mps=.001)
    f = replace(prediction(center=(.05, -.2), velocity=(0., 2.), config=cfg), radius_prior_m=.01)
    m = mpc(geometry(f), safety_substeps_per_interval=12, integration_substeps_per_interval=12)
    args = inputs()
    args['configuration_space'] = replace(args['configuration_space'],
        footprint=RectangularRobotFootprint(.01, .11, .01, .01, 0.))
    evaluation = m._evaluate_trajectory(np.zeros(3), np.zeros((6, 2)), temporal_safety_derivative=0., **args)
    assert evaluation is not None and np.min(evaluation.predictive_margins) > 0
    assert not evaluation.predictive_dense_safe and not m._trajectory_is_safe(evaluation)


def test_revalidation_uses_current_forecast_and_original_field_age_without_mutating_shared_mpc():
    shared = mpc()
    far = geometry(prediction(center=(4., 1.), velocity=(0., 0.)))
    command = np.tile([.2, 0.], (6, 1))
    assert shared.revalidate_commands(np.zeros(3), command, predictive_geometry=far,
        predictive_source_age_seconds=.3, **inputs())
    assert not shared.revalidate_commands(np.zeros(3), command, predictive_geometry=geometry(),
        predictive_source_age_seconds=.3, **inputs())
    for age in (-.01, .3501, np.nan):
        assert not shared.revalidate_commands(np.zeros(3), command, predictive_geometry=far,
            predictive_source_age_seconds=age, **inputs())
    assert shared.predictive_geometry is None and shared.predictive_source_age_seconds == 0.


def test_missing_current_fit_fails_closed_and_source_transfer_is_rejected():
    g = replace(geometry(), forecasts=())
    result = mpc(g).plan(np.zeros(3), (.2, 0.), **inputs())
    assert not result.success and result.first_command == (0., 0.)
    assert result.predictive_geometry['ready'] is False
    with pytest.raises(ValueError):
        replace(runtime_field(), predictive_human_geometry=geometry())


def test_runtime_requires_planning_epoch_and_keeps_forecast_request_local():
    f = runtime_field()
    track = f.tracking_result.tracked_obstacles[0]
    cfg = HumanGeometryForecastConfig(minimum_points=3)
    g = replace(geometry(prediction(center=track.centroid_world_xy, velocity=(0., 0.), key=track.track_id, config=cfg)),
        sensor_frame_id=f.sensor_frame_id, anchor_position_world_xy=tuple(f.anchor_position_world_xy), anchor_yaw=f.anchor_yaw)
    f = replace(f, predictive_human_geometry=g,
        pipeline=replace(f.pipeline, configuration_space=replace(f.pipeline.configuration_space, footprint=FOOTPRINT)))
    shared = mpc()
    runtime = SafeSAGEControlRuntime(mpc=shared, backup_filter=SafeSAGEUnicycleCBFFilter(shared.paper_filter))
    with pytest.raises(ValueError, match='actual planning-state epoch'):
        runtime.plan(np.zeros(3), (.2, 0.), f)
    result = runtime.plan(np.zeros(3), (.2, 0.), f, prediction_state_timestamp_seconds=1.1)
    assert result.predictive_geometry['planning_source_age_seconds'] == pytest.approx(.1)
    assert shared.predictive_geometry is None


@pytest.mark.parametrize('floor', [None, .3])
def test_nonlinear_rescue_uses_the_same_exact_prediction_values_and_analytic_jacobian(monkeypatch, floor):
    from types import SimpleNamespace
    import robot.safety.safe_sage_unicycle_mpc as module
    cfg = HumanGeometryForecastConfig(velocity_uncertainty_prior_mps=.5 if floor else .25,
        velocity_uncertainty_floor_prior_mps=floor)
    m = mpc(geometry(prediction(config=cfg)))
    reference = m._evaluate_trajectory(np.zeros(3), np.tile([.2, .05], (6, 1)),
        temporal_safety_derivative=0., **inputs())
    captured = {}

    def inspect(objective, x, **kwargs):
        values = kwargs['constraints']['fun']
        jacobian = kwargs['constraints']['jac']
        captured['values'] = values(x)
        actual = jacobian(x)
        eps = 1e-6
        numeric = np.stack([(values(x+eps*np.eye(12)[k])-values(x-eps*np.eye(12)[k]))/(2*eps)
            for k in range(12)], axis=1)
        np.testing.assert_allclose(actual[-6:], numeric[-6:], atol=1e-7)
        np.testing.assert_allclose(captured['values'][-6:], reference.predictive_margins.ravel(), atol=1e-12)
        assert actual.shape == (36, 12)
        return SimpleNamespace(x=x, nit=1, message='independent constraint inspection')

    monkeypatch.setattr(module, 'minimize', inspect)
    _, _, status = m._solve_exact_nonlinear_rescue(reference, initial_state=np.zeros(3), nominal_sequence=reference.commands,
        temporal_safety_derivative=0., **inputs())
    assert len(captured['values']) == 36
    assert 'minimum_predictive_geometry_margin_m=' in status


def test_explicit_two_second_horizon_retains_original_age_limit_and_exact_acceptance():
    cfg = HumanGeometryForecastConfig(horizon_seconds=2., velocity_uncertainty_prior_mps=.5)
    g = geometry(prediction(center=(5., 2.), velocity=(0., 0.), config=cfg))
    m = SafeSAGEUnicycleMPC(make_filter(), SafeSAGEUnicycleMPCConfig(horizon_steps=10), predictive_geometry=g)
    result = m.plan(np.zeros(3), (.2, 0.), **inputs())
    assert result.success and result.command_sequence.shape == (10, 2)
    assert m.revalidate_commands(np.zeros(3), result.command_sequence, predictive_geometry=g,
        predictive_source_age_seconds=.3, **inputs())
    assert not m.revalidate_commands(np.zeros(3), result.command_sequence, predictive_geometry=g,
        predictive_source_age_seconds=.36, **inputs())

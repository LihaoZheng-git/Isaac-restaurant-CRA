"""Piecewise analytic forecast derivatives; finite differences are TEST only."""

from dataclasses import fields, replace

import numpy as np
import pytest
from test_safe_sage_cbf_filter import make_fields, make_filter
from test_safe_sage_unicycle_mpc import make_mpc

from robot.safety.safe_sage_cbf_filter import SafeSAGECBFSafetyFilter
from robot.safety.safe_sage_temporal_derivative import SafeSAGETemporalDerivativeResult
from robot.safety.safe_sage_unicycle_mpc import SafeSAGEUnicycleMPCConfig
from robot.safety.temporal_forecast_gradient import scaled_temporal_forecast_gradient


def varying_fields(*, guide_scale=1.):
    cspace, h, guides = make_fields(safety_value=2.)
    cfg = cspace.occupancy_grids[0].config
    x = cfg.minimum_x + (np.arange(cfg.column_count) + .5) * cfg.resolution
    y = cfg.minimum_y + (np.arange(cfg.row_count) + .5) * cfg.resolution
    xx, yy = np.meshgrid(x, y)
    yaw = h.yaw_angles_radians[:, None, None]
    values = 2 + .3 * xx + .2 * yy + .04 * xx * yy + .1 * np.sin(yaw)
    gradients = np.broadcast_to(np.stack((.3 + .04 * yy, .2 + .04 * xx), axis=-1), values.shape + (2,)).copy()
    h = replace(h, values=values, spatial_gradients_robot_xy=gradients,
        yaw_derivatives=(np.roll(values, -1, axis=0) - np.roll(values, 1, axis=0)) / (2 * 2 * np.pi / len(guides)))
    guides = tuple(replace(g, vectors_robot_xy=guide_scale * np.stack((
        .08 + .01 * xx + .01 * np.cos(float(yaw[i, 0, 0])),
        .05 + .015 * yy + .005 * np.sin(float(yaw[i, 0, 0]))), axis=-1)) for i, g in enumerate(guides))
    tau = .07 + .03 * xx - .02 * yy + .04 * np.sin(yaw)
    temporal = SafeSAGETemporalDerivativeResult(values=tau, raw_values=tau,
        valid_mask=np.ones_like(tau, bool), yaw_angles_radians=h.yaw_angles_radians,
        grid_config=cfg, time_delta_seconds=.2, egomotion_translation_grid_xy=(0., 0.), smoothing_factor=1.)
    return dict(configuration_space=cspace, h_full=h, guidance_fields=guides,
        temporal_safety_derivative=temporal)


@pytest.mark.parametrize('temporal', ['field', 0., .2])
@pytest.mark.parametrize('guide_scale', [1., 100.])
def test_full_scaled_temporal_gradient_matches_public_values(temporal, guide_scale):
    kwargs = varying_fields(guide_scale=guide_scale)
    if temporal != 'field':
        kwargs['temporal_safety_derivative'] = temporal
    paper = SafeSAGECBFSafetyFilter(replace(make_filter().config,
        temporal_scaling_epsilon=.2, temporal_scaling_kappa=1.7))
    # Away from spatial/yaw/clamp knots; include the periodic seam's two sides.
    states = np.array([[.17, .28, -.51], [.43, -.16, 1.07], [-.27, .37, 3.02], [.12, .23, -3.04]])
    analytic = scaled_temporal_forecast_gradient(paper.config, states, **kwargs)
    numeric = np.empty_like(analytic)
    eps = 1e-6
    for i, state in enumerate(states):
        for axis in range(3):
            delta = np.zeros(3)
            delta[axis] = eps
            def value(s):
                return paper.evaluate_constraint(query_position_grid_xy=s[:2],
                    query_yaw_radians=float(s[2]), **kwargs).scaled_temporal_safety_derivative
            numeric[i, axis] = (value(state + delta) - value(state - delta)) / (2 * eps)
    np.testing.assert_allclose(analytic, numeric, atol=1e-8, rtol=1e-6)
    if temporal == 0.:
        np.testing.assert_array_equal(analytic, np.zeros_like(analytic))


def test_zero_guidance_and_zero_spatial_gradient_stay_finite():
    cspace, h, guides = make_fields(safety_value=2., gradient=(0., 0., 0.), guidance=(0., 0.))
    result = scaled_temporal_forecast_gradient(make_filter().config, np.array([[.17, .28, .21]]),
        configuration_space=cspace, h_full=h, guidance_fields=guides, temporal_safety_derivative=.2)
    np.testing.assert_array_equal(result, np.zeros((1, 3)))


def test_mpc_jacobian_matches_its_actual_resampled_forecast():
    kwargs = varying_fields()
    controller = make_mpc(horizon_steps=4, use_consistent_temporal_jacobian=True)
    state = np.array([.17, .28, .21])
    commands = np.tile((.15, .1), (4, 1))
    value = controller._evaluate_trajectory(state, commands, check_dense_geometry=False, **kwargs)
    assert value is not None
    sensitivities = controller._safety_state_sensitivities(value.chassis_states,
        controller._rollout_state_sensitivities(value.chassis_states, commands))
    gradients = value.samples.exact_gradients
    analytic = np.array([gradients[i + 1] @ sensitivities[i + 1]
        - controller.config.dtcbf_decay * gradients[i] @ sensitivities[i] for i in range(4)])
    numeric = np.empty_like(analytic)
    eps = 1e-6
    for j in range(8):
        delta = np.zeros_like(commands)
        delta.flat[j] = eps
        plus = controller._evaluate_trajectory(state, commands + delta, check_dense_geometry=False, **kwargs)
        minus = controller._evaluate_trajectory(state, commands - delta, check_dense_geometry=False, **kwargs)
        numeric[:, j] = (plus.margins - minus.margins) / (2 * eps)
    np.testing.assert_allclose(analytic, numeric, atol=1e-8, rtol=1e-6)
    legacy = make_mpc(horizon_steps=4)._evaluate_trajectory(state, commands, check_dense_geometry=False, **kwargs)
    np.testing.assert_array_equal(value.safety_values, legacy.safety_values)
    np.testing.assert_array_equal(value.margins, legacy.margins)


@pytest.mark.parametrize('value', [1, None])
def test_consistent_gradient_requires_boolean_opt_in(value):
    with pytest.raises(TypeError, match='use_consistent_temporal_jacobian'):
        SafeSAGEUnicycleMPCConfig(use_consistent_temporal_jacobian=value)


def test_zero_temporal_field_preserves_entire_static_result():
    kwargs = varying_fields()
    kwargs['temporal_safety_derivative'] = 0.
    outputs = [make_mpc(horizon_steps=4, use_consistent_temporal_jacobian=enabled).plan(
        np.array([.17, .28, .21]), (.15, .1), **kwargs) for enabled in (False, True)]
    for key in fields(outputs[0]):
        a, b = getattr(outputs[0], key.name), getattr(outputs[1], key.name)
        if isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        else:
            assert a == b

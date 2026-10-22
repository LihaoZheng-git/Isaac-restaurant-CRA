from dataclasses import replace

import numpy as np
import pytest
from test_laplace_guidance_tracking_mpc import config
from test_safe_sage_cbf_filter import make_fields, make_filter

from robot.safety.laplace_guidance_tracking_mpc import (
    LaplaceGuidanceTrackingConfig,
    LaplaceGuidanceTrackingMPC,
)


def planner(minimum=1.):
    return LaplaceGuidanceTrackingMPC(make_filter(), config(), guidance_config=
        LaplaceGuidanceTrackingConfig(weight=30., reference_mode="planar_forecast_velocity",
            tangent_velocity_fraction=1., minimum_reference_speed_fraction=minimum))


def fields(h=.6):
    cs, values, guide = make_fields(safety_value=h, gradient=(-3., 0., 0.), guidance=(0., 1.))
    return dict(configuration_space=cs, h_full=values, guidance_fields=guide)


@pytest.mark.parametrize("value", [0., -.1, .249, 1.01, True, np.nan, np.inf, "0.25"])
def test_invalid_reference_speed_scope_rejected(value):
    with pytest.raises(ValueError):
        planner(value)


def test_reduced_speed_finds_checked_progress_reference_when_full_speed_cannot():
    initial = np.zeros(3)
    kwargs = fields()
    nominal = np.tile([.2, 0.], (config().horizon_steps, 1))
    before = nominal.copy()
    legacy = planner().plan(initial, nominal, nominal_goal_grid_xy=(2., 0.), **kwargs)
    m = planner(.25)
    result = m.plan(initial, nominal, nominal_goal_grid_xy=(2., 0.), **kwargs)
    assert not legacy.guidance_tracking.active
    assert result.success and result.guidance_tracking.active
    detail = result.guidance_tracking
    assert detail.selected_reference_speed_fraction < 1
    assert detail.reference_displacement_m >= .01
    reference = m._evaluate_trajectory(initial, np.array(detail.reference_commands),
        temporal_safety_derivative=0., **kwargs)
    assert reference is not None and m._trajectory_is_safe(reference)
    executable = m._evaluate_trajectory(initial, result.command_sequence,
        temporal_safety_derivative=0., **kwargs)
    assert executable is not None and m._trajectory_is_safe(executable)
    np.testing.assert_array_equal(nominal, before)


def test_default_matches_explicit_single_speed_and_larger_pool_never_mutates_nominal():
    initial = np.zeros(3)
    nominal = np.tile([.2, 0.], (config().horizon_steps, 1))
    kwargs = fields(h=2.)
    a, b = planner(), planner(1.)
    results = [m.plan(initial, nominal, nominal_goal_grid_xy=(2., 0.), **kwargs) for m in (a, b)]
    np.testing.assert_array_equal(results[0].command_sequence, results[1].command_sequence)
    assert a.guidance_config.reference_speed_fractions() == (1.,)


@pytest.mark.parametrize("minimum", [.25, .5, .75, 1.])
def test_stationary_nominal_and_unsafe_state_cannot_be_laundered_by_speed_pool(minimum):
    m = planner(minimum)
    stationary = m.plan(np.zeros(3), (0., 0.), nominal_goal_grid_xy=(2., 0.), **fields())
    assert stationary.first_command == (0., 0.) and not stationary.guidance_tracking.active
    failed = m.plan(np.zeros(3), (.2, 0.), nominal_goal_grid_xy=(2., 0.), **fields(h=-.2))
    assert not failed.success and not failed.guidance_tracking.active


def test_zero_weight_retains_primary_even_with_reduced_candidate_option():
    m = planner(.25)
    m.guidance_config = replace(m.guidance_config, weight=0., tangent_velocity_fraction=0.)
    legacy = planner(1.)
    legacy.guidance_config = replace(legacy.guidance_config, weight=0., tangent_velocity_fraction=0.)
    a = m.plan(np.zeros(3), (.2, 0.), **fields())
    b = legacy.plan(np.zeros(3), (.2, 0.), **fields())
    np.testing.assert_array_equal(a.command_sequence, b.command_sequence)

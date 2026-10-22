from dataclasses import replace

import numpy as np
import pytest
from test_laplace_guidance_tracking_mpc import config
from test_safe_sage_cbf_filter import make_fields, make_filter
from test_safe_sage_runtime import make_runtime, make_observation, make_provider

from robot.safety.laplace_guidance_tracking_mpc import LaplaceGuidanceTrackingConfig, LaplaceGuidanceTrackingMPC
from robot.safety.safe_sage_runtime import SafeSAGEControlRuntime
from robot.safety.safe_sage_unicycle_cbf_filter import SafeSAGEUnicycleCBFFilter


def planner(fraction):
    return LaplaceGuidanceTrackingMPC(make_filter(), config(),
        guidance_config=LaplaceGuidanceTrackingConfig(weight=30,
            reference_mode="planar_forecast_velocity", tangent_velocity_fraction=fraction,
            guidance_fade_distance_m=.15))


def field(side=1., gradient=(-.1, 0., 0.), h=.6):
    cs, values, guide = make_fields(safety_value=h, gradient=gradient, guidance=(0., side))
    return dict(configuration_space=cs, h_full=values, guidance_fields=guide)


@pytest.mark.parametrize("value", [-.1, 1.1, True, np.nan, np.inf])
def test_invalid_tangent_fraction_rejected(value):
    with pytest.raises(ValueError):
        planner(value)


def test_tangent_requires_explicit_objective_and_forecast_reference():
    with pytest.raises(ValueError):
        LaplaceGuidanceTrackingConfig(tangent_velocity_fraction=.5, reference_mode="planar_forecast_velocity")
    with pytest.raises(ValueError):
        LaplaceGuidanceTrackingConfig(weight=10, tangent_velocity_fraction=.5)


@pytest.mark.parametrize("side", [-1., 1.])
@pytest.mark.parametrize("steps,goal_distance", [(8, 2.), (6, .2)])
def test_tangent_can_steer_before_halfspace_binds_but_retains_exact_horizon(side, steps, goal_distance):
    initial = np.zeros(3)
    kwargs = field(side)
    c = replace(config(), horizon_steps=steps)
    nominal = np.tile([.2, 0.], (steps, 1))
    original = nominal.copy()
    baseline = planner(0)
    baseline.config = c
    legacy = baseline.plan(initial, nominal, nominal_goal_grid_xy=(goal_distance, 0.), **kwargs)
    mpc = planner(1)
    mpc.config = c
    result = mpc.plan(initial, nominal, nominal_goal_grid_xy=(goal_distance, 0.), **kwargs)
    assert result.success and result.guidance_tracking.active
    assert abs(legacy.angular_velocity) < 1e-6
    assert side * result.angular_velocity > .02
    assert result.minimum_dtcbf_margin >= -mpc.config.constraint_tolerance
    evaluated = mpc._evaluate_trajectory(initial, result.command_sequence,
        temporal_safety_derivative=0., **kwargs)
    assert evaluated is not None and mpc._trajectory_is_safe(evaluated)
    assert result.guidance_tracking.tangent_velocity_fraction == 1
    np.testing.assert_array_equal(nominal, original)


@pytest.mark.parametrize("gradient,h", [((0., 0., 0.), .6), ((0., .1, 0.), .6), ((-.1, 0., 0.), 2.)])
def test_degenerate_normal_parallel_guidance_or_inactive_field_preserves_preference(gradient, h):
    kwargs = field(gradient=gradient, h=h)
    initial = np.zeros(3)
    goal = np.array([2., 0.])
    preferences = [m._reference_preference(initial, goal, .2, m._sample(initial, kwargs), kwargs)
                   for m in (planner(0), planner(1))]
    np.testing.assert_array_equal(preferences[0], preferences[1])


def runtime_field():
    perception = make_runtime()
    perception.initialize(make_observation(1, 1.), make_provider())
    return perception.field


@pytest.mark.parametrize("ids", [(999,), (0, 0), (True,), (-1,)])
def test_reference_identity_requires_unique_current_positive_tracks(ids):
    with pytest.raises(ValueError, match="unique current positive"):
        replace(runtime_field(), motion_passing_reference_track_ids=ids)


def test_reference_identity_epoch_must_match_field():
    f = runtime_field()
    with pytest.raises(ValueError, match="field epoch"):
        replace(f, tracking_result=replace(f.tracking_result, timestamp_seconds=.9),
            motion_passing_reference_track_ids=(0,))


def test_runtime_gates_tangent_per_request_without_mutating_shared_planner_or_sources():
    f = runtime_field()
    mpc = planner(1)
    runtime = SafeSAGEControlRuntime(mpc=mpc, backup_filter=SafeSAGEUnicycleCBFFilter(mpc.paper_filter))
    initial = np.zeros(3)
    without = runtime.plan(initial, (.2, 0), f, nominal_goal_grid_xy=(1., 0.))
    selected = replace(f, motion_passing_reference_track_ids=(0,))
    with_selection = runtime.plan(initial, (.2, 0), selected, nominal_goal_grid_xy=(1., 0.))
    after = runtime.plan(initial, (.2, 0), f, nominal_goal_grid_xy=(1., 0.))
    assert without.guidance_tracking.tangent_velocity_fraction == 0
    assert with_selection.guidance_tracking.tangent_velocity_fraction == 1
    assert with_selection.guidance_tracking.tangent_reference_track_ids == (0,)
    np.testing.assert_array_equal(without.command_sequence, after.command_sequence)
    assert runtime.mpc is mpc and mpc.guidance_config.tangent_velocity_fraction == 1
    assert f.motion_passing_reference_track_ids == ()
    assert f.sensor_timestamp_seconds == selected.sensor_timestamp_seconds == 1.

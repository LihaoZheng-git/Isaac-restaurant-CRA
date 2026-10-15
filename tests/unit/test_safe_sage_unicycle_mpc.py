import math
import time

import numpy as np
import pytest
from test_safe_sage_cbf_filter import make_fields, make_filter

from robot.safety.safe_sage_cbf_filter import SafeSAGECBFConfig
from robot.safety.safe_sage_unicycle_mpc import (
    SafeSAGEUnicycleMPC,
    SafeSAGEUnicycleMPCConfig,
)
from robot.safety.safe_sage_temporal_derivative import (
    SafeSAGETemporalDerivativeResult,
)


def make_mpc(**overrides: float) -> SafeSAGEUnicycleMPC:
    """Return one deterministic predictive unicycle filter."""

    values: dict[str, float | int] = {
        "horizon_steps": 10,
        "time_step": 0.20,
        "gamma": 1.0,
        "lookahead_distance": 0.10,
        "minimum_linear_velocity": 0.0,
        "maximum_linear_velocity": 0.30,
        "minimum_angular_velocity": -1.0,
        "maximum_angular_velocity": 1.0,
        "linear_velocity_weight": 1.0,
        "angular_velocity_weight": 0.02,
        "command_smoothness_weight": 0.001,
        "maximum_sqp_iterations": 8,
        "maximum_solver_iterations": 4000,
    }
    values.update(overrides)
    paper_filter = make_filter()
    return SafeSAGEUnicycleMPC(
        paper_filter,
        SafeSAGEUnicycleMPCConfig(**values),
    )


def plan_synthetic(
    *,
    guidance: tuple[float, float],
    gradient: tuple[float, float, float],
    safety_value: float,
    nominal: tuple[float, float] = (0.25, 0.0),
):
    """Run the MPC against one aligned synthetic Safe-SAGE field."""

    configuration_space, h_full, guidance_fields = make_fields(
        guidance=guidance,
        gradient=gradient,
        safety_value=safety_value,
    )
    return make_mpc().plan(
        np.array([0.0, 0.0, 0.0]),
        np.asarray(nominal, dtype=np.float64),
        configuration_space=configuration_space,
        h_full=h_full,
        guidance_fields=guidance_fields,
    )


def test_config_uses_safe_sage_exponential_decay() -> None:
    config = SafeSAGEUnicycleMPCConfig(
        time_step=0.2,
        gamma=1.5,
    )

    assert config.dtcbf_decay == pytest.approx(math.exp(-0.3))


def test_mpc_exact_gradient_uses_derivative_of_yaw_interpolant() -> None:
    configuration_space, h_full, guidance_fields = make_fields(
        gradient=(0.3, -0.2, 0.0),
        per_yaw_safety_values=(1.0, 2.0, 3.0, 4.0),
    )
    mpc = make_mpc()
    safety_states = np.repeat(
        np.array([[0.0, 0.0, -0.75 * np.pi]], dtype=np.float64),
        mpc.config.horizon_steps + 1,
        axis=0,
    )

    samples = mpc._sample_reference(
        safety_states,
        configuration_space=configuration_space,
        h_full=h_full,
        guidance_fields=guidance_fields,
        temporal_safety_derivative=0.0,
    )

    np.testing.assert_allclose(samples.exact_gradients[:, 0], 0.3)
    np.testing.assert_allclose(samples.exact_gradients[:, 1], -0.2)
    np.testing.assert_allclose(samples.exact_gradients[:, 2], 2.0 / np.pi)


def test_mpc_samples_spatial_temporal_field_along_prediction() -> None:
    configuration_space, h_full, guidance_fields = make_fields(
        safety_value=1.0,
        gradient=(-1.0, 0.0, 0.0),
        guidance=(-1.0, 0.0),
    )
    grid_config = configuration_space.occupancy_grids[0].config
    column_values = np.linspace(-0.2, 0.2, grid_config.column_count)
    temporal_values = np.broadcast_to(
        column_values,
        h_full.values.shape,
    ).copy()
    temporal_field = SafeSAGETemporalDerivativeResult(
        values=temporal_values,
        raw_values=temporal_values,
        valid_mask=np.ones_like(temporal_values, dtype=bool),
        yaw_angles_radians=h_full.yaw_angles_radians,
        grid_config=grid_config,
        time_delta_seconds=0.2,
        egomotion_translation_grid_xy=(0.0, 0.0),
        smoothing_factor=1.0,
    )
    mpc = make_mpc()
    safety_states = np.repeat(
        np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
        mpc.config.horizon_steps + 1,
        axis=0,
    )
    safety_states[:, 0] = np.linspace(
        0.0,
        1.0,
        mpc.config.horizon_steps + 1,
    )

    samples = mpc._sample_reference(
        safety_states,
        configuration_space=configuration_space,
        h_full=h_full,
        guidance_fields=guidance_fields,
        temporal_safety_derivative=temporal_field,
    )

    assert samples.scaled_temporal_derivatives[0] == pytest.approx(0.0)
    assert samples.scaled_temporal_derivatives[-1] == pytest.approx(0.1)


def test_open_space_passes_nominal_command_through() -> None:
    result = plan_synthetic(
        guidance=(-1.0, 0.0),
        gradient=(0.0, 0.0, 0.0),
        safety_value=2.0,
    )

    assert result.success
    assert not result.fallback_active
    assert not result.intervention_active
    assert result.first_command == pytest.approx((0.25, 0.0))
    assert result.minimum_dtcbf_margin >= -1e-6


def test_safe_warm_start_is_a_seed_not_the_returned_solution() -> None:
    configuration_space, h_full, guidance_fields = make_fields(
        guidance=(-1.0, 0.0),
        gradient=(0.0, 0.0, 0.0),
        safety_value=2.0,
    )
    mpc = make_mpc()

    result = mpc.plan(
        np.zeros(3, dtype=np.float64),
        np.array([0.25, 0.0], dtype=np.float64),
        initial_command_guess=np.zeros((10, 2), dtype=np.float64),
        configuration_space=configuration_space,
        h_full=h_full,
        guidance_fields=guidance_fields,
    )

    assert result.success, result.status
    assert result.first_command == pytest.approx((0.25, 0.0))
    assert not result.intervention_active


def test_predictive_filter_turns_before_current_filter_is_active() -> None:
    result = plan_synthetic(
        guidance=(-1.0, -1.0),
        gradient=(-1.0, -1.0, 0.0),
        safety_value=0.75,
    )

    assert result.success, result.status
    assert result.intervention_active
    assert result.angular_velocity < -1e-3
    assert result.linear_velocity > 0.0
    assert result.minimum_dtcbf_margin >= -1e-6
    assert result.predicted_chassis_states[-1, 1] < 0.0


def test_passing_side_reverses_predicted_steering() -> None:
    pass_right = plan_synthetic(
        guidance=(-1.0, -1.0),
        gradient=(-1.0, -1.0, 0.0),
        safety_value=0.75,
    )
    pass_left = plan_synthetic(
        guidance=(-1.0, 1.0),
        gradient=(-1.0, 1.0, 0.0),
        safety_value=0.75,
    )

    assert pass_right.success, pass_right.status
    assert pass_left.success, pass_left.status
    assert pass_right.angular_velocity < 0.0
    assert pass_left.angular_velocity > 0.0
    assert pass_right.angular_velocity == pytest.approx(
        -pass_left.angular_velocity,
        abs=2e-3,
    )


def test_exact_dtcbf_restoration_recovers_surrogate_mismatch() -> None:
    result = plan_synthetic(
        guidance=(-1.0, -1.0),
        gradient=(-1.0, 0.0, 0.0),
        safety_value=0.75,
    )

    assert result.success, result.status
    assert result.restoration_used
    assert result.status == "exact_dtcbf_feasibility_restored"
    assert result.minimum_dtcbf_margin >= -1e-6
    assert not result.fallback_active


def test_elastic_cbf_subproblem_remains_solvable_inside_tight_trust_region() -> None:
    configuration_space, h_full, guidance_fields = make_fields(
        guidance=(-5.0, -5.0),
        gradient=(-5.0, -5.0, 0.0),
        safety_value=0.6,
    )
    mpc = make_mpc(
        linear_velocity_trust_region=0.001,
        angular_velocity_trust_region=0.001,
    )
    nominal_sequence = mpc._nominal_sequence(np.array([0.25, 0.0]))
    reference = mpc._evaluate_trajectory(
        np.zeros(3, dtype=np.float64),
        nominal_sequence,
        configuration_space=configuration_space,
        h_full=h_full,
        guidance_fields=guidance_fields,
        temporal_safety_derivative=0.0,
    )

    assert reference is not None
    assert np.min(reference.margins) < 0.0
    solution = mpc._solve_sparse_subproblem(
        reference,
        nominal_sequence=nominal_sequence,
        configuration_space=configuration_space,
        derivative_mode="gradient",
    )

    assert solution.success, solution.status
    assert solution.maximum_cbf_slack > 0.0


@pytest.mark.parametrize("integration_substeps", [1, 12])
def test_analytical_rollout_sensitivities_match_finite_difference(
    integration_substeps,
) -> None:
    mpc = make_mpc(
        horizon_steps=4, integration_substeps_per_interval=integration_substeps
    )
    initial_state = np.array([0.1, -0.2, 0.35], dtype=np.float64)
    commands = np.array(
        [
            [0.20, -0.30],
            [0.18, 0.10],
            [0.22, 0.25],
            [0.16, -0.05],
        ],
        dtype=np.float64,
    )
    states = mpc._rollout(initial_state, commands)
    analytical = mpc._rollout_state_sensitivities(states, commands)
    epsilon = 1e-6

    for variable_index in range(commands.size):
        positive = commands.copy().reshape(-1)
        negative = commands.copy().reshape(-1)
        positive[variable_index] += epsilon
        negative[variable_index] -= epsilon
        positive_states = mpc._rollout(
            initial_state,
            positive.reshape(commands.shape),
        )
        negative_states = mpc._rollout(
            initial_state,
            negative.reshape(commands.shape),
        )
        finite_difference = (positive_states - negative_states) / (2.0 * epsilon)
        np.testing.assert_allclose(
            analytical[:, :, variable_index],
            finite_difference,
            rtol=1e-5,
            atol=1e-7,
        )


def test_substep_prediction_matches_executed_curved_unicycle_motion():
    from robot.control.unicycle_model import integrate_unicycle_euler

    mpc = make_mpc(horizon_steps=1, time_step=0.2, integration_substeps_per_interval=12)
    initial = np.array([0.1, -0.2, 0.35])
    command = np.array([0.2, 0.8])
    actual = initial.copy()
    for _ in range(12):
        actual = integrate_unicycle_euler(actual, command, time_step=0.2 / 12)
    predicted = mpc._rollout(initial, command[None, :])[-1]
    np.testing.assert_allclose(predicted, actual, atol=1e-12)
    coarse = integrate_unicycle_euler(initial, command, time_step=0.2)
    assert np.linalg.norm(predicted[:2] - coarse[:2]) > 0.002


def test_inter_sample_check_follows_the_integrated_turn_instead_of_its_chord(
    monkeypatch,
):
    from robot.control.unicycle_model import integrate_unicycle_euler, lookahead_state

    mpc = make_mpc(
        horizon_steps=1,
        time_step=0.2,
        safety_substeps_per_interval=12,
        integration_substeps_per_interval=12,
    )
    initial = np.zeros(3)
    commands = np.array([[0.2, 0.8]])
    actual = initial.copy()
    for _ in range(6):
        actual = integrate_unicycle_euler(actual, commands[0], time_step=0.2 / 12)
    query = lookahead_state(actual, lookahead_distance=mpc.config.lookahead_distance)
    monkeypatch.setattr(
        mpc.paper_filter,
        "unsafe_configuration_mask",
        lambda states, **kwargs: (
            np.linalg.norm(states[:, :2] - query[:2], axis=1) < 1e-8
        ),
    )
    configuration_space, h_full, _ = make_fields()
    mask = mpc._inter_sample_unsafe_mask(
        mpc._rollout(initial, commands),
        np.zeros(2, dtype=bool),
        commands=commands,
        configuration_space=configuration_space,
        h_full=h_full,
    )
    assert mask.tolist() == [False, True]


@pytest.mark.parametrize(
    "options",
    [
        {"integration_substeps_per_interval": 0},
        {"integration_substeps_per_interval": 2, "safety_substeps_per_interval": 3},
    ],
)
def test_invalid_integration_and_safety_substep_settings_are_rejected(options):
    with pytest.raises(ValueError):
        make_mpc(**options)


def test_thirty_step_sparse_mpc_solves_within_one_second() -> None:
    configuration_space, h_full, guidance_fields = make_fields(
        guidance=(-1.0, -1.0),
        gradient=(-1.0, -1.0, 0.0),
        safety_value=0.75,
    )
    mpc = make_mpc(
        horizon_steps=30,
        maximum_sqp_iterations=12,
        maximum_solver_iterations=4000,
    )

    start_time = time.perf_counter()
    result = mpc.plan(
        np.zeros(3, dtype=np.float64),
        np.array([0.25, 0.0], dtype=np.float64),
        configuration_space=configuration_space,
        h_full=h_full,
        guidance_fields=guidance_fields,
    )
    elapsed = time.perf_counter() - start_time

    assert result.success, result.status
    assert result.angular_velocity < 0.0
    assert result.minimum_dtcbf_margin >= -1e-6
    assert elapsed < 1.0


@pytest.mark.parametrize("rescue_budget", [300, 20])
def test_thirty_step_exact_nonlinear_rescue_is_safe_and_fast(rescue_budget) -> None:
    configuration_space, h_full, guidance_fields = make_fields(
        guidance=(-1.0, -1.0),
        gradient=(-1.0, -1.0, 0.0),
        safety_value=0.75,
    )
    mpc = make_mpc(horizon_steps=30, maximum_nonlinear_rescue_iterations=rescue_budget)
    initial_state = np.zeros(3, dtype=np.float64)
    nominal_sequence = mpc._nominal_sequence(np.array([0.25, 0.0]))
    reference = mpc._evaluate_trajectory(
        initial_state,
        nominal_sequence,
        configuration_space=configuration_space,
        h_full=h_full,
        guidance_fields=guidance_fields,
        temporal_safety_derivative=0.0,
    )

    assert reference is not None
    start_time = time.perf_counter()
    candidate, iteration_count, status = mpc._solve_exact_nonlinear_rescue(
        reference,
        initial_state=initial_state,
        nominal_sequence=nominal_sequence,
        configuration_space=configuration_space,
        h_full=h_full,
        guidance_fields=guidance_fields,
        temporal_safety_derivative=0.0,
    )
    elapsed = time.perf_counter() - start_time

    assert candidate is not None, status
    assert iteration_count > 0
    assert np.min(candidate.margins) >= -1e-6
    assert candidate.commands[0, 1] < 0.0
    assert candidate.chassis_states[-1, 1] < 0.0
    assert elapsed < 1.0


@pytest.mark.parametrize("rescue_budget", [300, 20])
def test_nonlinear_rescue_recovers_osqp_iteration_limit(rescue_budget) -> None:
    configuration_space, h_full, guidance_fields = make_fields(
        guidance=(-1.0, -1.0),
        gradient=(-1.0, -1.0, 0.0),
        safety_value=0.75,
    )
    mpc = make_mpc(
        horizon_steps=30,
        maximum_solver_iterations=1,
        maximum_nonlinear_rescue_iterations=rescue_budget,
    )

    result = mpc.plan(
        np.zeros(3, dtype=np.float64),
        np.array([0.25, 0.0], dtype=np.float64),
        configuration_space=configuration_space,
        h_full=h_full,
        guidance_fields=guidance_fields,
    )

    assert result.success, result.status
    assert result.restoration_used
    assert result.status == "exact_nonlinear_feasibility_restored"
    assert result.minimum_dtcbf_margin >= -1e-6
    assert result.angular_velocity < 0.0
    assert result.predicted_chassis_states[-1, 1] < 0.0


def test_exact_feasibility_recovery_survives_solver_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration_space, h_full, guidance_fields = make_fields(
        guidance=(-1.0, -1.0),
        gradient=(-1.0, -1.0, 0.0),
        safety_value=0.75,
    )
    mpc = make_mpc(
        horizon_steps=30,
        maximum_solver_iterations=1,
    )
    monkeypatch.setattr(
        mpc,
        "_solve_exact_nonlinear_rescue",
        lambda *args, **kwargs: (None, 1, "forced numerical failure"),
    )

    result = mpc.plan(
        np.zeros(3, dtype=np.float64),
        np.array([0.25, 0.0], dtype=np.float64),
        configuration_space=configuration_space,
        h_full=h_full,
        guidance_fields=guidance_fields,
    )

    assert result.success, result.status
    assert not result.fallback_active
    assert result.status in {
        "nominal_guidance_seed_feasibility_recovered",
        "guidance_steered_feasibility_recovered",
        "guidance_rotation_feasibility_recovered",
        "translation_scaled_feasibility_recovered",
        "stationary_feasibility_recovered",
    }
    assert result.minimum_dtcbf_margin >= -1e-6


def test_last_resort_recovery_preserves_guidance_passing_side() -> None:
    configuration_space, h_full, guidance_fields = make_fields(
        guidance=(-1.0, -1.0),
        gradient=(-1.0, 0.0, 0.0),
        safety_value=0.75,
    )
    mpc = make_mpc(horizon_steps=30)
    initial_state = np.zeros(3, dtype=np.float64)
    nominal_sequence = mpc._nominal_sequence(np.array([0.25, 0.0]))
    reference = mpc._evaluate_trajectory(
        initial_state,
        nominal_sequence,
        configuration_space=configuration_space,
        h_full=h_full,
        guidance_fields=guidance_fields,
        temporal_safety_derivative=0.0,
    )

    assert reference is not None
    assert not mpc._trajectory_is_safe(reference)
    candidate, status = mpc._recover_feasible_command_horizon(
        reference,
        initial_state=initial_state,
        nominal_sequence=nominal_sequence,
        configuration_space=configuration_space,
        h_full=h_full,
        guidance_fields=guidance_fields,
        temporal_safety_derivative=0.0,
        preferred_lateral_direction_grid_y=-1.0,
    )

    assert candidate is not None
    assert status == "nominal_guidance_seed_feasibility_recovered"
    assert candidate.commands[0, 1] < 0.0
    assert np.max(candidate.commands[:, 0]) > 0.0
    assert candidate.chassis_states[-1, 1] < 0.0
    assert mpc._trajectory_is_safe(candidate)


def test_recovery_does_not_confuse_local_turn_with_passing_side() -> None:
    configuration_space, h_full, guidance_fields = make_fields(
        guidance=(-1.0, 1.0),
        gradient=(-1.0, 0.0, 0.0),
        safety_value=0.75,
    )
    mpc = make_mpc(horizon_steps=30)
    initial_state = np.array([0.0, 0.0, -0.5], dtype=np.float64)
    nominal_sequence = mpc._nominal_sequence(np.array([0.25, 0.0]))
    reference = mpc._evaluate_trajectory(
        initial_state,
        nominal_sequence,
        configuration_space=configuration_space,
        h_full=h_full,
        guidance_fields=guidance_fields,
        temporal_safety_derivative=0.0,
    )

    assert reference is not None
    assert mpc._initial_guidance_angular_coefficient(reference) > 0.0
    candidate, status = mpc._recover_feasible_command_horizon(
        reference,
        initial_state=initial_state,
        nominal_sequence=nominal_sequence,
        configuration_space=configuration_space,
        h_full=h_full,
        guidance_fields=guidance_fields,
        temporal_safety_derivative=0.0,
        preferred_lateral_direction_grid_y=-1.0,
    )

    assert candidate is not None
    assert status == "nominal_guidance_seed_feasibility_recovered"
    assert candidate.chassis_states[-1, 1] < initial_state[1]
    assert mpc._trajectory_is_safe(candidate)


def test_zero_preferred_lateral_direction_is_rejected() -> None:
    configuration_space, h_full, guidance_fields = make_fields()

    with pytest.raises(ValueError, match="preferred_lateral_direction_grid_y"):
        make_mpc().plan(
            np.zeros(3, dtype=np.float64),
            np.array([0.25, 0.0]),
            preferred_lateral_direction_grid_y=0.0,
            configuration_space=configuration_space,
            h_full=h_full,
            guidance_fields=guidance_fields,
        )


def test_shifted_safe_horizon_warm_starts_next_control_cycle() -> None:
    configuration_space, h_full, guidance_fields = make_fields(
        guidance=(-1.0, -1.0),
        gradient=(-1.0, -1.0, 0.0),
        safety_value=0.75,
    )
    mpc = make_mpc(horizon_steps=30)
    nominal_command = np.array([0.25, 0.0], dtype=np.float64)
    first = mpc.plan(
        np.zeros(3, dtype=np.float64),
        nominal_command,
        configuration_space=configuration_space,
        h_full=h_full,
        guidance_fields=guidance_fields,
    )

    assert first.success, first.status
    shifted_commands = np.vstack(
        (first.command_sequence[1:], np.zeros((1, 2), dtype=np.float64))
    )
    second = mpc.plan(
        first.predicted_chassis_states[1],
        nominal_command,
        initial_command_guess=shifted_commands,
        configuration_space=configuration_space,
        h_full=h_full,
        guidance_fields=guidance_fields,
    )

    assert second.success, second.status
    assert not second.fallback_active
    assert second.minimum_dtcbf_margin >= -1e-6
    assert second.predicted_chassis_states[-1, 1] < 0.0


def test_initial_command_guess_shape_is_validated() -> None:
    configuration_space, h_full, guidance_fields = make_fields()
    mpc = make_mpc()

    with pytest.raises(ValueError, match="initial_command_guess"):
        mpc.plan(
            np.zeros(3, dtype=np.float64),
            np.array([0.25, 0.0], dtype=np.float64),
            initial_command_guess=np.zeros((3, 2), dtype=np.float64),
            configuration_space=configuration_space,
            h_full=h_full,
            guidance_fields=guidance_fields,
        )


def test_prediction_never_accepts_an_unsafe_configuration_cell() -> None:
    configuration_space, h_full, guidance_fields = make_fields(
        guidance=(0.0, 0.0),
        gradient=(0.0, 0.0, 0.0),
        safety_value=2.0,
    )
    mpc = make_mpc(horizon_steps=30)

    result = mpc.plan(
        np.zeros(3, dtype=np.float64),
        np.array([0.30, 0.0], dtype=np.float64),
        configuration_space=configuration_space,
        h_full=h_full,
        guidance_fields=guidance_fields,
    )

    assert result.success, result.status
    assert not result.fallback_active
    assert result.status in {
        "guidance_steered_feasibility_recovered",
        "guidance_rotation_feasibility_recovered",
        "translation_scaled_feasibility_recovered",
        "stationary_feasibility_recovered",
    }
    evaluations = [
        mpc.paper_filter.evaluate_constraint(
            query_position_grid_xy=state[:2],
            query_yaw_radians=float(state[2]),
            configuration_space=configuration_space,
            h_full=h_full,
            guidance_fields=guidance_fields,
        )
        for state in result.predicted_safety_states
    ]
    assert all(evaluation.unsafe_reason is None for evaluation in evaluations)


def test_inter_sample_check_rejects_skipping_through_unsafe_cell() -> None:
    configuration_space, h_full, guidance_fields = make_fields(
        guidance=(0.0, 0.0),
        gradient=(0.0, 0.0, 0.0),
        safety_value=2.0,
        unsafe_yaw_indices=(0, 1, 2, 3),
    )
    endpoint_only = make_mpc(
        horizon_steps=1,
        time_step=1.0,
        maximum_linear_velocity=2.0,
    )
    dense_check = make_mpc(
        horizon_steps=1,
        time_step=1.0,
        maximum_linear_velocity=2.0,
        safety_substeps_per_interval=2,
    )
    initial_state = np.array([-1.1, 0.0, 0.0], dtype=np.float64)
    nominal = np.array([2.0, 0.0], dtype=np.float64)

    endpoint_result = endpoint_only.plan(
        initial_state,
        nominal,
        configuration_space=configuration_space,
        h_full=h_full,
        guidance_fields=guidance_fields,
    )
    dense_result = dense_check.plan(
        initial_state,
        nominal,
        configuration_space=configuration_space,
        h_full=h_full,
        guidance_fields=guidance_fields,
    )

    assert endpoint_result.success, endpoint_result.status
    assert endpoint_result.linear_velocity == pytest.approx(2.0)
    assert dense_result.success, dense_result.status
    assert dense_result.linear_velocity < 2.0
    assert "feasibility_recovered" in dense_result.status


def test_result_shapes_and_arrays_are_read_only() -> None:
    result = plan_synthetic(
        guidance=(-1.0, 0.0),
        gradient=(0.0, 0.0, 0.0),
        safety_value=2.0,
    )

    assert result.command_sequence.shape == (10, 2)
    assert result.predicted_chassis_states.shape == (11, 3)
    assert result.predicted_safety_states.shape == (11, 3)
    assert result.predicted_safety_values.shape == (11,)
    assert result.dtcbf_margins.shape == (10,)
    assert not result.command_sequence.flags.writeable


def test_unsafe_current_configuration_fails_closed() -> None:
    configuration_space, h_full, guidance_fields = make_fields(
        guidance=(-1.0, -1.0),
        safety_value=0.0,
        unsafe_yaw_indices=(2,),
    )
    result = make_mpc().plan(
        np.array([0.0, 0.0, 0.0]),
        np.array([0.25, 0.0]),
        configuration_space=configuration_space,
        h_full=h_full,
        guidance_fields=guidance_fields,
    )

    assert not result.success
    assert result.fallback_active
    assert result.first_command == (0.0, 0.0)
    assert result.status == "unsafe_configuration_boundary"


@pytest.mark.parametrize(
    "override",
    (
        {"horizon_steps": 0},
        {"safety_substeps_per_interval": 0},
        {"time_step": 0.0},
        {"gamma": 0.0},
        {"lookahead_distance": 0.0},
        {"minimum_linear_velocity": 0.1},
        {"angular_velocity_weight": 0.0},
        {"command_smoothness_weight": -1.0},
        {"maximum_restoration_iterations": 0},
        {"maximum_nonlinear_rescue_iterations": 0},
        {"linear_velocity_trust_region": 0.0},
        {"cbf_slack_weight": 0.0},
        {"line_search_contraction": 1.0},
        {"enable_progress_seeds": 1},
        {"enable_recovery_cost_pruning": 1},
        {"enable_recovery_feasibility_pruning": 1},
        {"progress_seed_linear_fraction": 0.0},
        {"progress_seed_linear_fraction": 1.0},
        {"configuration_position_padding_m": -0.01},
        {"configuration_yaw_padding_radians": -0.01},
        {"configuration_padding_initial_fraction": -0.1},
        {"configuration_padding_initial_fraction": 1.1},
    ),
)
def test_invalid_configuration_is_rejected(
    override: dict[str, float | int],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        SafeSAGEUnicycleMPCConfig(**override)


def test_mpc_requires_paper_space_filter() -> None:
    with pytest.raises(TypeError):
        SafeSAGEUnicycleMPC(
            SafeSAGECBFConfig(),  # type: ignore[arg-type]
        )


def test_progress_seeds_preserve_open_space_nominal_and_unsafe_stop():
    mpc = make_mpc(enable_progress_seeds=True)
    kwargs = dict(
        zip(
            ("configuration_space", "h_full", "guidance_fields"),
            make_fields(safety_value=2.0),
        )
    )
    result = mpc.plan(np.zeros(3), np.array([0.25, 0.0]), **kwargs)
    assert result.success
    assert result.first_command == pytest.approx((0.25, 0.0))
    kwargs = dict(
        zip(
            ("configuration_space", "h_full", "guidance_fields"),
            make_fields(safety_value=2.0, unsafe_yaw_indices=(2,)),
        )
    )
    result = mpc.plan(np.zeros(3), np.array([0.25, 0.0]), **kwargs)
    assert not result.success
    assert result.first_command == (0.0, 0.0)


def test_progress_pool_can_restore_an_infeasible_nominal_horizon():
    configuration_space, h_full, guidance_fields = make_fields(
        guidance=(-1.0, -1.0), gradient=(-1.0, -1.0, 0.0), safety_value=.75,
    )
    result = make_mpc(horizon_steps=30, enable_progress_seeds=True).plan(
        np.zeros(3), np.array([.25, 0]), configuration_space=configuration_space,
        h_full=h_full, guidance_fields=guidance_fields,
    )
    assert result.success and result.restoration_used
    assert result.status == "exact_safe_seed_feasibility_recovered"
    assert result.minimum_dtcbf_margin >= -1e-6
    assert result.predicted_chassis_states[-1, 1] < 0


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_recovery_cost_pruning_matches_exhaustive_exact_winner(direction, monkeypatch):
    fields = make_fields(guidance=(0, direction), gradient=(0, direction, 0), safety_value=.15)
    kwargs = dict(configuration_space=fields[0], h_full=fields[1], guidance_fields=fields[2], temporal_safety_derivative=0.0)
    state = np.array([0., 0., -.8 * direction])
    outputs = []
    for prune in (False, True):
        mpc = make_mpc(horizon_steps=30, enable_recovery_cost_pruning=prune,
                       enable_recovery_feasibility_pruning=prune)
        nominal = mpc._nominal_sequence(np.array([.25, 0.]))
        reference = mpc._evaluate_trajectory(state, nominal, **kwargs)
        assert reference is not None
        original = mpc._evaluate_trajectory
        calls = []

        def counted(*args, **kw):
            calls.append(1)
            return original(*args, **kw)

        with monkeypatch.context() as patch:
            patch.setattr(mpc, "_evaluate_trajectory", counted)
            candidate, status = mpc._recover_feasible_command_horizon(
                reference, initial_state=state, nominal_sequence=nominal, **kwargs)
        assert candidate is not None and mpc._trajectory_is_safe(candidate)
        outputs.append((candidate, status, len(calls)))
    np.testing.assert_array_equal(outputs[0][0].commands, outputs[1][0].commands)
    np.testing.assert_allclose(outputs[0][0].chassis_states, outputs[1][0].chassis_states, atol=1e-12)
    np.testing.assert_allclose(outputs[0][0].margins, outputs[1][0].margins, atol=1e-12)
    assert outputs[0][1] == outputs[1][1]
    assert outputs[1][2] < outputs[0][2]


@pytest.mark.parametrize("temporal_derivative", [0.0, -1.0])
def test_recovery_feasibility_pruning_only_skips_already_failed_dtcbf(temporal_derivative, monkeypatch):
    fields = make_fields(safety_value=.15, gradient=(0, 1, 0), guidance=(0, 1))
    mpc = make_mpc(horizon_steps=3, safety_substeps_per_interval=12,
                   integration_substeps_per_interval=12)
    kwargs = dict(configuration_space=fields[0], h_full=fields[1], guidance_fields=fields[2],
                  temporal_safety_derivative=temporal_derivative)
    commands = mpc._nominal_sequence(np.array([.1, 0.]))
    original = mpc._inter_sample_unsafe_mask
    calls = []

    def counted(*args, **kw):
        calls.append(1)
        return original(*args, **kw)

    with monkeypatch.context() as patch:
        patch.setattr(mpc, "_inter_sample_unsafe_mask", counted)
        legacy = mpc._evaluate_trajectory(np.zeros(3), commands, **kwargs)
        assert legacy is not None and len(calls) == 1
        calls.clear()
        pruned = mpc._evaluate_trajectory(np.zeros(3), commands, require_feasible=True, **kwargs)
    if temporal_derivative < 0:
        assert not mpc._trajectory_is_safe(legacy)
        assert pruned is None and calls == []
    else:
        assert pruned is not None and mpc._trajectory_is_safe(pruned)
        assert len(calls) == 1
        np.testing.assert_array_equal(legacy.commands, pruned.commands)
        np.testing.assert_array_equal(legacy.margins, pruned.margins)
        np.testing.assert_array_equal(legacy.chassis_states, pruned.chassis_states)


def test_unhinted_recovery_can_return_upward_with_lower_nominal_cost():
    fields = make_fields(guidance=(0, 1), gradient=(0, 1, 0), safety_value=.15)
    kwargs = dict(configuration_space=fields[0], h_full=fields[1], guidance_fields=fields[2], temporal_safety_derivative=0.0)
    mpc = make_mpc(horizon_steps=30, enable_recovery_cost_pruning=True)
    state = np.array([0., 0., -.8])
    nominal = mpc._nominal_sequence(np.array([.25, 0.]))
    reference = mpc._evaluate_trajectory(state, nominal, **kwargs)
    free, _ = mpc._recover_feasible_command_horizon(reference, initial_state=state, nominal_sequence=nominal, **kwargs)
    forced, _ = mpc._recover_feasible_command_horizon(reference, initial_state=state, nominal_sequence=nominal,
                                                   preferred_lateral_direction_grid_y=-1., **kwargs)
    assert free is not None and forced is not None
    assert free.chassis_states[-1, 1] > 0 > forced.chassis_states[-1, 1]
    assert mpc._objective_value(free.commands, nominal) < mpc._objective_value(forced.commands, nominal)
    assert mpc._trajectory_is_safe(free) and mpc._trajectory_is_safe(forced)


def test_geometry_padding_grows_within_first_execution_interval(monkeypatch):
    configuration_space, h_full, _ = make_fields()
    mpc = make_mpc(
        horizon_steps=2, safety_substeps_per_interval=2,
        configuration_position_padding_m=.1,
        configuration_yaw_padding_radians=.2,
        configuration_padding_initial_fraction=.1,
    )
    captured = []

    def inspect(states, **kwargs):
        captured.append(states.copy())
        return np.zeros(len(states), dtype=bool)

    monkeypatch.setattr(mpc.paper_filter, "unsafe_configuration_mask", inspect)
    commands = np.zeros((2, 2))
    mpc._inter_sample_unsafe_mask(
        mpc._rollout(np.zeros(3), commands), np.zeros(3, dtype=bool),
        commands=commands, configuration_space=configuration_space, h_full=h_full,
    )
    queries = captured[0].reshape(2, 2, -1, 3)
    np.testing.assert_allclose(np.max(np.abs(queries[..., 0] - .1), axis=2), [[.055, .1], [.1, .1]])
    np.testing.assert_allclose(np.max(np.abs(queries[..., 2]), axis=2), [[.11, .2], [.2, .2]])


@pytest.mark.parametrize("padding", [
    {"configuration_position_padding_m": 0.02},
    {"configuration_yaw_padding_radians": 0.08},
])
def test_future_geometry_padding_rejects_nearby_unsafe_configuration(padding):
    configuration_space, h_full, _ = make_fields(unsafe_yaw_indices=(2,))
    if "configuration_position_padding_m" in padding:
        state = np.array([.41, 0, 0.0])  # lookahead x=.51 is outside unsafe center cell
    else:
        state = np.array([0, 0, np.pi / 4 + .02])  # nearest yaw is safe slice 3
    commands = np.zeros((1, 2))
    plain = make_mpc(horizon_steps=1)
    padded = make_mpc(horizon_steps=1, **padding)
    kwargs = dict(commands=commands, configuration_space=configuration_space, h_full=h_full)
    endpoints = np.zeros(2, dtype=bool)
    assert not plain._inter_sample_unsafe_mask(plain._rollout(state, commands), endpoints, **kwargs).any()
    assert padded._inter_sample_unsafe_mask(padded._rollout(state, commands), endpoints, **kwargs).tolist() == [False, True]

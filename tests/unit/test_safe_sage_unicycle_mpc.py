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


def test_analytical_rollout_sensitivities_match_finite_difference() -> None:
    mpc = make_mpc(horizon_steps=4)
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


def test_thirty_step_exact_nonlinear_rescue_is_safe_and_fast() -> None:
    configuration_space, h_full, guidance_fields = make_fields(
        guidance=(-1.0, -1.0),
        gradient=(-1.0, -1.0, 0.0),
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


def test_nonlinear_rescue_recovers_osqp_iteration_limit() -> None:
    configuration_space, h_full, guidance_fields = make_fields(
        guidance=(-1.0, -1.0),
        gradient=(-1.0, -1.0, 0.0),
        safety_value=0.75,
    )
    mpc = make_mpc(
        horizon_steps=30,
        maximum_solver_iterations=1,
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
        "translation_scaled_feasibility_recovered",
        "stationary_feasibility_recovered",
    }
    assert result.minimum_dtcbf_margin >= -1e-6


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

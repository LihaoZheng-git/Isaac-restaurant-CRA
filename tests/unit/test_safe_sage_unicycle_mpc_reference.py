import numpy as np
import pytest
from test_safe_sage_cbf_filter import make_fields, make_filter

from robot.control.planar_path_reference import (
    PlanarPathReferenceConfig,
    build_planar_path_reference,
)
from robot.safety.safe_sage_unicycle_mpc import (
    SafeSAGEUnicycleMPC,
    SafeSAGEUnicycleMPCConfig,
)


def make_reference_mpc(
    *,
    horizon_steps: int = 10,
    position_tracking_weight: float = 12.0,
    yaw_tracking_weight: float = 0.8,
) -> SafeSAGEUnicycleMPC:
    return SafeSAGEUnicycleMPC(
        make_filter(),
        SafeSAGEUnicycleMPCConfig(
            horizon_steps=horizon_steps,
            time_step=0.20,
            lookahead_distance=0.10,
            minimum_linear_velocity=0.0,
            maximum_linear_velocity=0.30,
            minimum_angular_velocity=-1.0,
            maximum_angular_velocity=1.0,
            linear_velocity_weight=1.0,
            angular_velocity_weight=0.02,
            command_smoothness_weight=0.001,
            position_tracking_weight=position_tracking_weight,
            yaw_tracking_weight=yaw_tracking_weight,
            maximum_sqp_iterations=8,
            maximum_solver_iterations=4000,
        ),
    )


def right_side_planar_reference(
    mpc: SafeSAGEUnicycleMPC,
):
    horizon = mpc.config.horizon_steps
    progress = np.arange(horizon + 1, dtype=np.float64)
    planar_states = np.column_stack(
        (
            0.04 * progress,
            -0.018 * progress,
            np.zeros(horizon + 1),
        )
    )
    return build_planar_path_reference(
        planar_states,
        initial_chassis_state=np.zeros(3),
        config=PlanarPathReferenceConfig(
            time_step=mpc.config.time_step,
            minimum_linear_velocity=mpc.config.minimum_linear_velocity,
            maximum_linear_velocity=mpc.config.maximum_linear_velocity,
            minimum_angular_velocity=mpc.config.minimum_angular_velocity,
            maximum_angular_velocity=mpc.config.maximum_angular_velocity,
        ),
    )


def test_state_reference_turns_toward_right_side_path_in_open_space() -> None:
    configuration_space, h_full, guidance_fields = make_fields(
        guidance=(0.0, 0.0),
        gradient=(0.0, 0.0, 0.0),
        safety_value=2.0,
    )
    mpc = make_reference_mpc()
    reference = right_side_planar_reference(mpc)

    result = mpc.plan(
        np.zeros(3),
        np.array([0.20, 0.0]),
        initial_command_guess=reference.seed_commands,
        target_chassis_states_grid=reference.target_states,
        configuration_space=configuration_space,
        h_full=h_full,
        guidance_fields=guidance_fields,
    )

    assert result.success, result.status
    assert result.angular_velocity < -1e-3
    assert result.predicted_chassis_states[-1, 1] < -0.02
    assert result.minimum_dtcbf_margin >= -1e-6


def test_state_reference_is_optimized_without_explicit_warm_start() -> None:
    configuration_space, h_full, guidance_fields = make_fields(
        guidance=(0.0, 0.0),
        gradient=(0.0, 0.0, 0.0),
        safety_value=2.0,
    )
    mpc = make_reference_mpc()
    reference = right_side_planar_reference(mpc)

    result = mpc.plan(
        np.zeros(3),
        np.array([0.20, 0.0]),
        target_chassis_states_grid=reference.target_states,
        configuration_space=configuration_space,
        h_full=h_full,
        guidance_fields=guidance_fields,
    )

    assert result.success, result.status
    assert result.angular_velocity < -1e-3
    assert result.predicted_chassis_states[-1, 1] < 0.0


def test_zero_state_weights_preserve_command_only_behavior() -> None:
    configuration_space, h_full, guidance_fields = make_fields(
        guidance=(0.0, 0.0),
        gradient=(0.0, 0.0, 0.0),
        safety_value=2.0,
    )
    mpc = make_reference_mpc(
        position_tracking_weight=0.0,
        yaw_tracking_weight=0.0,
    )
    reference = right_side_planar_reference(mpc)

    result = mpc.plan(
        np.zeros(3),
        np.array([0.20, 0.0]),
        target_chassis_states_grid=reference.target_states,
        configuration_space=configuration_space,
        h_full=h_full,
        guidance_fields=guidance_fields,
    )

    assert result.success, result.status
    assert result.first_command == pytest.approx((0.20, 0.0))


def test_state_tracking_objective_gradient_matches_finite_difference() -> None:
    mpc = make_reference_mpc(horizon_steps=4)
    initial_state = np.array([0.1, -0.2, 0.15])
    commands = np.array(
        [
            [0.18, -0.25],
            [0.20, -0.10],
            [0.16, 0.15],
            [0.14, 0.20],
        ]
    )
    nominal = np.repeat(np.array([[0.20, 0.0]]), 4, axis=0)
    target_commands = np.array(
        [
            [0.20, -0.35],
            [0.20, -0.20],
            [0.18, 0.05],
            [0.16, 0.15],
        ]
    )
    target_states = mpc._rollout(initial_state, target_commands)
    chassis_states = mpc._rollout(initial_state, commands)
    analytical = mpc._objective_gradient(
        commands,
        nominal,
        chassis_states=chassis_states,
        target_chassis_states=target_states,
    ).reshape(-1)

    epsilon = 1e-6
    numerical = np.empty(commands.size)
    for variable_index in range(commands.size):
        positive = commands.copy().reshape(-1)
        negative = commands.copy().reshape(-1)
        positive[variable_index] += epsilon
        negative[variable_index] -= epsilon
        positive_commands = positive.reshape(commands.shape)
        negative_commands = negative.reshape(commands.shape)
        positive_value = mpc._objective_value(
            positive_commands,
            nominal,
            chassis_states=mpc._rollout(initial_state, positive_commands),
            target_chassis_states=target_states,
        )
        negative_value = mpc._objective_value(
            negative_commands,
            nominal,
            chassis_states=mpc._rollout(initial_state, negative_commands),
            target_chassis_states=target_states,
        )
        numerical[variable_index] = (
            positive_value - negative_value
        ) / (2.0 * epsilon)

    np.testing.assert_allclose(analytical, numerical, rtol=1e-5, atol=1e-7)


def test_yaw_tracking_error_uses_shortest_wrapped_difference() -> None:
    error = SafeSAGEUnicycleMPC._state_tracking_error(
        np.array([0.0, 0.0, np.pi - 0.01]),
        np.array([0.0, 0.0, -np.pi + 0.01]),
    )
    assert error == pytest.approx([0.0, 0.0, -0.02])


@pytest.mark.parametrize(
    "target",
    (
        np.zeros((10, 3)),
        np.full((11, 3), np.nan),
    ),
)
def test_invalid_state_reference_is_rejected(target: np.ndarray) -> None:
    configuration_space, h_full, guidance_fields = make_fields()
    mpc = make_reference_mpc()

    with pytest.raises(ValueError, match="target_chassis_states_grid"):
        mpc.plan(
            np.zeros(3),
            np.array([0.20, 0.0]),
            target_chassis_states_grid=target,
            configuration_space=configuration_space,
            h_full=h_full,
            guidance_fields=guidance_fields,
        )


@pytest.mark.parametrize(
    "override",
    (
        {"position_tracking_weight": -1.0},
        {"yaw_tracking_weight": -1.0},
    ),
)
def test_negative_state_tracking_weight_is_rejected(override) -> None:
    with pytest.raises(ValueError):
        SafeSAGEUnicycleMPCConfig(**override)

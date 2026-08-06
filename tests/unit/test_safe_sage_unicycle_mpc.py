import math

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
        "maximum_solver_iterations": 200,
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

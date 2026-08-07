import math

import numpy as np
import pytest
from robot.safety.safe_sage_cbf_filter import SafeSAGECBFConfig
from robot.safety.safe_sage_planar_mpc import (
    SafeSAGEPlanarMPC,
    SafeSAGEPlanarMPCConfig,
)

from test_safe_sage_cbf_filter import make_fields, make_filter


def make_mpc(**overrides: float) -> SafeSAGEPlanarMPC:
    values: dict[str, float | int] = {
        "horizon_steps": 6,
        "time_step": 0.20,
        "gamma": 1.0,
        "minimum_planar_velocity": -0.30,
        "maximum_planar_velocity": 0.30,
        "minimum_angular_velocity": -1.0,
        "maximum_angular_velocity": 1.0,
        "planar_velocity_weight": 1.0,
        "angular_velocity_weight": 0.10,
        "command_smoothness_weight": 0.001,
        "lateral_seed_speed": 0.15,
        "maximum_solver_iterations": 200,
    }
    values.update(overrides)
    return SafeSAGEPlanarMPC(
        make_filter(),
        SafeSAGEPlanarMPCConfig(**values),
    )


def plan_synthetic(
    *,
    guidance: tuple[float, float],
    gradient: tuple[float, float, float],
    safety_value: float,
):
    configuration_space, h_full, guidance_fields = make_fields(
        guidance=guidance,
        gradient=gradient,
        safety_value=safety_value,
    )
    return make_mpc().plan(
        np.zeros(3, dtype=np.float64),
        np.array([0.25, 0.0, 0.0], dtype=np.float64),
        configuration_space=configuration_space,
        h_full=h_full,
        guidance_fields=guidance_fields,
    )


def test_config_uses_safe_sage_exponential_decay() -> None:
    config = SafeSAGEPlanarMPCConfig(time_step=0.2, gamma=1.5)
    assert config.dtcbf_decay == pytest.approx(math.exp(-0.3))


def test_open_space_passes_nominal_command_through() -> None:
    result = plan_synthetic(
        guidance=(-1.0, 0.0),
        gradient=(0.0, 0.0, 0.0),
        safety_value=2.0,
    )

    assert result.success, result.status
    assert not result.fallback_active
    assert not result.intervention_active
    assert result.first_command == pytest.approx((0.25, 0.0, 0.0))
    assert result.minimum_dtcbf_margin >= -1e-6


def test_planar_mpc_can_translate_laterally_without_turning() -> None:
    result = plan_synthetic(
        guidance=(-1.0, -1.0),
        gradient=(-1.0, -1.0, 0.0),
        safety_value=0.20,
    )

    assert result.success, result.status
    assert result.intervention_active
    assert result.first_command[1] < -1e-3
    assert result.first_command[2] == pytest.approx(0.0, abs=1e-6)
    assert result.predicted_states[-1, 1] < 0.0
    assert result.minimum_dtcbf_margin >= -1e-6


def test_passing_side_reverses_lateral_translation() -> None:
    pass_right = plan_synthetic(
        guidance=(-1.0, -1.0),
        gradient=(-1.0, -1.0, 0.0),
        safety_value=0.20,
    )
    pass_left = plan_synthetic(
        guidance=(-1.0, 1.0),
        gradient=(-1.0, 1.0, 0.0),
        safety_value=0.20,
    )

    assert pass_right.success, pass_right.status
    assert pass_left.success, pass_left.status
    assert pass_right.first_command[1] < 0.0
    assert pass_left.first_command[1] > 0.0
    assert pass_right.first_command[1] == pytest.approx(
        -pass_left.first_command[1],
        abs=2e-3,
    )


def test_full_nominal_sequence_and_warm_start_are_accepted() -> None:
    configuration_space, h_full, guidance_fields = make_fields(
        guidance=(-1.0, 0.0),
        gradient=(0.0, 0.0, 0.0),
        safety_value=2.0,
    )
    mpc = make_mpc()
    nominal = np.repeat(
        np.array([[0.20, -0.02, 0.10]], dtype=np.float64),
        mpc.config.horizon_steps,
        axis=0,
    )
    result = mpc.plan(
        np.zeros(3),
        nominal,
        initial_command_guess=np.zeros_like(nominal),
        configuration_space=configuration_space,
        h_full=h_full,
        guidance_fields=guidance_fields,
    )

    assert result.success, result.status
    assert result.first_command == pytest.approx((0.20, -0.02, 0.10))


def test_result_shapes_and_arrays_are_read_only() -> None:
    result = plan_synthetic(
        guidance=(-1.0, 0.0),
        gradient=(0.0, 0.0, 0.0),
        safety_value=2.0,
    )

    assert result.command_sequence.shape == (6, 3)
    assert result.predicted_states.shape == (7, 3)
    assert result.predicted_safety_values.shape == (7,)
    assert result.dtcbf_margins.shape == (6,)
    assert not result.command_sequence.flags.writeable
    assert not result.predicted_states.flags.writeable


def test_unsafe_current_configuration_fails_closed() -> None:
    configuration_space, h_full, guidance_fields = make_fields(
        guidance=(-1.0, -1.0),
        safety_value=0.0,
        unsafe_yaw_indices=(2,),
    )
    result = make_mpc().plan(
        np.zeros(3),
        np.array([0.25, 0.0, 0.0]),
        configuration_space=configuration_space,
        h_full=h_full,
        guidance_fields=guidance_fields,
    )

    assert not result.success
    assert result.fallback_active
    assert result.first_command == (0.0, 0.0, 0.0)
    assert result.status == "unsafe_configuration_boundary"


def test_initial_command_guess_shape_is_validated() -> None:
    configuration_space, h_full, guidance_fields = make_fields()
    with pytest.raises(ValueError, match="initial_command_guess"):
        make_mpc().plan(
            np.zeros(3),
            np.array([0.25, 0.0, 0.0]),
            initial_command_guess=np.zeros((3, 3)),
            configuration_space=configuration_space,
            h_full=h_full,
            guidance_fields=guidance_fields,
        )


@pytest.mark.parametrize(
    "override",
    (
        {"horizon_steps": 0},
        {"time_step": 0.0},
        {"gamma": 0.0},
        {"minimum_planar_velocity": 0.1},
        {"angular_velocity_weight": 0.0},
        {"command_smoothness_weight": -1.0},
        {"lateral_seed_speed": -0.1},
        {"maximum_solver_iterations": 0},
    ),
)
def test_invalid_configuration_is_rejected(
    override: dict[str, float | int],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        SafeSAGEPlanarMPCConfig(**override)


def test_mpc_requires_paper_space_filter() -> None:
    with pytest.raises(TypeError):
        SafeSAGEPlanarMPC(SafeSAGECBFConfig())  # type: ignore[arg-type]

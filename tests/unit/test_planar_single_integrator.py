import math

import numpy as np
import pytest
from robot.control.planar_single_integrator import (
    integrate_planar_euler,
    planar_state_derivative,
)


def test_state_derivative_equals_planar_command() -> None:
    derivative = planar_state_derivative([0.20, -0.12, 0.35])
    assert derivative == pytest.approx([0.20, -0.12, 0.35])


def test_lateral_velocity_does_not_require_yaw_change() -> None:
    next_state = integrate_planar_euler(
        [1.0, 2.0, 0.0],
        [0.0, -0.25, 0.0],
        time_step=0.4,
    )
    assert next_state == pytest.approx([1.0, 1.9, 0.0])


def test_euler_step_integrates_all_three_command_components() -> None:
    next_state = integrate_planar_euler(
        [1.0, -2.0, 0.4],
        [0.20, -0.10, 0.30],
        time_step=0.5,
    )
    assert next_state == pytest.approx([1.10, -2.05, 0.55])


def test_euler_step_wraps_yaw() -> None:
    next_state = integrate_planar_euler(
        [0.0, 0.0, math.pi - 0.05],
        [0.0, 0.0, 0.20],
        time_step=0.5,
    )
    assert next_state[2] == pytest.approx(-math.pi + 0.05)


def test_derivative_returns_independent_array() -> None:
    command = np.array([0.20, -0.10, 0.30])
    derivative = planar_state_derivative(command)
    derivative[0] = 99.0
    assert command == pytest.approx([0.20, -0.10, 0.30])


def test_invalid_command_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        planar_state_derivative([0.20, 0.0])


def test_invalid_state_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        integrate_planar_euler(
            [0.0, 0.0],
            [0.20, 0.0, 0.0],
            time_step=0.1,
        )


def test_nonfinite_command_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        planar_state_derivative([0.20, float("nan"), 0.0])


@pytest.mark.parametrize("time_step", [0.0, -0.1])
def test_time_step_must_be_positive(time_step: float) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        integrate_planar_euler(
            [0.0, 0.0, 0.0],
            [0.20, 0.0, 0.0],
            time_step=time_step,
        )

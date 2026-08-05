import math

import numpy as np
import pytest
from robot.control.unicycle_model import (
    command_from_lookahead_velocity,
    integrate_unicycle_euler,
    lookahead_control_matrix,
    lookahead_state,
    lookahead_state_derivative,
    unicycle_state_derivative,
    wrap_to_pi,
)


def test_state_derivative_follows_heading() -> None:
    derivative = unicycle_state_derivative(
        [1.0, 2.0, math.pi / 2.0],
        [0.4, -0.3],
    )
    assert derivative == pytest.approx([0.0, 0.4, -0.3])


def test_euler_step_updates_and_wraps_yaw() -> None:
    next_state = integrate_unicycle_euler(
        [1.0, 2.0, math.pi - 0.05],
        [0.5, 0.2],
        time_step=0.5,
    )
    assert next_state[:2] == pytest.approx(
        [
            1.0 + 0.25 * math.cos(math.pi - 0.05),
            2.0 + 0.25 * math.sin(math.pi - 0.05),
        ]
    )
    assert next_state[2] == pytest.approx(-math.pi + 0.05)


def test_lookahead_state_uses_chassis_heading() -> None:
    result = lookahead_state(
        [1.0, 2.0, math.pi / 2.0],
        lookahead_distance=0.3,
    )
    assert result == pytest.approx([1.0, 2.3, math.pi / 2.0])


def test_control_matrix_at_zero_yaw() -> None:
    mapping = lookahead_control_matrix(
        0.0,
        lookahead_distance=0.1,
    )
    np.testing.assert_allclose(
        mapping,
        [
            [1.0, 0.0],
            [0.0, 0.1],
            [0.0, 1.0],
        ],
        atol=1e-12,
    )


def test_control_matrix_at_ninety_degrees() -> None:
    mapping = lookahead_control_matrix(
        math.pi / 2.0,
        lookahead_distance=0.2,
    )
    np.testing.assert_allclose(
        mapping,
        [
            [0.0, -0.2],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        atol=1e-12,
    )


def test_lookahead_derivative_uses_control_matrix() -> None:
    state = np.array([2.0, -1.0, 0.7])
    command = np.array([0.2, -0.4])
    expected = (
        lookahead_control_matrix(0.7, lookahead_distance=0.1)
        @ command
    )
    result = lookahead_state_derivative(
        state,
        command,
        lookahead_distance=0.1,
    )
    assert result == pytest.approx(expected)


def test_planar_velocity_round_trip() -> None:
    yaw = 0.7
    distance = 0.1
    command = np.array([0.2, -0.4])
    mapping = lookahead_control_matrix(
        yaw,
        lookahead_distance=distance,
    )
    planar_velocity = mapping[:2] @ command
    recovered = command_from_lookahead_velocity(
        planar_velocity,
        yaw_radians=yaw,
        lookahead_distance=distance,
    )
    assert recovered == pytest.approx(command)


@pytest.mark.parametrize(
    "angle, expected",
    [
        (0.0, 0.0),
        (math.pi, -math.pi),
        (3.0 * math.pi, -math.pi),
        (-3.0 * math.pi, -math.pi),
    ],
)
def test_wrap_to_pi(angle: float, expected: float) -> None:
    assert wrap_to_pi(angle) == pytest.approx(expected)


@pytest.mark.parametrize(
    "distance",
    [0.0, -0.1],
)
def test_lookahead_distance_must_be_positive(distance: float) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        lookahead_control_matrix(
            0.0,
            lookahead_distance=distance,
        )


def test_invalid_state_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        unicycle_state_derivative([0.0, 0.0], [0.1, 0.0])


def test_nonfinite_command_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        unicycle_state_derivative(
            [0.0, 0.0, 0.0],
            [float("nan"), 0.0],
        )


def test_time_step_must_be_positive() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        integrate_unicycle_euler(
            [0.0, 0.0, 0.0],
            [0.1, 0.0],
            time_step=0.0,
        )

"""Pure kinematic model for a differential-drive mobile robot.

The physical robot state is ``(x, y, yaw)`` and its executable control input
is ``(linear_velocity, angular_velocity)``.  This module also defines the
standard forward look-ahead point used to expose a relative-degree-one planar
output to safety filters.

The module deliberately has no Isaac Sim, navigation, or safety dependencies.
"""

import math
from collections.abc import Sequence
from numbers import Real

import numpy as np


def _finite_scalar(value: object, *, name: str) -> float:
    """Return one finite real number while rejecting booleans."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _finite_vector(
    value: Sequence[float] | np.ndarray,
    *,
    expected_size: int,
    name: str,
) -> np.ndarray:
    """Return one validated one-dimensional floating-point vector."""

    array = np.asarray(value, dtype=np.float64)
    if array.shape != (expected_size,):
        raise ValueError(f"{name} must have shape ({expected_size},).")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return np.ascontiguousarray(array, dtype=np.float64)


def wrap_to_pi(angle_radians: float) -> float:
    """Wrap one finite angle to the half-open interval ``[-pi, pi)``."""

    angle = _finite_scalar(angle_radians, name="angle_radians")
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def unicycle_state_derivative(
    state: Sequence[float] | np.ndarray,
    command: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Return ``(x_dot, y_dot, yaw_dot)`` for one unicycle state.

    ``state`` is ordered as ``(x, y, yaw)`` and ``command`` is ordered as
    ``(linear_velocity, angular_velocity)``.
    """

    state_array = _finite_vector(
        state,
        expected_size=3,
        name="state",
    )
    command_array = _finite_vector(
        command,
        expected_size=2,
        name="command",
    )
    yaw = float(state_array[2])
    linear_velocity = float(command_array[0])
    angular_velocity = float(command_array[1])
    return np.array(
        [
            linear_velocity * math.cos(yaw),
            linear_velocity * math.sin(yaw),
            angular_velocity,
        ],
        dtype=np.float64,
    )


def integrate_unicycle_euler(
    state: Sequence[float] | np.ndarray,
    command: Sequence[float] | np.ndarray,
    *,
    time_step: float,
) -> np.ndarray:
    """Advance the unicycle state by one forward-Euler integration step."""

    state_array = _finite_vector(
        state,
        expected_size=3,
        name="state",
    )
    time_step_value = _finite_scalar(time_step, name="time_step")
    if time_step_value <= 0.0:
        raise ValueError("time_step must be greater than zero.")
    next_state = state_array + time_step_value * unicycle_state_derivative(
        state_array,
        command,
    )
    next_state[2] = wrap_to_pi(float(next_state[2]))
    return np.ascontiguousarray(next_state, dtype=np.float64)


def lookahead_state(
    chassis_state: Sequence[float] | np.ndarray,
    *,
    lookahead_distance: float,
) -> np.ndarray:
    """Express ``(x_l, y_l, yaw)`` at a point ahead of the chassis."""

    state_array = _finite_vector(
        chassis_state,
        expected_size=3,
        name="chassis_state",
    )
    distance = _finite_scalar(
        lookahead_distance,
        name="lookahead_distance",
    )
    if distance <= 0.0:
        raise ValueError("lookahead_distance must be greater than zero.")
    yaw = float(state_array[2])
    return np.array(
        [
            state_array[0] + distance * math.cos(yaw),
            state_array[1] + distance * math.sin(yaw),
            yaw,
        ],
        dtype=np.float64,
    )


def lookahead_control_matrix(
    yaw_radians: float,
    *,
    lookahead_distance: float,
) -> np.ndarray:
    """Map ``(v, omega)`` to ``(x_l_dot, y_l_dot, yaw_dot)``."""

    yaw = _finite_scalar(yaw_radians, name="yaw_radians")
    distance = _finite_scalar(
        lookahead_distance,
        name="lookahead_distance",
    )
    if distance <= 0.0:
        raise ValueError("lookahead_distance must be greater than zero.")
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return np.array(
        [
            [cosine, -distance * sine],
            [sine, distance * cosine],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )


def lookahead_state_derivative(
    chassis_state: Sequence[float] | np.ndarray,
    command: Sequence[float] | np.ndarray,
    *,
    lookahead_distance: float,
) -> np.ndarray:
    """Return ``(x_l_dot, y_l_dot, yaw_dot)`` for one command."""

    state_array = _finite_vector(
        chassis_state,
        expected_size=3,
        name="chassis_state",
    )
    command_array = _finite_vector(
        command,
        expected_size=2,
        name="command",
    )
    mapping = lookahead_control_matrix(
        float(state_array[2]),
        lookahead_distance=lookahead_distance,
    )
    return np.ascontiguousarray(mapping @ command_array, dtype=np.float64)


def command_from_lookahead_velocity(
    planar_velocity_world_xy: Sequence[float] | np.ndarray,
    *,
    yaw_radians: float,
    lookahead_distance: float,
) -> np.ndarray:
    """Map a desired look-ahead planar velocity to ``(v, omega)``."""

    velocity = _finite_vector(
        planar_velocity_world_xy,
        expected_size=2,
        name="planar_velocity_world_xy",
    )
    yaw = _finite_scalar(yaw_radians, name="yaw_radians")
    distance = _finite_scalar(
        lookahead_distance,
        name="lookahead_distance",
    )
    if distance <= 0.0:
        raise ValueError("lookahead_distance must be greater than zero.")
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return np.array(
        [
            cosine * velocity[0] + sine * velocity[1],
            (-sine * velocity[0] + cosine * velocity[1]) / distance,
        ],
        dtype=np.float64,
    )

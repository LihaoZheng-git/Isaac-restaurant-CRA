"""Pure reduced-order planar model used by the Safe-SAGE oracle.

Safe-SAGE represents the robot configuration as ``(x, y, yaw)`` and the
directly commanded velocity as ``(v_x, v_y, omega)``.  Unlike a unicycle,
this model permits lateral translation without first changing yaw:

    x_dot = v_x
    y_dot = v_y
    yaw_dot = omega

The module deliberately has no Isaac Sim, navigation, or safety-field
dependencies.  It defines the idealized paper-space dynamics only; it must
never be used to send commands directly to Nova Carter's drive wheels.
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


def _wrap_to_pi(angle_radians: float) -> float:
    """Wrap one finite angle to the half-open interval ``[-pi, pi)``."""

    return (angle_radians + math.pi) % math.tau - math.pi


def planar_state_derivative(
    command: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Return ``(x_dot, y_dot, yaw_dot)`` for a planar command.

    ``command`` is ordered as ``(v_x, v_y, omega)`` in the fixed field
    frame.  The result is a copy, so mutating it cannot modify the caller's
    command array.
    """

    command_array = _finite_vector(
        command,
        expected_size=3,
        name="command",
    )
    return command_array.copy()


def integrate_planar_euler(
    state: Sequence[float] | np.ndarray,
    command: Sequence[float] | np.ndarray,
    *,
    time_step: float,
) -> np.ndarray:
    """Advance ``(x, y, yaw)`` by one forward-Euler integration step."""

    state_array = _finite_vector(
        state,
        expected_size=3,
        name="state",
    )
    time_step_value = _finite_scalar(time_step, name="time_step")
    if time_step_value <= 0.0:
        raise ValueError("time_step must be greater than zero.")

    next_state = state_array + time_step_value * planar_state_derivative(
        command
    )
    next_state[2] = _wrap_to_pi(float(next_state[2]))
    return np.ascontiguousarray(next_state, dtype=np.float64)

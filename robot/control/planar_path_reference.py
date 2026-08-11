"""Adapt a holonomic planar path into a unicycle tracking reference.

The Safe-SAGE planar oracle can move directly with ``(v_x, v_y, omega)``,
whereas Nova Carter executes ``(v, omega)`` and cannot translate sideways.
This module preserves the planar oracle's position path, derives path-tangent
heading targets, and constructs a bounded, dynamically feasible unicycle
command seed.

The seed is an optimization initializer only. It is not a safety certificate
and must not be sent directly to the wheels. The unicycle MPC and final
full-state CBF remain responsible for exact safety validation.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

import numpy as np
from robot.control.unicycle_model import (
    integrate_unicycle_euler,
    wrap_to_pi,
)


def _finite_real(value: object, *, name: str) -> float:
    """Return one finite real number while rejecting booleans."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _finite_state(
    value: Sequence[float] | np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    """Return one finite ``(x, y, yaw)`` state."""

    state = np.asarray(value, dtype=np.float64)
    if state.shape != (3,):
        raise ValueError(f"{name} must have shape (3,).")
    if not np.isfinite(state).all():
        raise ValueError(f"{name} must contain only finite values.")
    return np.ascontiguousarray(state, dtype=np.float64)


def _finite_state_path(
    value: Sequence[Sequence[float]] | np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    """Return a finite planar path with at least two states."""

    states = np.asarray(value, dtype=np.float64)
    if states.ndim != 2 or states.shape[1:] != (3,):
        raise ValueError(f"{name} must have shape (N, 3).")
    if states.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two states.")
    if not np.isfinite(states).all():
        raise ValueError(f"{name} must contain only finite values.")
    return np.ascontiguousarray(states, dtype=np.float64)


def _frozen_array(value: np.ndarray) -> np.ndarray:
    """Return one owned, contiguous, read-only float64 array."""

    result = np.array(value, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PlanarPathReferenceConfig:
    """Timing, actuator limits, and deterministic seed-controller settings."""

    time_step: float = 0.20
    minimum_linear_velocity: float = 0.0
    maximum_linear_velocity: float = 0.25
    minimum_angular_velocity: float = -0.8
    maximum_angular_velocity: float = 0.8
    heading_gain: float = 2.0
    stationary_position_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        values = {
            name: _finite_real(value, name=name)
            for name, value in (
                ("time_step", self.time_step),
                ("minimum_linear_velocity", self.minimum_linear_velocity),
                ("maximum_linear_velocity", self.maximum_linear_velocity),
                ("minimum_angular_velocity", self.minimum_angular_velocity),
                ("maximum_angular_velocity", self.maximum_angular_velocity),
                ("heading_gain", self.heading_gain),
                (
                    "stationary_position_tolerance",
                    self.stationary_position_tolerance,
                ),
            )
        }
        if values["time_step"] <= 0.0:
            raise ValueError("time_step must be greater than zero.")
        if (
            values["minimum_linear_velocity"]
            > values["maximum_linear_velocity"]
        ):
            raise ValueError(
                "minimum_linear_velocity cannot exceed "
                "maximum_linear_velocity."
            )
        if (
            values["minimum_angular_velocity"]
            > values["maximum_angular_velocity"]
        ):
            raise ValueError(
                "minimum_angular_velocity cannot exceed "
                "maximum_angular_velocity."
            )
        if not (
            values["minimum_linear_velocity"]
            <= 0.0
            <= values["maximum_linear_velocity"]
        ):
            raise ValueError("Linear-velocity bounds must contain zero.")
        if not (
            values["minimum_angular_velocity"]
            <= 0.0
            <= values["maximum_angular_velocity"]
        ):
            raise ValueError("Angular-velocity bounds must contain zero.")
        if values["heading_gain"] <= 0.0:
            raise ValueError("heading_gain must be greater than zero.")
        if values["stationary_position_tolerance"] < 0.0:
            raise ValueError(
                "stationary_position_tolerance cannot be negative."
            )

        for name, value in values.items():
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class PlanarPathReference:
    """Planar targets and one dynamically feasible unicycle seed rollout."""

    target_states: np.ndarray
    seed_chassis_states: np.ndarray
    seed_commands: np.ndarray

    def __post_init__(self) -> None:
        target_states = np.asarray(self.target_states, dtype=np.float64)
        seed_states = np.asarray(self.seed_chassis_states, dtype=np.float64)
        seed_commands = np.asarray(self.seed_commands, dtype=np.float64)
        if target_states.ndim != 2 or target_states.shape[1:] != (3,):
            raise ValueError("target_states must have shape (N, 3).")
        if seed_states.shape != target_states.shape:
            raise ValueError(
                "seed_chassis_states must match target_states shape."
            )
        expected_command_shape = (target_states.shape[0] - 1, 2)
        if seed_commands.shape != expected_command_shape:
            raise ValueError(
                "seed_commands must have shape "
                f"{expected_command_shape}."
            )
        if not all(
            np.isfinite(array).all()
            for array in (target_states, seed_states, seed_commands)
        ):
            raise ValueError("Reference arrays must contain only finite values.")

        object.__setattr__(
            self,
            "target_states",
            _frozen_array(target_states),
        )
        object.__setattr__(
            self,
            "seed_chassis_states",
            _frozen_array(seed_states),
        )
        object.__setattr__(
            self,
            "seed_commands",
            _frozen_array(seed_commands),
        )


def build_planar_path_reference(
    planar_states: Sequence[Sequence[float]] | np.ndarray,
    *,
    initial_chassis_state: Sequence[float] | np.ndarray,
    config: PlanarPathReferenceConfig | None = None,
) -> PlanarPathReference:
    """Build path-tangent targets and a feasible unicycle command seed.

    The planar path is translated so its first ``(x, y)`` position coincides
    with ``initial_chassis_state``. The oracle yaw values are deliberately not
    copied: a holonomic robot can translate laterally without facing its path,
    while Nova Carter's heading targets must follow the path tangent.
    """

    configuration = config or PlanarPathReferenceConfig()
    source_states = _finite_state_path(planar_states, name="planar_states")
    initial_state = _finite_state(
        initial_chassis_state,
        name="initial_chassis_state",
    )

    target_states = np.array(source_states, dtype=np.float64, copy=True)
    target_states[:, :2] += initial_state[:2] - source_states[0, :2]
    target_states[0, 2] = wrap_to_pi(float(initial_state[2]))

    previous_heading = float(target_states[0, 2])
    for state_index in range(1, target_states.shape[0]):
        segment = target_states[state_index, :2] - target_states[
            state_index - 1, :2
        ]
        if (
            float(np.linalg.norm(segment))
            > configuration.stationary_position_tolerance
        ):
            previous_heading = math.atan2(
                float(segment[1]),
                float(segment[0]),
            )
        target_states[state_index, 2] = wrap_to_pi(previous_heading)

    command_count = target_states.shape[0] - 1
    seed_states = np.empty_like(target_states)
    seed_commands = np.empty((command_count, 2), dtype=np.float64)
    seed_states[0] = initial_state

    for command_index in range(command_count):
        current_state = seed_states[command_index]
        next_target = target_states[command_index + 1]
        position_error = next_target[:2] - current_state[:2]
        distance = float(np.linalg.norm(position_error))

        if distance <= configuration.stationary_position_tolerance:
            desired_heading = float(next_target[2])
            desired_speed = 0.0
        else:
            desired_heading = math.atan2(
                float(position_error[1]),
                float(position_error[0]),
            )
            desired_speed = distance / configuration.time_step

        heading_error = wrap_to_pi(
            desired_heading - float(current_state[2])
        )
        angular_velocity = float(
            np.clip(
                configuration.heading_gain * heading_error,
                configuration.minimum_angular_velocity,
                configuration.maximum_angular_velocity,
            )
        )
        alignment = max(0.0, math.cos(heading_error))
        linear_velocity = float(
            np.clip(
                desired_speed * alignment,
                configuration.minimum_linear_velocity,
                configuration.maximum_linear_velocity,
            )
        )
        seed_commands[command_index] = (
            linear_velocity,
            angular_velocity,
        )
        seed_states[command_index + 1] = integrate_unicycle_euler(
            current_state,
            seed_commands[command_index],
            time_step=configuration.time_step,
        )

    return PlanarPathReference(
        target_states=target_states,
        seed_chassis_states=seed_states,
        seed_commands=seed_commands,
    )

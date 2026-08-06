"""Predictive Safe-SAGE safety filter for a unicycle robot.

The Safe-SAGE paper optimizes a finite sequence of reduced-order commands,
enforces a discrete-time CBF condition along the predicted trajectory, and
passes the first optimized command to a faster analytical safety filter.

Nova Carter cannot execute an arbitrary planar velocity ``(vx, vy)``.  This
module therefore preserves the Safe-SAGE predictive-filter structure while
rolling out the executable unicycle state ``(x, y, yaw)`` under commands
``(linear_velocity, angular_velocity)``.  The Safe-SAGE positional guidance
and numerical yaw derivative are used to linearize the predictive CBF
constraints, as described in the paper.

The module is pure Python/NumPy/SciPy and has no Isaac Sim dependency.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

import numpy as np
from scipy.optimize import minimize

from robot.control.unicycle_model import (
    integrate_unicycle_euler,
    lookahead_state,
    wrap_to_pi,
)
from robot.safety.laplace_guidance_field import (
    LaplaceGuidanceFieldResult,
)
from robot.safety.safe_sage_cbf_filter import SafeSAGECBFSafetyFilter
from robot.safety.safe_sage_configuration_space import (
    SafeSAGEConfigurationSpaceResult,
)
from robot.safety.safe_sage_h_full import SafeSAGEHFullResult


def _finite_real(value: object, *, name: str) -> float:
    """Return one finite real value while rejecting booleans."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _positive_integer(value: object, *, name: str) -> int:
    """Return one strictly positive integer while rejecting booleans."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    return value


def _finite_vector(
    value: Sequence[float] | np.ndarray,
    *,
    size: int,
    name: str,
) -> np.ndarray:
    """Return one validated one-dimensional float64 vector."""

    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},).")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return np.ascontiguousarray(array, dtype=np.float64)


def _frozen_array(value: np.ndarray) -> np.ndarray:
    """Return an owned, contiguous, read-only float64 array."""

    result = np.array(value, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class SafeSAGEUnicycleMPCConfig:
    """Prediction, actuator, cost, and numerical SQP settings."""

    horizon_steps: int = 12
    time_step: float = 0.20
    gamma: float = 1.0
    lookahead_distance: float = 0.10
    minimum_linear_velocity: float = 0.0
    maximum_linear_velocity: float = 0.25
    minimum_angular_velocity: float = -0.8
    maximum_angular_velocity: float = 0.8
    linear_velocity_weight: float = 1.0
    angular_velocity_weight: float = 0.10
    command_smoothness_weight: float = 0.01
    maximum_sqp_iterations: int = 6
    maximum_solver_iterations: int = 150
    solver_tolerance: float = 1e-7
    trajectory_tolerance: float = 1e-5
    constraint_tolerance: float = 1e-6
    intervention_tolerance: float = 1e-5

    def __post_init__(self) -> None:
        integer_values = {
            "horizon_steps": _positive_integer(
                self.horizon_steps,
                name="horizon_steps",
            ),
            "maximum_sqp_iterations": _positive_integer(
                self.maximum_sqp_iterations,
                name="maximum_sqp_iterations",
            ),
            "maximum_solver_iterations": _positive_integer(
                self.maximum_solver_iterations,
                name="maximum_solver_iterations",
            ),
        }
        real_values = {
            name: _finite_real(value, name=name)
            for name, value in (
                ("time_step", self.time_step),
                ("gamma", self.gamma),
                ("lookahead_distance", self.lookahead_distance),
                (
                    "minimum_linear_velocity",
                    self.minimum_linear_velocity,
                ),
                (
                    "maximum_linear_velocity",
                    self.maximum_linear_velocity,
                ),
                (
                    "minimum_angular_velocity",
                    self.minimum_angular_velocity,
                ),
                (
                    "maximum_angular_velocity",
                    self.maximum_angular_velocity,
                ),
                ("linear_velocity_weight", self.linear_velocity_weight),
                ("angular_velocity_weight", self.angular_velocity_weight),
                (
                    "command_smoothness_weight",
                    self.command_smoothness_weight,
                ),
                ("solver_tolerance", self.solver_tolerance),
                ("trajectory_tolerance", self.trajectory_tolerance),
                ("constraint_tolerance", self.constraint_tolerance),
                ("intervention_tolerance", self.intervention_tolerance),
            )
        }

        if real_values["time_step"] <= 0.0:
            raise ValueError("time_step must be greater than zero.")
        if real_values["gamma"] <= 0.0:
            raise ValueError("gamma must be greater than zero.")
        if real_values["lookahead_distance"] <= 0.0:
            raise ValueError(
                "lookahead_distance must be greater than zero."
            )
        if (
            real_values["minimum_linear_velocity"]
            > real_values["maximum_linear_velocity"]
        ):
            raise ValueError(
                "minimum_linear_velocity cannot exceed "
                "maximum_linear_velocity."
            )
        if (
            real_values["minimum_angular_velocity"]
            > real_values["maximum_angular_velocity"]
        ):
            raise ValueError(
                "minimum_angular_velocity cannot exceed "
                "maximum_angular_velocity."
            )
        if not (
            real_values["minimum_linear_velocity"]
            <= 0.0
            <= real_values["maximum_linear_velocity"]
        ):
            raise ValueError("Linear-velocity bounds must contain zero.")
        if not (
            real_values["minimum_angular_velocity"]
            <= 0.0
            <= real_values["maximum_angular_velocity"]
        ):
            raise ValueError("Angular-velocity bounds must contain zero.")
        for name in (
            "linear_velocity_weight",
            "angular_velocity_weight",
            "solver_tolerance",
            "trajectory_tolerance",
            "constraint_tolerance",
        ):
            if real_values[name] <= 0.0:
                raise ValueError(f"{name} must be greater than zero.")
        if real_values["command_smoothness_weight"] < 0.0:
            raise ValueError(
                "command_smoothness_weight cannot be negative."
            )
        if real_values["intervention_tolerance"] < 0.0:
            raise ValueError("intervention_tolerance cannot be negative.")

        for name, value in integer_values.items():
            object.__setattr__(self, name, value)
        for name, value in real_values.items():
            object.__setattr__(self, name, value)

    @property
    def dtcbf_decay(self) -> float:
        """Return the paper's ``exp(-gamma * dt)`` decay factor."""

        return math.exp(-self.gamma * self.time_step)


@dataclass(frozen=True)
class SafeSAGEUnicycleMPCResult:
    """Optimized command, predicted trajectory, and solver diagnostics."""

    command_sequence: np.ndarray
    predicted_chassis_states: np.ndarray
    predicted_safety_states: np.ndarray
    predicted_safety_values: np.ndarray
    dtcbf_margins: np.ndarray
    nominal_first_command: tuple[float, float]
    objective_value: float
    sqp_iteration_count: int
    solver_iteration_count: int
    success: bool
    intervention_active: bool
    fallback_active: bool
    status: str

    @property
    def first_command(self) -> tuple[float, float]:
        """Return ``(linear_velocity, angular_velocity)`` for execution."""

        return (
            float(self.command_sequence[0, 0]),
            float(self.command_sequence[0, 1]),
        )

    @property
    def linear_velocity(self) -> float:
        return self.first_command[0]

    @property
    def angular_velocity(self) -> float:
        return self.first_command[1]

    @property
    def minimum_predicted_safety_value(self) -> float:
        return float(np.min(self.predicted_safety_values))

    @property
    def minimum_dtcbf_margin(self) -> float:
        return float(np.min(self.dtcbf_margins))


@dataclass(frozen=True)
class _ReferenceSamples:
    """Field samples used by one predictive CBF linearization."""

    safety_values: np.ndarray
    constraint_normals: np.ndarray
    scaled_temporal_derivatives: np.ndarray


class SafeSAGEUnicycleMPC:
    """Finite-horizon Safe-SAGE predictive filter for ``(v, omega)``."""

    def __init__(
        self,
        paper_filter: SafeSAGECBFSafetyFilter,
        config: SafeSAGEUnicycleMPCConfig | None = None,
    ) -> None:
        if not isinstance(paper_filter, SafeSAGECBFSafetyFilter):
            raise TypeError("paper_filter must be a SafeSAGECBFSafetyFilter.")
        self.paper_filter = paper_filter
        self.config = config or SafeSAGEUnicycleMPCConfig()
        if not isinstance(self.config, SafeSAGEUnicycleMPCConfig):
            raise TypeError("config must be a SafeSAGEUnicycleMPCConfig.")

    def plan(
        self,
        initial_chassis_state_grid: Sequence[float] | np.ndarray,
        nominal_command: Sequence[float] | np.ndarray,
        *,
        configuration_space: SafeSAGEConfigurationSpaceResult,
        h_full: SafeSAGEHFullResult,
        guidance_fields: Sequence[LaplaceGuidanceFieldResult],
        temporal_safety_derivative: float = 0.0,
    ) -> SafeSAGEUnicycleMPCResult:
        """Optimize a safe command horizon and return its first command.

        ``initial_chassis_state_grid`` is ``(x, y, yaw)`` in the fixed local
        field frame. ``nominal_command`` may be one ``(v, omega)`` command or
        an explicit ``(horizon_steps, 2)`` reference sequence.
        """

        initial_state = _finite_vector(
            initial_chassis_state_grid,
            size=3,
            name="initial_chassis_state_grid",
        )
        temporal_derivative = _finite_real(
            temporal_safety_derivative,
            name="temporal_safety_derivative",
        )
        nominal_sequence = self._nominal_sequence(nominal_command)
        clipped_nominal = self._clip_commands(nominal_sequence)
        command_guess = clipped_nominal.copy()
        reference_chassis = self._rollout(initial_state, command_guess)
        reference_safety = self._safety_states(reference_chassis)

        try:
            reference_samples = self._sample_reference(
                reference_safety,
                configuration_space=configuration_space,
                h_full=h_full,
                guidance_fields=guidance_fields,
                temporal_safety_derivative=temporal_derivative,
            )
        except (TypeError, ValueError) as error:
            return self._fallback_result(
                initial_state,
                clipped_nominal,
                status=f"invalid_initial_reference: {error}",
            )

        current_constraint = self.paper_filter.evaluate_constraint(
            query_position_grid_xy=reference_safety[0, :2],
            query_yaw_radians=float(reference_safety[0, 2]),
            configuration_space=configuration_space,
            h_full=h_full,
            guidance_fields=guidance_fields,
            temporal_safety_derivative=temporal_derivative,
        )
        if current_constraint.unsafe_reason is not None:
            return self._fallback_result(
                initial_state,
                clipped_nominal,
                status=current_constraint.unsafe_reason,
            )

        solver_iterations = 0
        completed_sqp_iterations = 0
        latest_result = None
        latest_linearized_margins = np.empty(0, dtype=np.float64)

        for sqp_index in range(self.config.maximum_sqp_iterations):
            completed_sqp_iterations = sqp_index + 1
            previous_guess = command_guess.copy()
            flattened_guess = command_guess.reshape(-1)
            linearization_states = reference_safety
            linearization_samples = reference_samples

            def objective(flattened_commands: np.ndarray) -> float:
                commands = flattened_commands.reshape(
                    self.config.horizon_steps,
                    2,
                )
                command_error = commands - nominal_sequence
                value = np.sum(
                    self.config.linear_velocity_weight
                    * np.square(command_error[:, 0])
                    + self.config.angular_velocity_weight
                    * np.square(command_error[:, 1])
                )
                if self.config.command_smoothness_weight > 0.0:
                    command_delta = np.diff(commands, axis=0)
                    value += self.config.command_smoothness_weight * float(
                        np.sum(np.square(command_delta))
                    )
                return float(value)

            def predictive_constraints(
                flattened_commands: np.ndarray,
                reference_states: np.ndarray = linearization_states,
                samples: _ReferenceSamples = linearization_samples,
            ) -> np.ndarray:
                commands = flattened_commands.reshape(
                    self.config.horizon_steps,
                    2,
                )
                candidate_chassis = self._rollout(
                    initial_state,
                    commands,
                )
                candidate_safety = self._safety_states(candidate_chassis)
                margins = self._linearized_dtcbf_margins(
                    candidate_safety,
                    reference_safety=reference_states,
                    reference_samples=samples,
                )
                grid_margins = self._grid_domain_margins(
                    candidate_safety,
                    configuration_space=configuration_space,
                )
                return np.concatenate((margins, grid_margins))

            optimization = minimize(
                objective,
                flattened_guess,
                method="SLSQP",
                bounds=self._solver_bounds(),
                constraints={"type": "ineq", "fun": predictive_constraints},
                options={
                    "ftol": self.config.solver_tolerance,
                    "maxiter": self.config.maximum_solver_iterations,
                    "disp": False,
                },
            )
            latest_result = optimization
            solver_iterations += int(getattr(optimization, "nit", 0))
            if not bool(optimization.success):
                return self._fallback_result(
                    initial_state,
                    clipped_nominal,
                    status=f"solver_failure: {optimization.message}",
                    sqp_iterations=completed_sqp_iterations,
                    solver_iterations=solver_iterations,
                )

            command_guess = self._clip_commands(
                np.asarray(optimization.x, dtype=np.float64).reshape(
                    self.config.horizon_steps,
                    2,
                )
            )
            candidate_chassis = self._rollout(initial_state, command_guess)
            candidate_safety = self._safety_states(candidate_chassis)
            latest_linearized_margins = self._linearized_dtcbf_margins(
                candidate_safety,
                reference_safety=reference_safety,
                reference_samples=reference_samples,
            )
            command_change = float(
                np.max(np.abs(command_guess - previous_guess))
            )
            reference_chassis = candidate_chassis
            reference_safety = candidate_safety

            try:
                reference_samples = self._sample_reference(
                    reference_safety,
                    configuration_space=configuration_space,
                    h_full=h_full,
                    guidance_fields=guidance_fields,
                    temporal_safety_derivative=temporal_derivative,
                )
            except (TypeError, ValueError) as error:
                return self._fallback_result(
                    initial_state,
                    clipped_nominal,
                    status=f"invalid_predicted_reference: {error}",
                    sqp_iterations=completed_sqp_iterations,
                    solver_iterations=solver_iterations,
                )

            true_margins = self._sampled_dtcbf_margins(reference_samples)
            if (
                command_change <= self.config.trajectory_tolerance
                and float(np.min(true_margins))
                >= -self.config.constraint_tolerance
            ):
                break

        if latest_result is None:
            raise RuntimeError("The SQP loop did not execute.")

        predicted_values = self._time_adjusted_safety_values(
            reference_samples
        )
        true_margins = self._sampled_dtcbf_margins(reference_samples)
        if float(np.min(true_margins)) < -self.config.constraint_tolerance:
            return self._fallback_result(
                initial_state,
                clipped_nominal,
                status="sqp_did_not_satisfy_predictive_cbf",
                sqp_iterations=completed_sqp_iterations,
                solver_iterations=solver_iterations,
            )
        if latest_linearized_margins.size and (
            float(np.min(latest_linearized_margins))
            < -self.config.constraint_tolerance
        ):
            return self._fallback_result(
                initial_state,
                clipped_nominal,
                status="linearized_cbf_constraint_violation",
                sqp_iterations=completed_sqp_iterations,
                solver_iterations=solver_iterations,
            )

        intervention = bool(
            np.any(
                np.abs(command_guess[0] - nominal_sequence[0])
                > self.config.intervention_tolerance
            )
        )
        return SafeSAGEUnicycleMPCResult(
            command_sequence=_frozen_array(command_guess),
            predicted_chassis_states=_frozen_array(reference_chassis),
            predicted_safety_states=_frozen_array(reference_safety),
            predicted_safety_values=_frozen_array(predicted_values),
            dtcbf_margins=_frozen_array(true_margins),
            nominal_first_command=(
                float(nominal_sequence[0, 0]),
                float(nominal_sequence[0, 1]),
            ),
            objective_value=float(latest_result.fun),
            sqp_iteration_count=completed_sqp_iterations,
            solver_iteration_count=solver_iterations,
            success=True,
            intervention_active=intervention,
            fallback_active=False,
            status="converged",
        )

    def _nominal_sequence(
        self,
        nominal_command: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Expand one nominal command or validate a complete sequence."""

        command = np.asarray(nominal_command, dtype=np.float64)
        if command.shape == (2,):
            command = np.repeat(
                command[np.newaxis, :],
                self.config.horizon_steps,
                axis=0,
            )
        expected_shape = (self.config.horizon_steps, 2)
        if command.shape != expected_shape:
            raise ValueError(
                "nominal_command must have shape (2,) or "
                f"{expected_shape}."
            )
        if not np.isfinite(command).all():
            raise ValueError("nominal_command must contain finite values.")
        return np.ascontiguousarray(command, dtype=np.float64)

    def _clip_commands(self, commands: np.ndarray) -> np.ndarray:
        """Clip a command sequence to the executable actuator box."""

        clipped = np.array(commands, dtype=np.float64, copy=True)
        clipped[:, 0] = np.clip(
            clipped[:, 0],
            self.config.minimum_linear_velocity,
            self.config.maximum_linear_velocity,
        )
        clipped[:, 1] = np.clip(
            clipped[:, 1],
            self.config.minimum_angular_velocity,
            self.config.maximum_angular_velocity,
        )
        return clipped

    def _rollout(
        self,
        initial_state: np.ndarray,
        commands: np.ndarray,
    ) -> np.ndarray:
        """Roll out chassis states using executable unicycle dynamics."""

        states = np.empty(
            (self.config.horizon_steps + 1, 3),
            dtype=np.float64,
        )
        states[0] = initial_state
        for index, command in enumerate(commands):
            states[index + 1] = integrate_unicycle_euler(
                states[index],
                command,
                time_step=self.config.time_step,
            )
        return states

    def _safety_states(self, chassis_states: np.ndarray) -> np.ndarray:
        """Map every chassis state to the forward look-ahead safety state."""

        return np.stack(
            [
                lookahead_state(
                    state,
                    lookahead_distance=self.config.lookahead_distance,
                )
                for state in chassis_states
            ]
        )

    def _sample_reference(
        self,
        safety_states: np.ndarray,
        *,
        configuration_space: SafeSAGEConfigurationSpaceResult,
        h_full: SafeSAGEHFullResult,
        guidance_fields: Sequence[LaplaceGuidanceFieldResult],
        temporal_safety_derivative: float,
    ) -> _ReferenceSamples:
        """Sample ``h_full``, LGF/yaw normals, and temporal scaling."""

        values: list[float] = []
        normals: list[tuple[float, float, float]] = []
        temporal_terms: list[float] = []
        for state in safety_states:
            evaluation = self.paper_filter.evaluate_constraint(
                query_position_grid_xy=state[:2],
                query_yaw_radians=float(state[2]),
                configuration_space=configuration_space,
                h_full=h_full,
                guidance_fields=guidance_fields,
                temporal_safety_derivative=temporal_safety_derivative,
            )
            values.append(evaluation.sampled_safety_value)
            normals.append(evaluation.constraint_normal)
            temporal_terms.append(
                evaluation.scaled_temporal_safety_derivative
            )
        return _ReferenceSamples(
            safety_values=np.asarray(values, dtype=np.float64),
            constraint_normals=np.asarray(normals, dtype=np.float64),
            scaled_temporal_derivatives=np.asarray(
                temporal_terms,
                dtype=np.float64,
            ),
        )

    def _time_adjusted_safety_values(
        self,
        samples: _ReferenceSamples,
    ) -> np.ndarray:
        """Forward-propagate the paper's temporal safety estimate."""

        times = self.config.time_step * np.arange(
            self.config.horizon_steps + 1,
            dtype=np.float64,
        )
        return (
            samples.safety_values
            + times * samples.scaled_temporal_derivatives
        )

    def _linearized_dtcbf_margins(
        self,
        candidate_safety: np.ndarray,
        *,
        reference_safety: np.ndarray,
        reference_samples: _ReferenceSamples,
    ) -> np.ndarray:
        """Return affine Safe-SAGE DTCBF margins for one SQP subproblem."""

        state_delta = candidate_safety - reference_safety
        state_delta[:, 2] = np.asarray(
            [wrap_to_pi(float(value)) for value in state_delta[:, 2]],
            dtype=np.float64,
        )
        linearized_values = (
            self._time_adjusted_safety_values(reference_samples)
            + np.sum(
                reference_samples.constraint_normals * state_delta,
                axis=1,
            )
        )
        return (
            linearized_values[1:]
            - self.config.dtcbf_decay * linearized_values[:-1]
        )

    def _sampled_dtcbf_margins(
        self,
        samples: _ReferenceSamples,
    ) -> np.ndarray:
        """Return nonlinear sampled DTCBF margins for diagnostics."""

        values = self._time_adjusted_safety_values(samples)
        return values[1:] - self.config.dtcbf_decay * values[:-1]

    def _grid_domain_margins(
        self,
        safety_states: np.ndarray,
        *,
        configuration_space: SafeSAGEConfigurationSpaceResult,
    ) -> np.ndarray:
        """Keep predicted queries inside the bilinear sampling domain."""

        grid_config = configuration_space.occupancy_grids[0].config
        half_cell = 0.5 * grid_config.resolution
        minimum_x = grid_config.minimum_x + half_cell
        maximum_x = grid_config.maximum_x - half_cell
        minimum_y = grid_config.minimum_y + half_cell
        maximum_y = grid_config.maximum_y - half_cell
        x_values = safety_states[:, 0]
        y_values = safety_states[:, 1]
        return np.concatenate(
            (
                x_values - minimum_x,
                maximum_x - x_values,
                y_values - minimum_y,
                maximum_y - y_values,
            )
        )

    def _solver_bounds(self) -> tuple[tuple[float, float], ...]:
        """Return flattened per-stage actuator bounds for SciPy."""

        stage_bounds = (
            (
                self.config.minimum_linear_velocity,
                self.config.maximum_linear_velocity,
            ),
            (
                self.config.minimum_angular_velocity,
                self.config.maximum_angular_velocity,
            ),
        )
        return stage_bounds * self.config.horizon_steps

    def _fallback_result(
        self,
        initial_state: np.ndarray,
        nominal_sequence: np.ndarray,
        *,
        status: str,
        sqp_iterations: int = 0,
        solver_iterations: int = 0,
    ) -> SafeSAGEUnicycleMPCResult:
        """Return a deterministic fail-closed stop trajectory."""

        commands = np.zeros(
            (self.config.horizon_steps, 2),
            dtype=np.float64,
        )
        chassis_states = self._rollout(initial_state, commands)
        safety_states = self._safety_states(chassis_states)
        values = np.full(
            self.config.horizon_steps + 1,
            np.nan,
            dtype=np.float64,
        )
        margins = np.full(
            self.config.horizon_steps,
            np.nan,
            dtype=np.float64,
        )
        return SafeSAGEUnicycleMPCResult(
            command_sequence=_frozen_array(commands),
            predicted_chassis_states=_frozen_array(chassis_states),
            predicted_safety_states=_frozen_array(safety_states),
            predicted_safety_values=_frozen_array(values),
            dtcbf_margins=_frozen_array(margins),
            nominal_first_command=(
                float(nominal_sequence[0, 0]),
                float(nominal_sequence[0, 1]),
            ),
            objective_value=math.inf,
            sqp_iteration_count=sqp_iterations,
            solver_iteration_count=solver_iterations,
            success=False,
            intervention_active=True,
            fallback_active=True,
            status=status,
        )

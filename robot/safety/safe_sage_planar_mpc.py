"""Predictive Safe-SAGE filter in the paper's planar command space.

This module is an experimental oracle, not a Nova Carter wheel controller.
It deliberately uses the reduced-order state and input from Safe-SAGE:

    state   = (x, y, yaw)
    command = (v_x, v_y, omega)
    state[k + 1] = state[k] + dt * command[k]

The optimization minimizes deviation from a nominal planar velocity while
enforcing the exact sampled discrete-time CBF condition

    h[k + 1] >= exp(-gamma * dt) * h[k].

SciPy's SLSQP implementation supplies the sequential quadratic programming
loop.  Its CBF Jacobian uses ``(v_sem_x, v_sem_y, dh_full/dyaw)``, matching
the Safe-SAGE predictive linearization.  Every returned trajectory is then
checked against the nonlinear interpolated ``h_full`` values and unsafe
configuration-space masks.  An exact-gradient restoration is attempted if
the semantic linearization does not produce a feasible trajectory.

No Isaac Sim dependency is introduced here, and no result from this module
may be sent directly to a differential-drive base.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

import numpy as np
from robot.control.planar_single_integrator import integrate_planar_euler
from robot.safety.laplace_guidance_field import (
    LaplaceGuidanceFieldResult,
)
from robot.safety.safe_sage_cbf_filter import SafeSAGECBFSafetyFilter
from robot.safety.safe_sage_configuration_space import (
    SafeSAGEConfigurationSpaceResult,
)
from robot.safety.safe_sage_h_full import SafeSAGEHFullResult
from scipy.optimize import Bounds, minimize


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
class SafeSAGEPlanarMPCConfig:
    """Prediction horizon, planar bounds, costs, and SQP tolerances."""

    horizon_steps: int = 30
    time_step: float = 0.20
    gamma: float = 1.0
    minimum_planar_velocity: float = -0.25
    maximum_planar_velocity: float = 0.25
    minimum_angular_velocity: float = -0.8
    maximum_angular_velocity: float = 0.8
    planar_velocity_weight: float = 1.0
    angular_velocity_weight: float = 0.02
    command_smoothness_weight: float = 0.005
    lateral_seed_speed: float = 0.15
    maximum_solver_iterations: int = 500
    solver_tolerance: float = 1e-8
    constraint_tolerance: float = 1e-6
    intervention_tolerance: float = 1e-5

    def __post_init__(self) -> None:
        integer_values = {
            "horizon_steps": _positive_integer(
                self.horizon_steps,
                name="horizon_steps",
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
                (
                    "minimum_planar_velocity",
                    self.minimum_planar_velocity,
                ),
                (
                    "maximum_planar_velocity",
                    self.maximum_planar_velocity,
                ),
                (
                    "minimum_angular_velocity",
                    self.minimum_angular_velocity,
                ),
                (
                    "maximum_angular_velocity",
                    self.maximum_angular_velocity,
                ),
                ("planar_velocity_weight", self.planar_velocity_weight),
                ("angular_velocity_weight", self.angular_velocity_weight),
                (
                    "command_smoothness_weight",
                    self.command_smoothness_weight,
                ),
                ("lateral_seed_speed", self.lateral_seed_speed),
                ("solver_tolerance", self.solver_tolerance),
                ("constraint_tolerance", self.constraint_tolerance),
                ("intervention_tolerance", self.intervention_tolerance),
            )
        }

        if real_values["time_step"] <= 0.0:
            raise ValueError("time_step must be greater than zero.")
        if real_values["gamma"] <= 0.0:
            raise ValueError("gamma must be greater than zero.")
        if (
            real_values["minimum_planar_velocity"]
            > real_values["maximum_planar_velocity"]
        ):
            raise ValueError(
                "minimum_planar_velocity cannot exceed "
                "maximum_planar_velocity."
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
            real_values["minimum_planar_velocity"]
            <= 0.0
            <= real_values["maximum_planar_velocity"]
        ):
            raise ValueError("Planar-velocity bounds must contain zero.")
        if not (
            real_values["minimum_angular_velocity"]
            <= 0.0
            <= real_values["maximum_angular_velocity"]
        ):
            raise ValueError("Angular-velocity bounds must contain zero.")
        for name in (
            "planar_velocity_weight",
            "angular_velocity_weight",
            "solver_tolerance",
            "constraint_tolerance",
        ):
            if real_values[name] <= 0.0:
                raise ValueError(f"{name} must be greater than zero.")
        for name in (
            "command_smoothness_weight",
            "lateral_seed_speed",
            "intervention_tolerance",
        ):
            if real_values[name] < 0.0:
                raise ValueError(f"{name} cannot be negative.")

        for name, value in integer_values.items():
            object.__setattr__(self, name, value)
        for name, value in real_values.items():
            object.__setattr__(self, name, value)

    @property
    def dtcbf_decay(self) -> float:
        """Return ``exp(-gamma * dt)`` from the Safe-SAGE DTCBF."""

        return math.exp(-self.gamma * self.time_step)


@dataclass(frozen=True)
class SafeSAGEPlanarMPCResult:
    """Optimized planar command horizon and exact safety diagnostics."""

    command_sequence: np.ndarray
    predicted_states: np.ndarray
    predicted_safety_values: np.ndarray
    dtcbf_margins: np.ndarray
    nominal_first_command: tuple[float, float, float]
    objective_value: float
    solver_iteration_count: int
    success: bool
    intervention_active: bool
    fallback_active: bool
    exact_restoration_used: bool
    status: str

    @property
    def first_command(self) -> tuple[float, float, float]:
        """Return the first idealized ``(v_x, v_y, omega)`` command."""

        return (
            float(self.command_sequence[0, 0]),
            float(self.command_sequence[0, 1]),
            float(self.command_sequence[0, 2]),
        )

    @property
    def planar_velocity_grid_xy(self) -> tuple[float, float]:
        return self.first_command[0], self.first_command[1]

    @property
    def angular_velocity(self) -> float:
        return self.first_command[2]

    @property
    def minimum_predicted_safety_value(self) -> float:
        return float(np.min(self.predicted_safety_values))

    @property
    def minimum_dtcbf_margin(self) -> float:
        return float(np.min(self.dtcbf_margins))


@dataclass(frozen=True)
class _TrajectoryEvaluation:
    """One planar rollout and all sampled field values."""

    commands: np.ndarray
    states: np.ndarray
    safety_values: np.ndarray
    semantic_normals: np.ndarray
    exact_gradients: np.ndarray
    unsafe_configuration_mask: np.ndarray
    margins: np.ndarray


class SafeSAGEPlanarMPC:
    """Finite-horizon Safe-SAGE oracle for ``(v_x, v_y, omega)``."""

    def __init__(
        self,
        paper_filter: SafeSAGECBFSafetyFilter,
        config: SafeSAGEPlanarMPCConfig | None = None,
    ) -> None:
        if not isinstance(paper_filter, SafeSAGECBFSafetyFilter):
            raise TypeError("paper_filter must be a SafeSAGECBFSafetyFilter.")
        self.paper_filter = paper_filter
        self.config = config or SafeSAGEPlanarMPCConfig()
        if not isinstance(self.config, SafeSAGEPlanarMPCConfig):
            raise TypeError("config must be a SafeSAGEPlanarMPCConfig.")

    def plan(
        self,
        initial_state_grid: Sequence[float] | np.ndarray,
        nominal_command: Sequence[float] | np.ndarray,
        *,
        initial_command_guess: Sequence[float] | np.ndarray | None = None,
        configuration_space: SafeSAGEConfigurationSpaceResult,
        h_full: SafeSAGEHFullResult,
        guidance_fields: Sequence[LaplaceGuidanceFieldResult],
    ) -> SafeSAGEPlanarMPCResult:
        """Return a safe idealized planar command horizon.

        ``nominal_command`` may be one ``(v_x, v_y, omega)`` vector or a
        complete ``(horizon_steps, 3)`` reference sequence.  The returned
        command remains in the paper's reduced-order space and is not an
        executable Nova Carter wheel command.
        """

        initial_state = _finite_vector(
            initial_state_grid,
            size=3,
            name="initial_state_grid",
        )
        nominal_sequence = self._nominal_sequence(nominal_command)
        initial_evaluation = self._evaluate_commands(
            initial_state,
            np.zeros_like(nominal_sequence),
            configuration_space=configuration_space,
            h_full=h_full,
            guidance_fields=guidance_fields,
        )
        if initial_evaluation is None:
            return self._failed_result(
                initial_state,
                nominal_sequence,
                status="initial_state_outside_field_grid",
            )
        if initial_evaluation.unsafe_configuration_mask[0]:
            return self._failed_result(
                initial_state,
                nominal_sequence,
                status="unsafe_configuration_boundary",
            )

        seeds: list[tuple[str, np.ndarray]] = []
        if initial_command_guess is not None:
            seeds.append(
                (
                    "provided_warm_start",
                    self._command_sequence(
                        initial_command_guess,
                        name="initial_command_guess",
                    ),
                )
            )
        seeds.append(("nominal", nominal_sequence.copy()))
        seeds.append(
            (
                "analytical_rollout",
                self._analytical_rollout_seed(
                    initial_state,
                    nominal_sequence,
                    configuration_space=configuration_space,
                    h_full=h_full,
                    guidance_fields=guidance_fields,
                ),
            )
        )
        directional_seed = self._directional_seed(
            initial_evaluation,
            nominal_sequence,
        )
        if directional_seed is not None:
            seeds.append(("semantic_lateral", directional_seed))

        unique_seeds = self._unique_clipped_seeds(seeds)
        feasible_candidates: list[
            tuple[float, _TrajectoryEvaluation, int, bool, str]
        ] = []
        best_attempt: _TrajectoryEvaluation | None = None
        best_attempt_violation = math.inf
        total_iterations = 0

        for seed_name, seed in unique_seeds:
            evaluation, iterations, solver_success, solver_status = (
                self._solve_from_seed(
                    initial_state,
                    seed,
                    nominal_sequence=nominal_sequence,
                    configuration_space=configuration_space,
                    h_full=h_full,
                    guidance_fields=guidance_fields,
                    derivative_mode="semantic",
                )
            )
            total_iterations += iterations
            if evaluation is None:
                continue
            violation = self._maximum_violation(evaluation)
            if violation < best_attempt_violation:
                best_attempt = evaluation
                best_attempt_violation = violation
            if self._trajectory_is_safe(evaluation):
                feasible_candidates.append(
                    (
                        self._objective_value(
                            evaluation.commands,
                            nominal_sequence,
                        ),
                        evaluation,
                        iterations,
                        False,
                        (
                            "converged"
                            if solver_success
                            else f"feasible_{seed_name}_{solver_status}"
                        ),
                    )
                )

        if not feasible_candidates and best_attempt is not None:
            restored, iterations, solver_success, solver_status = (
                self._solve_from_seed(
                    initial_state,
                    best_attempt.commands,
                    nominal_sequence=nominal_sequence,
                    configuration_space=configuration_space,
                    h_full=h_full,
                    guidance_fields=guidance_fields,
                    derivative_mode="exact",
                )
            )
            total_iterations += iterations
            if restored is not None and self._trajectory_is_safe(restored):
                feasible_candidates.append(
                    (
                        self._objective_value(
                            restored.commands,
                            nominal_sequence,
                        ),
                        restored,
                        iterations,
                        True,
                        (
                            "exact_nonlinear_feasibility_restored"
                            if solver_success
                            else f"exact_feasible_{solver_status}"
                        ),
                    )
                )

        if feasible_candidates:
            (
                objective_value,
                evaluation,
                _,
                restoration_used,
                status,
            ) = min(feasible_candidates, key=lambda candidate: candidate[0])
            intervention = bool(
                np.any(
                    np.abs(evaluation.commands - nominal_sequence)
                    > self.config.intervention_tolerance
                )
            )
            return self._result(
                evaluation,
                nominal_sequence=nominal_sequence,
                objective_value=objective_value,
                solver_iteration_count=total_iterations,
                success=True,
                intervention_active=intervention,
                fallback_active=False,
                exact_restoration_used=restoration_used,
                status=status,
            )

        stationary = self._evaluate_commands(
            initial_state,
            np.zeros_like(nominal_sequence),
            configuration_space=configuration_space,
            h_full=h_full,
            guidance_fields=guidance_fields,
        )
        if stationary is not None and self._trajectory_is_safe(stationary):
            return self._result(
                stationary,
                nominal_sequence=nominal_sequence,
                objective_value=self._objective_value(
                    stationary.commands,
                    nominal_sequence,
                ),
                solver_iteration_count=total_iterations,
                success=True,
                intervention_active=True,
                fallback_active=True,
                exact_restoration_used=False,
                status="stationary_feasibility_recovered",
            )

        return self._failed_result(
            initial_state,
            nominal_sequence,
            status="no_safe_planar_horizon",
            solver_iteration_count=total_iterations,
        )

    def _solve_from_seed(
        self,
        initial_state: np.ndarray,
        seed: np.ndarray,
        *,
        nominal_sequence: np.ndarray,
        configuration_space: SafeSAGEConfigurationSpaceResult,
        h_full: SafeSAGEHFullResult,
        guidance_fields: Sequence[LaplaceGuidanceFieldResult],
        derivative_mode: str,
    ) -> tuple[_TrajectoryEvaluation | None, int, bool, str]:
        """Run one bounded SLSQP solve from a deterministic seed."""

        if derivative_mode not in {"semantic", "exact"}:
            raise ValueError("derivative_mode must be 'semantic' or 'exact'.")
        flat_seed = self._clip_commands(seed).reshape(-1)

        def constraint_values(flat_commands: np.ndarray) -> np.ndarray:
            evaluation = self._evaluate_commands(
                initial_state,
                flat_commands.reshape(self.config.horizon_steps, 3),
                configuration_space=configuration_space,
                h_full=h_full,
                guidance_fields=guidance_fields,
            )
            if evaluation is None:
                return np.full(
                    self.config.horizon_steps,
                    -1e3,
                    dtype=np.float64,
                )
            values = evaluation.margins.copy()
            invalid_next = evaluation.unsafe_configuration_mask[1:]
            values[invalid_next] = np.minimum(values[invalid_next], -1.0)
            return values

        def constraint_jacobian(flat_commands: np.ndarray) -> np.ndarray:
            evaluation = self._evaluate_commands(
                initial_state,
                flat_commands.reshape(self.config.horizon_steps, 3),
                configuration_space=configuration_space,
                h_full=h_full,
                guidance_fields=guidance_fields,
            )
            if evaluation is None:
                return np.zeros(
                    (self.config.horizon_steps, flat_commands.size),
                    dtype=np.float64,
                )
            derivatives = (
                evaluation.semantic_normals
                if derivative_mode == "semantic"
                else evaluation.exact_gradients
            )
            return self._dtcbf_command_jacobian(derivatives)

        result = minimize(
            fun=lambda flat: self._objective_value(
                flat.reshape(self.config.horizon_steps, 3),
                nominal_sequence,
            ),
            x0=flat_seed,
            jac=lambda flat: self._objective_gradient(
                flat.reshape(self.config.horizon_steps, 3),
                nominal_sequence,
            ).reshape(-1),
            method="SLSQP",
            bounds=self._flat_bounds(),
            constraints=(
                {
                    "type": "ineq",
                    "fun": constraint_values,
                    "jac": constraint_jacobian,
                },
            ),
            options={
                "maxiter": self.config.maximum_solver_iterations,
                "ftol": self.config.solver_tolerance,
                "disp": False,
            },
        )
        commands = self._clip_commands(
            np.asarray(result.x, dtype=np.float64).reshape(
                self.config.horizon_steps,
                3,
            )
        )
        evaluation = self._evaluate_commands(
            initial_state,
            commands,
            configuration_space=configuration_space,
            h_full=h_full,
            guidance_fields=guidance_fields,
        )
        return (
            evaluation,
            int(getattr(result, "nit", 0)),
            bool(result.success),
            self._normalized_status(str(result.message)),
        )

    def _analytical_rollout_seed(
        self,
        initial_state: np.ndarray,
        nominal_sequence: np.ndarray,
        *,
        configuration_space: SafeSAGEConfigurationSpaceResult,
        h_full: SafeSAGEHFullResult,
        guidance_fields: Sequence[LaplaceGuidanceFieldResult],
    ) -> np.ndarray:
        """Roll out the existing paper-space analytical filter as a seed."""

        commands = np.zeros_like(nominal_sequence)
        state = initial_state.copy()
        for index, nominal in enumerate(nominal_sequence):
            try:
                result = self.paper_filter.filter_command(
                    nominal[:2],
                    float(nominal[2]),
                    query_position_grid_xy=state[:2],
                    query_yaw_radians=float(state[2]),
                    configuration_space=configuration_space,
                    h_full=h_full,
                    guidance_fields=guidance_fields,
                )
            except ValueError:
                break
            if result.emergency_stop_active:
                break
            commands[index] = (
                result.safe_planar_velocity_grid_xy[0],
                result.safe_planar_velocity_grid_xy[1],
                result.safe_angular_velocity,
            )
            state = integrate_planar_euler(
                state,
                commands[index],
                time_step=self.config.time_step,
            )
        return self._clip_commands(commands)

    def _directional_seed(
        self,
        initial_evaluation: _TrajectoryEvaluation,
        nominal_sequence: np.ndarray,
    ) -> np.ndarray | None:
        """Bias one warm start toward the initial semantic passing side."""

        lateral_guidance = float(initial_evaluation.semantic_normals[0, 1])
        if (
            abs(lateral_guidance) <= self.config.constraint_tolerance
            or self.config.lateral_seed_speed <= 0.0
        ):
            return None
        seed = nominal_sequence.copy()
        seed[:, 1] = math.copysign(
            self.config.lateral_seed_speed,
            lateral_guidance,
        )
        return self._clip_commands(seed)

    def _evaluate_commands(
        self,
        initial_state: np.ndarray,
        commands: np.ndarray,
        *,
        configuration_space: SafeSAGEConfigurationSpaceResult,
        h_full: SafeSAGEHFullResult,
        guidance_fields: Sequence[LaplaceGuidanceFieldResult],
    ) -> _TrajectoryEvaluation | None:
        """Roll out commands and sample exact/semantic field information."""

        states = self._rollout(initial_state, commands)
        safety_values = np.empty(states.shape[0], dtype=np.float64)
        semantic_normals = np.empty_like(states)
        exact_gradients = np.empty_like(states)
        unsafe_mask = np.zeros(states.shape[0], dtype=bool)
        for index, state in enumerate(states):
            try:
                evaluation = self.paper_filter.evaluate_constraint(
                    query_position_grid_xy=state[:2],
                    query_yaw_radians=float(state[2]),
                    configuration_space=configuration_space,
                    h_full=h_full,
                    guidance_fields=guidance_fields,
                )
            except ValueError:
                return None
            safety_values[index] = evaluation.sampled_safety_value
            semantic_normals[index] = evaluation.constraint_normal
            exact_gradients[index] = evaluation.interpolant_gradient
            unsafe_mask[index] = evaluation.unsafe_reason is not None
        margins = (
            safety_values[1:]
            - self.config.dtcbf_decay * safety_values[:-1]
        )
        return _TrajectoryEvaluation(
            commands=np.ascontiguousarray(commands, dtype=np.float64),
            states=states,
            safety_values=safety_values,
            semantic_normals=semantic_normals,
            exact_gradients=exact_gradients,
            unsafe_configuration_mask=unsafe_mask,
            margins=margins,
        )

    def _rollout(
        self,
        initial_state: np.ndarray,
        commands: np.ndarray,
    ) -> np.ndarray:
        """Roll out the direct planar dynamics over one command horizon."""

        states = np.empty(
            (self.config.horizon_steps + 1, 3),
            dtype=np.float64,
        )
        states[0] = initial_state
        for index, command in enumerate(commands):
            states[index + 1] = integrate_planar_euler(
                states[index],
                command,
                time_step=self.config.time_step,
            )
        return states

    def _dtcbf_command_jacobian(
        self,
        state_derivatives: np.ndarray,
    ) -> np.ndarray:
        """Return DTCBF sensitivities with respect to the command horizon."""

        horizon = self.config.horizon_steps
        state_sensitivities = np.zeros(
            (horizon + 1, 3, 3 * horizon),
            dtype=np.float64,
        )
        identity_step = self.config.time_step * np.eye(3, dtype=np.float64)
        for state_index in range(1, horizon + 1):
            state_sensitivities[state_index] = state_sensitivities[
                state_index - 1
            ]
            command_start = 3 * (state_index - 1)
            state_sensitivities[
                state_index,
                :,
                command_start : command_start + 3,
            ] += identity_step

        jacobian = np.empty((horizon, 3 * horizon), dtype=np.float64)
        for index in range(horizon):
            jacobian[index] = (
                state_derivatives[index + 1]
                @ state_sensitivities[index + 1]
                - self.config.dtcbf_decay
                * state_derivatives[index]
                @ state_sensitivities[index]
            )
        return jacobian

    def _objective_value(
        self,
        commands: np.ndarray,
        nominal_sequence: np.ndarray,
    ) -> float:
        """Return command tracking plus optional smoothness cost."""

        differences = commands - nominal_sequence
        weights = np.array(
            [
                self.config.planar_velocity_weight,
                self.config.planar_velocity_weight,
                self.config.angular_velocity_weight,
            ],
            dtype=np.float64,
        )
        value = float(np.sum(weights * differences * differences))
        if self.config.command_smoothness_weight > 0.0:
            changes = np.diff(commands, axis=0)
            value += self.config.command_smoothness_weight * float(
                np.sum(changes * changes)
            )
        return value

    def _objective_gradient(
        self,
        commands: np.ndarray,
        nominal_sequence: np.ndarray,
    ) -> np.ndarray:
        """Return the analytical gradient of the quadratic objective."""

        weights = np.array(
            [
                self.config.planar_velocity_weight,
                self.config.planar_velocity_weight,
                self.config.angular_velocity_weight,
            ],
            dtype=np.float64,
        )
        gradient = 2.0 * weights * (commands - nominal_sequence)
        smoothness = self.config.command_smoothness_weight
        if smoothness > 0.0 and commands.shape[0] > 1:
            changes = commands[1:] - commands[:-1]
            gradient[:-1] -= 2.0 * smoothness * changes
            gradient[1:] += 2.0 * smoothness * changes
        return gradient

    def _nominal_sequence(
        self,
        nominal_command: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Expand one nominal command or validate a full reference horizon."""

        array = np.asarray(nominal_command, dtype=np.float64)
        if array.shape == (3,):
            array = np.repeat(
                array[np.newaxis, :],
                self.config.horizon_steps,
                axis=0,
            )
        elif array.shape != (self.config.horizon_steps, 3):
            raise ValueError(
                "nominal_command must have shape (3,) or "
                f"({self.config.horizon_steps}, 3)."
            )
        if not np.isfinite(array).all():
            raise ValueError("nominal_command must contain only finite values.")
        return self._clip_commands(array)

    def _command_sequence(
        self,
        value: Sequence[float] | np.ndarray,
        *,
        name: str,
    ) -> np.ndarray:
        """Validate one explicit command horizon without broadcasting."""

        array = np.asarray(value, dtype=np.float64)
        expected_shape = (self.config.horizon_steps, 3)
        if array.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}.")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain only finite values.")
        return self._clip_commands(array)

    def _clip_commands(self, commands: np.ndarray) -> np.ndarray:
        """Clip one command horizon to the ideal planar actuator bounds."""

        result = np.asarray(commands, dtype=np.float64).copy()
        result[:, :2] = np.clip(
            result[:, :2],
            self.config.minimum_planar_velocity,
            self.config.maximum_planar_velocity,
        )
        result[:, 2] = np.clip(
            result[:, 2],
            self.config.minimum_angular_velocity,
            self.config.maximum_angular_velocity,
        )
        return np.ascontiguousarray(result, dtype=np.float64)

    def _flat_bounds(self) -> Bounds:
        """Return repeated box bounds for SciPy's SLSQP solver."""

        lower_one = np.array(
            [
                self.config.minimum_planar_velocity,
                self.config.minimum_planar_velocity,
                self.config.minimum_angular_velocity,
            ],
            dtype=np.float64,
        )
        upper_one = np.array(
            [
                self.config.maximum_planar_velocity,
                self.config.maximum_planar_velocity,
                self.config.maximum_angular_velocity,
            ],
            dtype=np.float64,
        )
        return Bounds(
            np.tile(lower_one, self.config.horizon_steps),
            np.tile(upper_one, self.config.horizon_steps),
        )

    def _unique_clipped_seeds(
        self,
        seeds: Sequence[tuple[str, np.ndarray]],
    ) -> list[tuple[str, np.ndarray]]:
        """Remove numerically identical multi-start command horizons."""

        unique: list[tuple[str, np.ndarray]] = []
        for name, seed in seeds:
            clipped = self._clip_commands(seed)
            if not any(
                np.array_equal(clipped, existing)
                for _, existing in unique
            ):
                unique.append((name, clipped))
        return unique

    def _trajectory_is_safe(self, evaluation: _TrajectoryEvaluation) -> bool:
        """Apply the exact nonlinear acceptance boundary."""

        return bool(
            np.isfinite(evaluation.safety_values).all()
            and np.isfinite(evaluation.margins).all()
            and not np.any(evaluation.unsafe_configuration_mask)
            and np.min(evaluation.margins)
            >= -self.config.constraint_tolerance
        )

    def _maximum_violation(self, evaluation: _TrajectoryEvaluation) -> float:
        """Return the largest exact DTCBF or unsafe-cell violation."""

        margin_violation = max(0.0, -float(np.min(evaluation.margins)))
        if np.any(evaluation.unsafe_configuration_mask):
            return max(1.0, margin_violation)
        return margin_violation

    def _result(
        self,
        evaluation: _TrajectoryEvaluation,
        *,
        nominal_sequence: np.ndarray,
        objective_value: float,
        solver_iteration_count: int,
        success: bool,
        intervention_active: bool,
        fallback_active: bool,
        exact_restoration_used: bool,
        status: str,
    ) -> SafeSAGEPlanarMPCResult:
        """Freeze one accepted trajectory into the public result type."""

        return SafeSAGEPlanarMPCResult(
            command_sequence=_frozen_array(evaluation.commands),
            predicted_states=_frozen_array(evaluation.states),
            predicted_safety_values=_frozen_array(evaluation.safety_values),
            dtcbf_margins=_frozen_array(evaluation.margins),
            nominal_first_command=tuple(
                float(value) for value in nominal_sequence[0]
            ),
            objective_value=float(objective_value),
            solver_iteration_count=int(solver_iteration_count),
            success=bool(success),
            intervention_active=bool(intervention_active),
            fallback_active=bool(fallback_active),
            exact_restoration_used=bool(exact_restoration_used),
            status=str(status),
        )

    def _failed_result(
        self,
        initial_state: np.ndarray,
        nominal_sequence: np.ndarray,
        *,
        status: str,
        solver_iteration_count: int = 0,
    ) -> SafeSAGEPlanarMPCResult:
        """Return a fail-closed zero-command diagnostic result."""

        zero_commands = np.zeros_like(nominal_sequence)
        states = self._rollout(initial_state, zero_commands)
        safety_values = np.full(
            self.config.horizon_steps + 1,
            np.nan,
            dtype=np.float64,
        )
        margins = np.full(
            self.config.horizon_steps,
            np.nan,
            dtype=np.float64,
        )
        return SafeSAGEPlanarMPCResult(
            command_sequence=_frozen_array(zero_commands),
            predicted_states=_frozen_array(states),
            predicted_safety_values=_frozen_array(safety_values),
            dtcbf_margins=_frozen_array(margins),
            nominal_first_command=tuple(
                float(value) for value in nominal_sequence[0]
            ),
            objective_value=math.inf,
            solver_iteration_count=int(solver_iteration_count),
            success=False,
            intervention_active=True,
            fallback_active=True,
            exact_restoration_used=False,
            status=status,
        )

    @staticmethod
    def _normalized_status(message: str) -> str:
        """Convert one SciPy solver message to a compact report token."""

        normalized = "_".join(message.strip().lower().split())
        return normalized.replace("/", "_") or "unknown_solver_status"

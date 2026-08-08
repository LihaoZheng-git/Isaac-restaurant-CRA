"""Sparse Safe-SAGE MPC in the paper's planar command space.

This module is a diagnostic oracle, not a Nova Carter wheel controller.  It
uses the reduced-order state and directly commanded velocity from Safe-SAGE:

    state   = (x, y, yaw)
    command = (v_x, v_y, omega)
    state[k + 1] = state[k] + dt * command[k]

The predictive filter applies explicit sequential quadratic programming
(SQP).  Each convex subproblem linearizes the discrete-time CBF condition

    h[k + 1] >= exp(-gamma * dt) * h[k]

about one reference trajectory.  The semantic guidance field supplies the
position components of the predictive derivative, while the interpolated
``h_full`` yaw derivative supplies its third component.  OSQP solves the
resulting sparse QP.  A separate exact-gradient SQP phase is used only when
the semantic phase does not restore the sampled nonlinear DTCBF.

Elastic variables keep an individual QP from failing on an infeasible local
linearization.  They never relax final acceptance: every returned moving
trajectory is checked against exact interpolated ``h_full`` values, exact
DTCBF margins, field bounds, and configuration-space unsafe masks.  The
controller otherwise returns a verified stationary horizon or fails closed.

No Isaac Sim dependency is introduced here, and no result from this module
may be sent directly to a differential-drive base.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

import numpy as np
import osqp

from robot.control.planar_single_integrator import integrate_planar_euler
from robot.safety.laplace_guidance_field import (
    LaplaceGuidanceFieldResult,
)
from robot.safety.safe_sage_cbf_filter import SafeSAGECBFSafetyFilter
from robot.safety.safe_sage_configuration_space import (
    SafeSAGEConfigurationSpaceResult,
)
from robot.safety.safe_sage_h_full import SafeSAGEHFullResult
from scipy import sparse


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
    """Prediction, cost, sparse SQP, and OSQP configuration."""

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
    maximum_sqp_iterations: int = 8
    maximum_restoration_iterations: int = 12
    maximum_solver_iterations: int = 4000
    planar_velocity_trust_region: float = 0.15
    angular_velocity_trust_region: float = 0.40
    cbf_slack_weight: float = 100.0
    solver_tolerance: float = 1e-6
    trajectory_tolerance: float = 1e-4
    constraint_tolerance: float = 1e-6
    intervention_tolerance: float = 1e-5
    line_search_contraction: float = 0.5
    maximum_line_search_iterations: int = 8

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
            "maximum_restoration_iterations": _positive_integer(
                self.maximum_restoration_iterations,
                name="maximum_restoration_iterations",
            ),
            "maximum_solver_iterations": _positive_integer(
                self.maximum_solver_iterations,
                name="maximum_solver_iterations",
            ),
            "maximum_line_search_iterations": _positive_integer(
                self.maximum_line_search_iterations,
                name="maximum_line_search_iterations",
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
                (
                    "planar_velocity_trust_region",
                    self.planar_velocity_trust_region,
                ),
                (
                    "angular_velocity_trust_region",
                    self.angular_velocity_trust_region,
                ),
                ("cbf_slack_weight", self.cbf_slack_weight),
                ("solver_tolerance", self.solver_tolerance),
                ("trajectory_tolerance", self.trajectory_tolerance),
                ("constraint_tolerance", self.constraint_tolerance),
                ("intervention_tolerance", self.intervention_tolerance),
                ("line_search_contraction", self.line_search_contraction),
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
            "planar_velocity_trust_region",
            "angular_velocity_trust_region",
            "cbf_slack_weight",
            "solver_tolerance",
            "trajectory_tolerance",
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
        if not 0.0 < real_values["line_search_contraction"] < 1.0:
            raise ValueError(
                "line_search_contraction must lie strictly between zero "
                "and one."
            )

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


@dataclass(frozen=True)
class _SparseQPSolution:
    """One sparse SQP subproblem result."""

    success: bool
    commands: np.ndarray
    iteration_count: int
    status: str
    maximum_cbf_slack: float


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
        complete ``(horizon_steps, 3)`` reference sequence.  A supplied
        ``initial_command_guess`` is the primary SQP reference; it does not
        change the nominal command-tracking objective.
        """

        initial_state = _finite_vector(
            initial_state_grid,
            size=3,
            name="initial_state_grid",
        )
        nominal_sequence = self._nominal_sequence(nominal_command)
        warm_start_supplied = initial_command_guess is not None
        if warm_start_supplied:
            reference_commands = self._command_sequence(
                initial_command_guess,
                name="initial_command_guess",
            )
            reference_source = "warm_start"
        else:
            reference_commands = nominal_sequence.copy()
            reference_source = "nominal"

        stationary_commands = np.zeros_like(nominal_sequence)
        stationary = self._evaluate_commands(
            initial_state,
            stationary_commands,
            configuration_space=configuration_space,
            h_full=h_full,
            guidance_fields=guidance_fields,
        )
        if stationary is None:
            return self._failed_result(
                initial_state,
                nominal_sequence,
                status="initial_state_outside_field_grid",
            )
        if stationary.unsafe_configuration_mask[0]:
            return self._failed_result(
                initial_state,
                nominal_sequence,
                status="unsafe_configuration_boundary",
            )

        reference = self._evaluate_commands(
            initial_state,
            reference_commands,
            configuration_space=configuration_space,
            h_full=h_full,
            guidance_fields=guidance_fields,
        )
        if reference is None:
            reference = stationary
            reference_source = "stationary"

        if not warm_start_supplied and not self._trajectory_is_safe(reference):
            directional_commands = self._directional_seed(
                stationary,
                nominal_sequence,
            )
            if directional_commands is not None:
                directional = self._evaluate_commands(
                    initial_state,
                    directional_commands,
                    configuration_space=configuration_space,
                    h_full=h_full,
                    guidance_fields=guidance_fields,
                )
                if (
                    directional is not None
                    and self._violation_metric(directional)
                    < self._violation_metric(reference)
                ):
                    reference = directional
                    reference_source = "directional_seed"

        total_solver_iterations = 0
        exact_restoration_used = False
        status = "nominal_horizon_safe"

        if (
            reference_source != "nominal"
            and self._trajectory_is_safe(reference)
        ):
            solution = self._solve_sparse_subproblem(
                reference,
                nominal_sequence=nominal_sequence,
                configuration_space=configuration_space,
                derivative_mode="guidance",
            )
            total_solver_iterations += solution.iteration_count
            if solution.success:
                candidate = self._evaluate_commands(
                    initial_state,
                    solution.commands,
                    configuration_space=configuration_space,
                    h_full=h_full,
                    guidance_fields=guidance_fields,
                )
                improved = self._select_safe_objective_step(
                    reference,
                    candidate,
                    initial_state=initial_state,
                    proposed_commands=solution.commands,
                    nominal_sequence=nominal_sequence,
                    configuration_space=configuration_space,
                    h_full=h_full,
                    guidance_fields=guidance_fields,
                )
                if improved is not None:
                    reference = improved
                    status = f"{reference_source}_objective_improved"
                else:
                    status = f"{reference_source}_exact_safe"
            else:
                status = f"{reference_source}_exact_safe_{solution.status}"

        if not self._trajectory_is_safe(reference):
            guidance_failure: str | None = None
            for _ in range(self.config.maximum_sqp_iterations):
                solution = self._solve_sparse_subproblem(
                    reference,
                    nominal_sequence=nominal_sequence,
                    configuration_space=configuration_space,
                    derivative_mode="guidance",
                )
                total_solver_iterations += solution.iteration_count
                if not solution.success:
                    guidance_failure = solution.status
                    break

                candidate = self._evaluate_commands(
                    initial_state,
                    solution.commands,
                    configuration_space=configuration_space,
                    h_full=h_full,
                    guidance_fields=guidance_fields,
                )
                accepted = self._select_violation_reducing_step(
                    reference,
                    candidate,
                    initial_state=initial_state,
                    proposed_commands=solution.commands,
                    configuration_space=configuration_space,
                    h_full=h_full,
                    guidance_fields=guidance_fields,
                )
                if accepted is None:
                    guidance_failure = "guidance_sqp_did_not_improve"
                    break
                command_change = float(
                    np.max(np.abs(accepted.commands - reference.commands))
                )
                reference = accepted
                if self._trajectory_is_safe(reference):
                    status = "guidance_sqp_converged"
                    break
                if command_change <= self.config.trajectory_tolerance:
                    guidance_failure = "guidance_sqp_stalled"
                    break

            if not self._trajectory_is_safe(reference):
                exact_restoration_used = True
                restoration_failure: str | None = guidance_failure
                for _ in range(self.config.maximum_restoration_iterations):
                    solution = self._solve_sparse_subproblem(
                        reference,
                        nominal_sequence=nominal_sequence,
                        configuration_space=configuration_space,
                        derivative_mode="gradient",
                    )
                    total_solver_iterations += solution.iteration_count
                    if not solution.success:
                        restoration_failure = solution.status
                        break

                    candidate = self._evaluate_commands(
                        initial_state,
                        solution.commands,
                        configuration_space=configuration_space,
                        h_full=h_full,
                        guidance_fields=guidance_fields,
                    )
                    accepted = self._select_violation_reducing_step(
                        reference,
                        candidate,
                        initial_state=initial_state,
                        proposed_commands=solution.commands,
                        configuration_space=configuration_space,
                        h_full=h_full,
                        guidance_fields=guidance_fields,
                    )
                    if accepted is None:
                        restoration_failure = (
                            "exact_qp_restoration_did_not_improve"
                        )
                        break
                    command_change = float(
                        np.max(
                            np.abs(accepted.commands - reference.commands)
                        )
                    )
                    reference = accepted
                    if self._trajectory_is_safe(reference):
                        status = "exact_dtcbf_feasibility_restored"
                        break
                    if command_change <= self.config.trajectory_tolerance:
                        restoration_failure = "exact_qp_restoration_stalled"
                        break

                if not self._trajectory_is_safe(reference):
                    return self._stationary_or_failed_result(
                        stationary,
                        initial_state=initial_state,
                        nominal_sequence=nominal_sequence,
                        solver_iteration_count=total_solver_iterations,
                        failure_status=(
                            restoration_failure
                            or "exact_qp_restoration_exhausted"
                        ),
                    )

        intervention = bool(
            np.any(
                np.abs(reference.commands - nominal_sequence)
                > self.config.intervention_tolerance
            )
        )
        return self._result(
            reference,
            nominal_sequence=nominal_sequence,
            objective_value=self._objective_value(
                reference.commands,
                nominal_sequence,
            ),
            solver_iteration_count=total_solver_iterations,
            success=True,
            intervention_active=intervention,
            fallback_active=False,
            exact_restoration_used=exact_restoration_used,
            status=status,
        )

    def _solve_sparse_subproblem(
        self,
        reference: _TrajectoryEvaluation,
        *,
        nominal_sequence: np.ndarray,
        configuration_space: SafeSAGEConfigurationSpaceResult,
        derivative_mode: str,
    ) -> _SparseQPSolution:
        """Assemble and solve one sparse Safe-SAGE SQP subproblem."""

        if derivative_mode == "guidance":
            derivatives = reference.semantic_normals
        elif derivative_mode == "gradient":
            derivatives = reference.exact_gradients
        else:
            raise ValueError("derivative_mode must be guidance or gradient.")

        horizon = self.config.horizon_steps
        command_variable_count = 3 * horizon
        slack_variable_count = horizon
        reference_flat = reference.commands.reshape(-1)
        state_sensitivities = self._state_sensitivities()

        cbf_rows = np.empty(
            (horizon, command_variable_count),
            dtype=np.float64,
        )
        cbf_lower = np.empty(horizon, dtype=np.float64)
        for index in range(horizon):
            row = (
                derivatives[index + 1] @ state_sensitivities[index + 1]
                - self.config.dtcbf_decay
                * derivatives[index]
                @ state_sensitivities[index]
            )
            cbf_rows[index] = row
            cbf_lower[index] = (
                float(row @ reference_flat) - reference.margins[index]
            )

        grid = configuration_space.occupancy_grids[0]
        half_cell = 0.5 * grid.config.resolution
        grid_minimum = np.array(
            [
                grid.config.minimum_x + half_cell,
                grid.config.minimum_y + half_cell,
            ],
            dtype=np.float64,
        )
        grid_maximum = np.array(
            [
                grid.config.maximum_x - half_cell,
                grid.config.maximum_y - half_cell,
            ],
            dtype=np.float64,
        )
        grid_rows: list[np.ndarray] = []
        grid_lower: list[float] = []
        grid_upper: list[float] = []
        for index in range(1, horizon + 1):
            for axis in (0, 1):
                row = state_sensitivities[index, axis]
                offset = float(row @ reference_flat)
                grid_rows.append(row)
                grid_lower.append(
                    grid_minimum[axis] - reference.states[index, axis] + offset
                )
                grid_upper.append(
                    grid_maximum[axis] - reference.states[index, axis] + offset
                )

        actuator_lower, actuator_upper = self._actuator_bounds()
        trust_radius = np.tile(
            np.array(
                [
                    self.config.planar_velocity_trust_region,
                    self.config.planar_velocity_trust_region,
                    self.config.angular_velocity_trust_region,
                ],
                dtype=np.float64,
            ),
            horizon,
        )
        command_lower = np.maximum(
            actuator_lower,
            reference_flat - trust_radius,
        )
        command_upper = np.minimum(
            actuator_upper,
            reference_flat + trust_radius,
        )

        command_identity = sparse.eye(command_variable_count, format="csc")
        slack_identity = sparse.eye(slack_variable_count, format="csc")
        zero_command_slack = sparse.csc_matrix(
            (command_variable_count, slack_variable_count)
        )
        zero_slack_command = sparse.csc_matrix(
            (slack_variable_count, command_variable_count)
        )
        zero_grid_slack = sparse.csc_matrix(
            (len(grid_rows), slack_variable_count)
        )
        command_matrix = sparse.hstack(
            (command_identity, zero_command_slack),
            format="csc",
        )
        slack_matrix = sparse.hstack(
            (zero_slack_command, slack_identity),
            format="csc",
        )
        cbf_matrix = sparse.hstack(
            (sparse.csc_matrix(cbf_rows), slack_identity),
            format="csc",
        )
        grid_matrix = sparse.hstack(
            (sparse.csc_matrix(np.asarray(grid_rows)), zero_grid_slack),
            format="csc",
        )
        constraint_matrix = sparse.vstack(
            (command_matrix, slack_matrix, cbf_matrix, grid_matrix),
            format="csc",
        )
        constraint_lower = np.concatenate(
            (
                command_lower,
                np.zeros(slack_variable_count, dtype=np.float64),
                cbf_lower,
                np.asarray(grid_lower, dtype=np.float64),
            )
        )
        constraint_upper = np.concatenate(
            (
                command_upper,
                np.full(slack_variable_count, np.inf),
                np.full(horizon, np.inf),
                np.asarray(grid_upper, dtype=np.float64),
            )
        )
        objective_matrix, objective_vector = self._quadratic_objective(
            nominal_sequence,
            slack_variable_count=slack_variable_count,
        )
        warm_start = np.concatenate(
            (
                reference_flat,
                np.maximum(0.0, -reference.margins),
            )
        )

        solver = osqp.OSQP()
        try:
            solver.setup(
                P=objective_matrix,
                q=objective_vector,
                A=constraint_matrix,
                l=constraint_lower,
                u=constraint_upper,
                verbose=False,
                eps_abs=self.config.solver_tolerance,
                eps_rel=self.config.solver_tolerance,
                max_iter=self.config.maximum_solver_iterations,
                polishing=True,
                warm_starting=True,
            )
            solver.warm_start(x=warm_start)
            result = solver.solve(raise_error=False)
        except (RuntimeError, ValueError) as error:
            return _SparseQPSolution(
                success=False,
                commands=reference.commands,
                iteration_count=0,
                status=f"osqp_exception: {error}",
                maximum_cbf_slack=math.inf,
            )

        solver_status = str(result.info.status)
        solved = solver_status.casefold() in {"solved", "solved inaccurate"}
        if not solved or result.x is None:
            return _SparseQPSolution(
                success=False,
                commands=reference.commands,
                iteration_count=int(result.info.iter),
                status=f"osqp_{solver_status}",
                maximum_cbf_slack=math.inf,
            )

        optimizer = np.asarray(result.x, dtype=np.float64)
        commands = self._clip_commands(
            optimizer[:command_variable_count].reshape(horizon, 3)
        )
        maximum_cbf_slack = float(
            np.max(optimizer[command_variable_count:], initial=0.0)
        )
        return _SparseQPSolution(
            success=True,
            commands=commands,
            iteration_count=int(result.info.iter),
            status=solver_status,
            maximum_cbf_slack=maximum_cbf_slack,
        )

    def _quadratic_objective(
        self,
        nominal_sequence: np.ndarray,
        *,
        slack_variable_count: int,
    ) -> tuple[sparse.csc_matrix, np.ndarray]:
        """Return OSQP's upper-triangular ``P`` and linear vector ``q``."""

        horizon = self.config.horizon_steps
        weights = np.tile(
            np.array(
                [
                    self.config.planar_velocity_weight,
                    self.config.planar_velocity_weight,
                    self.config.angular_velocity_weight,
                ],
                dtype=np.float64,
            ),
            horizon,
        )
        matrix = np.diag(2.0 * weights + 1e-10)
        vector = -2.0 * weights * nominal_sequence.reshape(-1)

        smoothness = self.config.command_smoothness_weight
        if smoothness > 0.0:
            for stage in range(horizon - 1):
                for command_axis in (0, 1, 2):
                    first = 3 * stage + command_axis
                    second = first + 3
                    matrix[first, first] += 2.0 * smoothness
                    matrix[second, second] += 2.0 * smoothness
                    matrix[first, second] -= 2.0 * smoothness
                    matrix[second, first] -= 2.0 * smoothness

        command_objective = sparse.csc_matrix(np.triu(matrix))
        slack_objective = (
            sparse.eye(slack_variable_count, format="csc") * 1e-10
        )
        objective_matrix = sparse.block_diag(
            (command_objective, slack_objective),
            format="csc",
        )
        objective_vector = np.concatenate(
            (
                vector,
                np.full(
                    slack_variable_count,
                    self.config.cbf_slack_weight,
                    dtype=np.float64,
                ),
            )
        )
        return objective_matrix, objective_vector

    def _state_sensitivities(self) -> np.ndarray:
        """Return exact direct-planar state sensitivities over the horizon."""

        horizon = self.config.horizon_steps
        sensitivities = np.zeros(
            (horizon + 1, 3, 3 * horizon),
            dtype=np.float64,
        )
        step_identity = self.config.time_step * np.eye(3, dtype=np.float64)
        for state_index in range(1, horizon + 1):
            sensitivities[state_index] = sensitivities[state_index - 1]
            command_start = 3 * (state_index - 1)
            sensitivities[
                state_index,
                :,
                command_start : command_start + 3,
            ] += step_identity
        return sensitivities

    def _select_safe_objective_step(
        self,
        reference: _TrajectoryEvaluation,
        candidate: _TrajectoryEvaluation | None,
        *,
        initial_state: np.ndarray,
        proposed_commands: np.ndarray,
        nominal_sequence: np.ndarray,
        configuration_space: SafeSAGEConfigurationSpaceResult,
        h_full: SafeSAGEHFullResult,
        guidance_fields: Sequence[LaplaceGuidanceFieldResult],
    ) -> _TrajectoryEvaluation | None:
        """Backtrack to an exact-safe step with a lower nominal cost."""

        reference_objective = self._objective_value(
            reference.commands,
            nominal_sequence,
        )

        def acceptable(trial: _TrajectoryEvaluation | None) -> bool:
            return bool(
                trial is not None
                and self._trajectory_is_safe(trial)
                and self._objective_value(trial.commands, nominal_sequence)
                < reference_objective - 1e-12
            )

        if acceptable(candidate):
            return candidate

        step_size = 1.0
        for _ in range(self.config.maximum_line_search_iterations):
            step_size *= self.config.line_search_contraction
            trial_commands = reference.commands + step_size * (
                proposed_commands - reference.commands
            )
            trial = self._evaluate_commands(
                initial_state,
                trial_commands,
                configuration_space=configuration_space,
                h_full=h_full,
                guidance_fields=guidance_fields,
            )
            if acceptable(trial):
                return trial
        return None

    def _select_violation_reducing_step(
        self,
        reference: _TrajectoryEvaluation,
        candidate: _TrajectoryEvaluation | None,
        *,
        initial_state: np.ndarray,
        proposed_commands: np.ndarray,
        configuration_space: SafeSAGEConfigurationSpaceResult,
        h_full: SafeSAGEHFullResult,
        guidance_fields: Sequence[LaplaceGuidanceFieldResult],
    ) -> _TrajectoryEvaluation | None:
        """Backtrack until exact nonlinear safety violation decreases."""

        reference_metric = self._violation_metric(reference)
        if candidate is not None and (
            self._trajectory_is_safe(candidate)
            or self._violation_metric(candidate) < reference_metric
        ):
            return candidate

        step_size = 1.0
        best: _TrajectoryEvaluation | None = None
        best_metric = reference_metric
        for _ in range(self.config.maximum_line_search_iterations):
            step_size *= self.config.line_search_contraction
            trial_commands = reference.commands + step_size * (
                proposed_commands - reference.commands
            )
            trial = self._evaluate_commands(
                initial_state,
                trial_commands,
                configuration_space=configuration_space,
                h_full=h_full,
                guidance_fields=guidance_fields,
            )
            if trial is None:
                continue
            metric = self._violation_metric(trial)
            if metric < best_metric:
                best = trial
                best_metric = metric
            if self._trajectory_is_safe(trial):
                return trial
        return best

    def _directional_seed(
        self,
        stationary_evaluation: _TrajectoryEvaluation,
        nominal_sequence: np.ndarray,
    ) -> np.ndarray | None:
        """Return one deterministic seed using the initial passing side."""

        lateral_guidance = float(
            stationary_evaluation.semantic_normals[0, 1]
        )
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
        """Roll out commands and sample exact and semantic field data."""

        clipped_commands = self._clip_commands(commands)
        states = self._rollout(initial_state, clipped_commands)
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
            except (TypeError, ValueError):
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
            commands=np.ascontiguousarray(
                clipped_commands,
                dtype=np.float64,
            ),
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

    def _nominal_sequence(
        self,
        nominal_command: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Expand one nominal command or validate a full reference."""

        return self._command_sequence(
            nominal_command,
            name="nominal_command",
        )

    def _command_sequence(
        self,
        value: Sequence[float] | np.ndarray,
        *,
        name: str,
    ) -> np.ndarray:
        """Expand one command or validate an explicit command horizon."""

        array = np.asarray(value, dtype=np.float64)
        if array.shape == (3,):
            array = np.repeat(
                array[np.newaxis, :],
                self.config.horizon_steps,
                axis=0,
            )
        expected_shape = (self.config.horizon_steps, 3)
        if array.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape (3,) or {expected_shape}."
            )
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain only finite values.")
        return self._clip_commands(array)

    def _clip_commands(self, commands: np.ndarray) -> np.ndarray:
        """Clip one command horizon to the ideal planar actuator box."""

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

    def _actuator_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return flattened lower and upper ideal command bounds."""

        lower = np.tile(
            np.array(
                [
                    self.config.minimum_planar_velocity,
                    self.config.minimum_planar_velocity,
                    self.config.minimum_angular_velocity,
                ],
                dtype=np.float64,
            ),
            self.config.horizon_steps,
        )
        upper = np.tile(
            np.array(
                [
                    self.config.maximum_planar_velocity,
                    self.config.maximum_planar_velocity,
                    self.config.maximum_angular_velocity,
                ],
                dtype=np.float64,
            ),
            self.config.horizon_steps,
        )
        return lower, upper

    def _trajectory_is_safe(self, evaluation: _TrajectoryEvaluation) -> bool:
        """Apply the exact nonlinear acceptance boundary."""

        return bool(
            np.isfinite(evaluation.safety_values).all()
            and np.isfinite(evaluation.margins).all()
            and not np.any(evaluation.unsafe_configuration_mask)
            and float(np.min(evaluation.safety_values))
            >= -self.config.constraint_tolerance
            and float(np.min(evaluation.margins))
            >= -self.config.constraint_tolerance
        )

    def _violation_metric(
        self,
        evaluation: _TrajectoryEvaluation,
    ) -> tuple[int, float]:
        """Return a lexicographic exact-safety violation measure."""

        unsafe_count = int(np.count_nonzero(
            evaluation.unsafe_configuration_mask
        ))
        if (
            not np.isfinite(evaluation.safety_values).all()
            or not np.isfinite(evaluation.margins).all()
        ):
            return unsafe_count + self.config.horizon_steps + 1, math.inf
        safety_violation = max(
            0.0,
            -float(np.min(evaluation.safety_values)),
        )
        margin_violation = max(
            0.0,
            -float(np.min(evaluation.margins)),
        )
        return unsafe_count, max(safety_violation, margin_violation)

    def _stationary_or_failed_result(
        self,
        stationary: _TrajectoryEvaluation,
        *,
        initial_state: np.ndarray,
        nominal_sequence: np.ndarray,
        solver_iteration_count: int,
        failure_status: str,
    ) -> SafeSAGEPlanarMPCResult:
        """Return a verified stop, otherwise a deterministic failure."""

        if self._trajectory_is_safe(stationary):
            return self._result(
                stationary,
                nominal_sequence=nominal_sequence,
                objective_value=self._objective_value(
                    stationary.commands,
                    nominal_sequence,
                ),
                solver_iteration_count=solver_iteration_count,
                success=True,
                intervention_active=True,
                fallback_active=True,
                exact_restoration_used=True,
                status=(
                    "stationary_feasibility_recovered_after_"
                    f"{failure_status}"
                ),
            )
        return self._failed_result(
            initial_state,
            nominal_sequence,
            status=f"no_safe_planar_horizon_after_{failure_status}",
            solver_iteration_count=solver_iteration_count,
        )

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

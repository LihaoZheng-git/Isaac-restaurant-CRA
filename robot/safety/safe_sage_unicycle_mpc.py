"""Sparse predictive Safe-SAGE safety filter for a unicycle robot.

Safe-SAGE uses a finite-horizon MPC safety filter followed by a faster
analytical CBF layer.  Nova Carter executes ``(linear_velocity,
angular_velocity)`` rather than an arbitrary planar velocity, so this module
rolls out the physical unicycle state ``(x, y, yaw)`` and evaluates safety at
the standard forward look-ahead point.

Each sequential quadratic programming (SQP) subproblem is assembled from
analytical unicycle rollout sensitivities and solved as a sparse QP with
OSQP.  Safe-SAGE semantic guidance supplies the positional derivatives in
the predictive phase.  If the resulting trajectory does not satisfy the
sampled nonlinear DTCBF exactly, a second sparse SQP phase uses the sampled
gradient of ``h_full`` to restore feasibility.  No finite-difference
optimization is used.

Each local QP includes nonnegative elastic DTCBF variables with a strong
linear penalty.  They prevent an infeasible linearization from terminating
SQP prematurely; they do not relax the exact nonlinear DTCBF acceptance
check applied to every returned trajectory.  If OSQP reaches its iteration
limit, a bounded SLSQP rescue uses analytical unicycle sensitivities and
sampled ``h_full`` gradients to restore the same exact nonlinear constraints.
If both numerical solvers fail, a deterministic last-resort recovery reduces
translation while preserving the planned turn, then verifies a stationary
horizon.  A recovery is returned only when the exact DTCBF check passes.
Final acceptance also rejects every predicted configuration-space cell marked
unsafe; interpolated positive ``h_full`` values cannot override that mask.

Optional integration substeps match the curved constant-command rollout to a
finer execution time step, including analytical translation/turn Jacobians.
With more than one integration substep, inter-sample geometry checks use that
same curved rollout. The default remains one Euler step per MPC interval.

An opt-in bounded turn-seed search addresses nearly stopped local solutions.
It retains the nominal tracking objective and accepts only a cheaper horizon
that passes the same exact DTCBF and dense geometry checks. It is an online
nonholonomic adaptation, not an obstacle-aware nominal planner or a claim of
global convergence. Legacy callers leave this search disabled by default.

The module has no Isaac Sim dependency.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from numbers import Real

import numpy as np
import osqp
from scipy import sparse
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
from robot.safety.safe_sage_temporal_derivative import (
    TemporalSafetyDerivativeInput,
    validate_temporal_safety_derivative,
)


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


def _optional_direction(value: object, *, name: str) -> float | None:
    """Return a normalized optional signed direction."""

    if value is None:
        return None
    direction = _finite_real(value, name=name)
    if direction == 0.0:
        raise ValueError(f"{name} must be nonzero when supplied.")
    return math.copysign(1.0, direction)


def _frozen_array(value: np.ndarray) -> np.ndarray:
    """Return an owned, contiguous, read-only float64 array."""

    result = np.array(value, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class SafeSAGEUnicycleMPCConfig:
    """Prediction, actuator, cost, sparse SQP, and OSQP settings."""

    horizon_steps: int = 12
    safety_substeps_per_interval: int = 1
    integration_substeps_per_interval: int = 1
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
    position_tracking_weight: float = 2.0
    yaw_tracking_weight: float = 0.10
    maximum_sqp_iterations: int = 6
    maximum_restoration_iterations: int = 12
    maximum_nonlinear_rescue_iterations: int = 300
    maximum_solver_iterations: int = 4000
    linear_velocity_trust_region: float = 0.10
    angular_velocity_trust_region: float = 0.40
    cbf_slack_weight: float = 100.0
    solver_tolerance: float = 1e-6
    trajectory_tolerance: float = 1e-4
    constraint_tolerance: float = 1e-6
    intervention_tolerance: float = 1e-5
    line_search_contraction: float = 0.5
    maximum_line_search_iterations: int = 8
    enable_progress_seeds: bool = False
    enable_recovery_cost_pruning: bool = False
    enable_recovery_feasibility_pruning: bool = False
    progress_seed_linear_fraction: float = 0.25
    configuration_position_padding_m: float = 0.0
    configuration_yaw_padding_radians: float = 0.0
    configuration_padding_initial_fraction: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.enable_progress_seeds, bool):
            raise TypeError("enable_progress_seeds must be a boolean.")
        if not isinstance(self.enable_recovery_cost_pruning, bool):
            raise TypeError("enable_recovery_cost_pruning must be a boolean.")
        if not isinstance(self.enable_recovery_feasibility_pruning, bool):
            raise TypeError("enable_recovery_feasibility_pruning must be a boolean.")
        integer_values = {
            "horizon_steps": _positive_integer(
                self.horizon_steps,
                name="horizon_steps",
            ),
            "safety_substeps_per_interval": _positive_integer(
                self.safety_substeps_per_interval,
                name="safety_substeps_per_interval",
            ),
            "integration_substeps_per_interval": _positive_integer(
                self.integration_substeps_per_interval,
                name="integration_substeps_per_interval",
            ),
            "maximum_sqp_iterations": _positive_integer(
                self.maximum_sqp_iterations,
                name="maximum_sqp_iterations",
            ),
            "maximum_restoration_iterations": _positive_integer(
                self.maximum_restoration_iterations,
                name="maximum_restoration_iterations",
            ),
            "maximum_nonlinear_rescue_iterations": _positive_integer(
                self.maximum_nonlinear_rescue_iterations,
                name="maximum_nonlinear_rescue_iterations",
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
        integration_count = integer_values["integration_substeps_per_interval"]
        safety_count = integer_values["safety_substeps_per_interval"]
        if integration_count > 1 and integration_count % safety_count:
            raise ValueError(
                "integration substeps must be a multiple of safety substeps"
            )
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
                (
                    "position_tracking_weight",
                    self.position_tracking_weight,
                ),
                ("yaw_tracking_weight", self.yaw_tracking_weight),
                (
                    "linear_velocity_trust_region",
                    self.linear_velocity_trust_region,
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
                ("progress_seed_linear_fraction", self.progress_seed_linear_fraction),
                ("configuration_position_padding_m", self.configuration_position_padding_m),
                ("configuration_yaw_padding_radians", self.configuration_yaw_padding_radians),
                ("configuration_padding_initial_fraction", self.configuration_padding_initial_fraction),
            )
        }

        if real_values["time_step"] <= 0.0:
            raise ValueError("time_step must be greater than zero.")
        if real_values["gamma"] <= 0.0:
            raise ValueError("gamma must be greater than zero.")
        if real_values["lookahead_distance"] <= 0.0:
            raise ValueError("lookahead_distance must be greater than zero.")
        if (
            real_values["minimum_linear_velocity"]
            > real_values["maximum_linear_velocity"]
        ):
            raise ValueError(
                "minimum_linear_velocity cannot exceed maximum_linear_velocity."
            )
        if (
            real_values["minimum_angular_velocity"]
            > real_values["maximum_angular_velocity"]
        ):
            raise ValueError(
                "minimum_angular_velocity cannot exceed maximum_angular_velocity."
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
            "linear_velocity_trust_region",
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
            "position_tracking_weight",
            "yaw_tracking_weight",
        ):
            if real_values[name] < 0.0:
                raise ValueError(f"{name} cannot be negative.")
        if real_values["intervention_tolerance"] < 0.0:
            raise ValueError("intervention_tolerance cannot be negative.")
        if not 0.0 < real_values["line_search_contraction"] < 1.0:
            raise ValueError(
                "line_search_contraction must lie strictly between zero and one."
            )
        if not 0.0 < real_values["progress_seed_linear_fraction"] < 1.0:
            raise ValueError("progress_seed_linear_fraction must lie in (0, 1).")
        if min(real_values["configuration_position_padding_m"],
               real_values["configuration_yaw_padding_radians"]) < 0.0:
            raise ValueError("configuration padding must be nonnegative.")
        if not 0.0 <= real_values["configuration_padding_initial_fraction"] <= 1.0:
            raise ValueError("configuration_padding_initial_fraction must lie in [0, 1].")

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
    restoration_used: bool
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
    guidance_normals: np.ndarray
    exact_gradients: np.ndarray
    scaled_temporal_derivatives: np.ndarray
    unsafe_configuration_mask: np.ndarray


@dataclass(frozen=True)
class _TrajectoryEvaluation:
    """One command rollout and its sampled nonlinear safety diagnostics."""

    commands: np.ndarray
    chassis_states: np.ndarray
    safety_states: np.ndarray
    samples: _ReferenceSamples
    safety_values: np.ndarray
    margins: np.ndarray


@dataclass(frozen=True)
class _SparseQPSolution:
    """One sparse SQP subproblem result."""

    success: bool
    commands: np.ndarray
    iteration_count: int
    status: str
    maximum_cbf_slack: float


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

    def revalidate_commands(
        self,
        initial_chassis_state_grid,
        commands,
        *,
        configuration_space,
        h_full,
        guidance_fields,
        temporal_safety_derivative=0.0,
    ) -> bool:
        """Check an asynchronous horizon at the current state/field, no solve.

        The exact nonlinear DTCBF, actuator bounds and dense configuration
        checks are identical to plan acceptance. A field generation change or
        delayed planner result never grants permission to execute an old plan.
        """
        initial = _finite_vector(
            initial_chassis_state_grid, size=3, name="initial_chassis_state_grid"
        )
        sequence = np.asarray(commands, dtype=float)
        if (
            sequence.shape != (self.config.horizon_steps, 2)
            or not np.isfinite(sequence).all()
        ):
            raise ValueError("commands must be a finite complete horizon")
        if not np.array_equal(sequence, self._clip_commands(sequence)):
            return False
        temporal = validate_temporal_safety_derivative(
            temporal_safety_derivative,
            h_full=h_full,
            grid_config=configuration_space.occupancy_grids[0].config,
        )
        evaluated = self._evaluate_trajectory(
            initial,
            sequence,
            configuration_space=configuration_space,
            h_full=h_full,
            guidance_fields=guidance_fields,
            temporal_safety_derivative=temporal,
        )
        return evaluated is not None and self._trajectory_is_safe(evaluated)

    def plan(
        self,
        initial_chassis_state_grid: Sequence[float] | np.ndarray,
        nominal_command: Sequence[float] | np.ndarray,
        *,
        initial_command_guess: Sequence[float] | np.ndarray | None = None,
        target_chassis_states_grid: (
            Sequence[Sequence[float]] | np.ndarray | None
        ) = None,
        preferred_lateral_direction_grid_y: float | None = None,
        configuration_space: SafeSAGEConfigurationSpaceResult,
        h_full: SafeSAGEHFullResult,
        guidance_fields: Sequence[LaplaceGuidanceFieldResult],
        temporal_safety_derivative: TemporalSafetyDerivativeInput = 0.0,
    ) -> SafeSAGEUnicycleMPCResult:
        """Optimize a safe command horizon and return its first command.

        ``initial_chassis_state_grid`` is ``(x, y, yaw)`` in the fixed local
        field frame. ``nominal_command`` may be one ``(v, omega)`` command or
        an explicit ``(horizon_steps, 2)`` reference sequence.
        ``initial_command_guess`` optionally supplies a distinct warm-start
        sequence without changing the command-tracking objective.
        ``target_chassis_states_grid`` optionally supplies an
        ``(horizon_steps + 1, 3)`` position-and-yaw path. When supplied, the
        MPC tracks that path in addition to the nominal command sequence.
        ``preferred_lateral_direction_grid_y`` optionally preserves a
        semantic passing-side convention during last-resort numerical
        recovery. It does not add a cost or constraint to ordinary MPC.
        """

        initial_state = _finite_vector(
            initial_chassis_state_grid,
            size=3,
            name="initial_chassis_state_grid",
        )
        temporal_derivative = validate_temporal_safety_derivative(
            temporal_safety_derivative,
            h_full=h_full,
            grid_config=configuration_space.occupancy_grids[0].config,
        )
        preferred_lateral_direction = _optional_direction(
            preferred_lateral_direction_grid_y,
            name="preferred_lateral_direction_grid_y",
        )
        nominal_sequence = self._nominal_sequence(nominal_command)
        target_states = self._target_state_sequence(
            target_chassis_states_grid
        )
        state_tracking_active = bool(
            target_states is not None
            and (
                self.config.position_tracking_weight > 0.0
                or self.config.yaw_tracking_weight > 0.0
            )
        )
        if initial_command_guess is None:
            reference_commands = nominal_sequence
        else:
            reference_commands = self._command_sequence(
                initial_command_guess,
                name="initial_command_guess",
            )
        reference_commands = self._clip_commands(reference_commands)
        reference = self._evaluate_trajectory(
            initial_state,
            reference_commands,
            configuration_space=configuration_space,
            h_full=h_full,
            guidance_fields=guidance_fields,
            temporal_safety_derivative=temporal_derivative,
        )
        if reference is None:
            return self._fallback_result(
                initial_state,
                nominal_sequence,
                status="invalid_initial_reference",
            )

        current_constraint = self.paper_filter.evaluate_constraint(
            query_position_grid_xy=reference.safety_states[0, :2],
            query_yaw_radians=float(reference.safety_states[0, 2]),
            configuration_space=configuration_space,
            h_full=h_full,
            guidance_fields=guidance_fields,
            temporal_safety_derivative=temporal_derivative,
        )
        if current_constraint.unsafe_reason is not None:
            return self._fallback_result(
                initial_state,
                nominal_sequence,
                status=current_constraint.unsafe_reason,
            )

        seed_restoration_used = False
        if self.config.enable_progress_seeds and not self._trajectory_is_safe(reference):
            seed = self._select_progress_seed(
                reference, initial_state=initial_state,
                nominal_sequence=nominal_sequence, target_chassis_states=target_states,
                configuration_space=configuration_space, h_full=h_full,
                guidance_fields=guidance_fields,
                temporal_safety_derivative=temporal_derivative,
            )
            if seed is not reference:
                reference = seed
                seed_restoration_used = True

        total_qp_iterations = 0
        completed_sqp_iterations = 0
        guidance_failure: str | None = None

        if (
            initial_command_guess is not None
            and self._trajectory_is_safe(reference)
            and not state_tracking_active
        ):
            nominal_reference = self._evaluate_trajectory(
                initial_state,
                self._clip_commands(nominal_sequence),
                configuration_space=configuration_space,
                h_full=h_full,
                guidance_fields=guidance_fields,
                temporal_safety_derivative=temporal_derivative,
            )
            if nominal_reference is not None and self._trajectory_is_safe(
                nominal_reference
            ):
                reference = nominal_reference
            else:
                reference, improvement_steps, improvement_iterations = (
                    self._improve_safe_warm_start(
                        reference,
                        initial_state=initial_state,
                        nominal_sequence=nominal_sequence,
                        target_chassis_states=target_states,
                        configuration_space=configuration_space,
                        h_full=h_full,
                        guidance_fields=guidance_fields,
                        temporal_safety_derivative=temporal_derivative,
                    )
                )
                completed_sqp_iterations += improvement_steps
                total_qp_iterations += improvement_iterations

        if state_tracking_active and self._trajectory_is_safe(reference):
            reference, improvement_steps, improvement_iterations = (
                self._improve_safe_warm_start(
                    reference,
                    initial_state=initial_state,
                    nominal_sequence=nominal_sequence,
                    target_chassis_states=target_states,
                    configuration_space=configuration_space,
                    h_full=h_full,
                    guidance_fields=guidance_fields,
                    temporal_safety_derivative=temporal_derivative,
                )
            )
            completed_sqp_iterations += improvement_steps
            total_qp_iterations += improvement_iterations

        if not self._trajectory_is_safe(reference):
            for sqp_index in range(self.config.maximum_sqp_iterations):
                completed_sqp_iterations = sqp_index + 1
                solution = self._solve_sparse_subproblem(
                    reference,
                    nominal_sequence=nominal_sequence,
                    target_chassis_states=target_states,
                    configuration_space=configuration_space,
                    derivative_mode="guidance",
                )
                total_qp_iterations += solution.iteration_count
                if not solution.success:
                    guidance_failure = solution.status
                    break

                candidate = self._evaluate_trajectory(
                    initial_state,
                    solution.commands,
                    configuration_space=configuration_space,
                    h_full=h_full,
                    guidance_fields=guidance_fields,
                    temporal_safety_derivative=temporal_derivative,
                )
                if candidate is None:
                    guidance_failure = "invalid_guidance_sqp_trajectory"
                    break
                command_change = float(
                    np.max(np.abs(candidate.commands - reference.commands))
                )
                reference = candidate
                if self._trajectory_is_safe(reference):
                    break
                if command_change <= self.config.trajectory_tolerance:
                    guidance_failure = "guidance_sqp_stalled"
                    break

        restoration_used = not self._trajectory_is_safe(reference)
        nonlinear_rescue_used = False
        feasibility_recovery_status: str | None = (
            "exact_safe_seed_feasibility_recovered" if seed_restoration_used else None
        )
        if restoration_used:
            restoration_success = False
            restoration_failure: str | None = None
            for _ in range(self.config.maximum_restoration_iterations):
                completed_sqp_iterations += 1
                solution = self._solve_sparse_subproblem(
                    reference,
                    nominal_sequence=nominal_sequence,
                    target_chassis_states=target_states,
                    configuration_space=configuration_space,
                    derivative_mode="gradient",
                )
                total_qp_iterations += solution.iteration_count
                if not solution.success:
                    restoration_failure = solution.status
                    if guidance_failure is not None:
                        restoration_failure = (
                            f"{guidance_failure}; {restoration_failure}"
                        )
                    break

                candidate = self._evaluate_trajectory(
                    initial_state,
                    solution.commands,
                    configuration_space=configuration_space,
                    h_full=h_full,
                    guidance_fields=guidance_fields,
                    temporal_safety_derivative=temporal_derivative,
                )
                candidate = self._select_restoration_step(
                    reference,
                    candidate,
                    initial_state=initial_state,
                    proposed_commands=solution.commands,
                    configuration_space=configuration_space,
                    h_full=h_full,
                    guidance_fields=guidance_fields,
                    temporal_safety_derivative=temporal_derivative,
                )
                if candidate is None:
                    restoration_failure = "exact_qp_restoration_did_not_improve"
                    break
                reference = candidate
                if self._trajectory_is_safe(reference):
                    restoration_success = True
                    break

            if not restoration_success:
                if restoration_failure is None:
                    restoration_failure = (
                        "exact_qp_restoration_exhausted; "
                        f"minimum_margin={float(np.min(reference.margins)):.3e}"
                    )
                rescue, rescue_iterations, rescue_status = (
                    self._solve_exact_nonlinear_rescue(
                        reference,
                        initial_state=initial_state,
                        nominal_sequence=nominal_sequence,
                        target_chassis_states=target_states,
                        configuration_space=configuration_space,
                        h_full=h_full,
                        guidance_fields=guidance_fields,
                        temporal_safety_derivative=temporal_derivative,
                    )
                )
                total_qp_iterations += rescue_iterations
                if rescue is None:
                    recovery, recovery_status = self._recover_feasible_command_horizon(
                        reference,
                        initial_state=initial_state,
                        nominal_sequence=nominal_sequence,
                        configuration_space=configuration_space,
                        h_full=h_full,
                        guidance_fields=guidance_fields,
                        temporal_safety_derivative=temporal_derivative,
                        preferred_lateral_direction_grid_y=(
                            preferred_lateral_direction
                        ),
                    )
                    if recovery is None:
                        return self._fallback_result(
                            initial_state,
                            nominal_sequence,
                            status=(
                                "exact_nonlinear_rescue_failed: "
                                f"{restoration_failure}; {rescue_status}"
                            ),
                            sqp_iterations=completed_sqp_iterations,
                            solver_iterations=total_qp_iterations,
                        )
                    reference = recovery
                    feasibility_recovery_status = recovery_status
                    (
                        reference,
                        improvement_steps,
                        improvement_iterations,
                    ) = self._improve_safe_warm_start(
                        reference,
                        initial_state=initial_state,
                        nominal_sequence=nominal_sequence,
                        target_chassis_states=target_states,
                        configuration_space=configuration_space,
                        h_full=h_full,
                        guidance_fields=guidance_fields,
                        temporal_safety_derivative=temporal_derivative,
                    )
                    completed_sqp_iterations += improvement_steps
                    total_qp_iterations += improvement_iterations
                else:
                    reference = rescue
                    nonlinear_rescue_used = True

        if (
            self.config.enable_progress_seeds
            and nominal_sequence[0, 0] > self.config.intervention_tolerance
            and reference.commands[0, 0]
            < self.config.progress_seed_linear_fraction * nominal_sequence[0, 0]
        ):
            improved = self._select_progress_seed(
                reference,
                initial_state=initial_state,
                nominal_sequence=nominal_sequence,
                target_chassis_states=target_states,
                configuration_space=configuration_space,
                h_full=h_full,
                guidance_fields=guidance_fields,
                temporal_safety_derivative=temporal_derivative,
            )
            if improved is not reference:
                reference = improved
                feasibility_recovery_status = "exact_safe_progress_seed_selected"

        intervention = bool(
            np.any(
                np.abs(reference.commands[0] - nominal_sequence[0])
                > self.config.intervention_tolerance
            )
        )
        return SafeSAGEUnicycleMPCResult(
            command_sequence=_frozen_array(reference.commands),
            predicted_chassis_states=_frozen_array(reference.chassis_states),
            predicted_safety_states=_frozen_array(reference.safety_states),
            predicted_safety_values=_frozen_array(reference.safety_values),
            dtcbf_margins=_frozen_array(reference.margins),
            nominal_first_command=(
                float(nominal_sequence[0, 0]),
                float(nominal_sequence[0, 1]),
            ),
            objective_value=self._objective_value(
                reference.commands,
                nominal_sequence,
                chassis_states=reference.chassis_states,
                target_chassis_states=target_states,
            ),
            sqp_iteration_count=completed_sqp_iterations,
            solver_iteration_count=total_qp_iterations,
            success=True,
            intervention_active=intervention,
            fallback_active=False,
            restoration_used=restoration_used or seed_restoration_used,
            status=(
                feasibility_recovery_status
                or (
                    "exact_nonlinear_feasibility_restored"
                    if nonlinear_rescue_used
                    else (
                        "exact_dtcbf_feasibility_restored"
                        if restoration_used
                        else "converged"
                    )
                )
            ),
        )

    def _select_progress_seed(
        self,
        reference: _TrajectoryEvaluation,
        *,
        initial_state: np.ndarray,
        nominal_sequence: np.ndarray,
        target_chassis_states: np.ndarray | None,
        configuration_space: SafeSAGEConfigurationSpaceResult,
        h_full: SafeSAGEHFullResult,
        guidance_fields: Sequence[LaplaceGuidanceFieldResult],
        temporal_safety_derivative: TemporalSafetyDerivativeInput,
    ) -> _TrajectoryEvaluation:
        """Escape a stopped local solution with bounded online turn seeds.

        A nonholonomic robot may need to turn before translating. Local SQP
        about a stopped horizon cannot discover that coordinated maneuver.
        Try both turn directions, several durations and reduced translation,
        then retain only a strictly cheaper, exact-safe horizon. The nominal
        objective, actuator limits, DTCBF and dense mask checks are unchanged.
        No obstacle pose or authored bypass waypoint enters this search.
        """

        best = reference
        best_cost = self._objective_value(
            best.commands,
            nominal_sequence,
            chassis_states=best.chassis_states,
            target_chassis_states=target_chassis_states,
        )
        if not self._trajectory_is_safe(reference):
            # An infeasible nominal/warm start has no admissible objective
            # value. First find a feasible horizon, then optimize its cost.
            best_cost = math.inf
        durations = sorted(
            {
                min(self.config.horizon_steps, max(1, round(t / self.config.time_step)))
                for t in (0.4, 0.8, 1.2, 1.6, 2.4, 3.2)
            }
        )
        direction = math.copysign(
            1.0, self._initial_guidance_angular_coefficient(reference)
        )
        seeds = []
        for linear_fraction in (0.5, 0.75, 1.0):
            straight = nominal_sequence.copy()
            straight[:, 0] *= linear_fraction
            straight[:, 1] = 0.0
            seeds.append(straight)
        for sign in (direction, -direction):
            angular_limit = (
                self.config.maximum_angular_velocity
                if sign > 0 else abs(self.config.minimum_angular_velocity)
            )
            if angular_limit == 0:
                continue
            for angular_fraction in (0.125, 0.5, 1.0):
                for count in durations:
                    for linear_fraction in (0.5, 0.75, 1.0):
                        for turning_fraction in (0.0, 0.25):
                            for hold_heading in (False, True):
                                commands = nominal_sequence.copy()
                                commands[:, 0] *= linear_fraction
                                if hold_heading:
                                    commands[:, 1] = 0.0
                                commands[:count, 0] *= turning_fraction
                                commands[:count, 1] = sign * angular_fraction * angular_limit
                                seeds.append(commands)
        # Command cost is a lower bound on the optional nonnegative path cost.
        # Check cheapest seeds first and stop once none can improve the result.
        ranked = []
        for commands in seeds:
            commands = self._clip_commands(commands)
            lower_cost = self._objective_value(commands, nominal_sequence)
            ranked.append((lower_cost, commands))
        for lower_cost, commands in sorted(ranked, key=lambda item: item[0]):
            if lower_cost >= best_cost - self.config.trajectory_tolerance:
                break
            candidate = self._evaluate_trajectory(
                initial_state, commands,
                configuration_space=configuration_space,
                h_full=h_full,
                guidance_fields=guidance_fields,
                temporal_safety_derivative=temporal_safety_derivative,
                require_feasible=self.config.enable_recovery_feasibility_pruning,
            )
            if candidate is None or not self._trajectory_is_safe(candidate):
                continue
            cost = self._objective_value(
                candidate.commands, nominal_sequence,
                chassis_states=candidate.chassis_states,
                target_chassis_states=target_chassis_states,
            )
            if cost < best_cost - self.config.trajectory_tolerance:
                best, best_cost = candidate, cost
        return best

    def _improve_safe_warm_start(
        self,
        reference: _TrajectoryEvaluation,
        *,
        initial_state: np.ndarray,
        nominal_sequence: np.ndarray,
        target_chassis_states: np.ndarray | None = None,
        configuration_space: SafeSAGEConfigurationSpaceResult,
        h_full: SafeSAGEHFullResult,
        guidance_fields: Sequence[LaplaceGuidanceFieldResult],
        temporal_safety_derivative: TemporalSafetyDerivativeInput,
    ) -> tuple[_TrajectoryEvaluation, int, int]:
        """Move a feasible warm start toward the nominal command sequence.

        A warm start is a numerical seed, not the MPC solution.  Each sparse
        guidance SQP proposal is accepted only when an exact-DTCBF-safe
        backtracking step also improves the nominal tracking objective.
        """

        current = reference
        completed_steps = 0
        solver_iterations = 0
        for _ in range(self.config.maximum_sqp_iterations):
            solution = self._solve_sparse_subproblem(
                current,
                nominal_sequence=nominal_sequence,
                target_chassis_states=target_chassis_states,
                configuration_space=configuration_space,
                derivative_mode="guidance",
            )
            completed_steps += 1
            solver_iterations += solution.iteration_count
            if not solution.success:
                break

            candidate = self._evaluate_trajectory(
                initial_state,
                solution.commands,
                configuration_space=configuration_space,
                h_full=h_full,
                guidance_fields=guidance_fields,
                temporal_safety_derivative=temporal_safety_derivative,
            )
            improved = self._select_safe_objective_step(
                current,
                candidate,
                initial_state=initial_state,
                proposed_commands=solution.commands,
                nominal_sequence=nominal_sequence,
                target_chassis_states=target_chassis_states,
                configuration_space=configuration_space,
                h_full=h_full,
                guidance_fields=guidance_fields,
                temporal_safety_derivative=temporal_safety_derivative,
            )
            if improved is None:
                break
            command_change = float(
                np.max(np.abs(improved.commands - current.commands))
            )
            current = improved
            if command_change <= self.config.trajectory_tolerance:
                break
        return current, completed_steps, solver_iterations

    def _select_safe_objective_step(
        self,
        reference: _TrajectoryEvaluation,
        candidate: _TrajectoryEvaluation | None,
        *,
        initial_state: np.ndarray,
        proposed_commands: np.ndarray,
        nominal_sequence: np.ndarray,
        target_chassis_states: np.ndarray | None = None,
        configuration_space: SafeSAGEConfigurationSpaceResult,
        h_full: SafeSAGEHFullResult,
        guidance_fields: Sequence[LaplaceGuidanceFieldResult],
        temporal_safety_derivative: TemporalSafetyDerivativeInput,
    ) -> _TrajectoryEvaluation | None:
        """Backtrack to an exact-safe step with lower nominal objective."""

        reference_objective = self._objective_value(
            reference.commands,
            nominal_sequence,
            chassis_states=reference.chassis_states,
            target_chassis_states=target_chassis_states,
        )

        def is_acceptable(trial: _TrajectoryEvaluation | None) -> bool:
            return bool(
                trial is not None
                and self._trajectory_is_safe(trial)
                and self._objective_value(
                    trial.commands,
                    nominal_sequence,
                    chassis_states=trial.chassis_states,
                    target_chassis_states=target_chassis_states,
                )
                < reference_objective - 1e-12
            )

        if is_acceptable(candidate):
            return candidate

        step_size = 1.0
        for _ in range(self.config.maximum_line_search_iterations):
            step_size *= self.config.line_search_contraction
            trial_commands = reference.commands + step_size * (
                proposed_commands - reference.commands
            )
            trial = self._evaluate_trajectory(
                initial_state,
                trial_commands,
                configuration_space=configuration_space,
                h_full=h_full,
                guidance_fields=guidance_fields,
                temporal_safety_derivative=temporal_safety_derivative,
            )
            if is_acceptable(trial):
                return trial
        return None

    def _recover_feasible_command_horizon(
        self,
        reference: _TrajectoryEvaluation,
        *,
        initial_state: np.ndarray,
        nominal_sequence: np.ndarray,
        configuration_space: SafeSAGEConfigurationSpaceResult,
        h_full: SafeSAGEHFullResult,
        guidance_fields: Sequence[LaplaceGuidanceFieldResult],
        temporal_safety_derivative: TemporalSafetyDerivativeInput,
        preferred_lateral_direction_grid_y: float | None = None,
    ) -> tuple[_TrajectoryEvaluation | None, str | None]:
        """Recover exact feasibility without discarding Safe-SAGE direction.

        This deterministic last-resort path is used only after both sparse
        SQP and the exact nonlinear rescue fail. A failed numerical iterate is
        not a trustworthy motion reference, so guidance-steered candidates are
        constructed from the obstacle-unaware nominal command sequence. The
        exact-safe candidate with the smallest nominal tracking objective is
        selected. Only if no such candidate exists do we reduce translation on
        the failed iterate and finally try a full stop.
        Every returned candidate passes exact DTCBF evaluation.
        """

        preferred_lateral_direction = _optional_direction(
            preferred_lateral_direction_grid_y,
            name="preferred_lateral_direction_grid_y",
        )
        initial_lateral_position = float(reference.chassis_states[0, 1])

        def has_preferred_lateral_progress(
            candidate: _TrajectoryEvaluation,
        ) -> bool:
            if preferred_lateral_direction is None:
                return True
            lateral_change = (
                float(candidate.chassis_states[-1, 1])
                - initial_lateral_position
            )
            return preferred_lateral_direction * lateral_change > 1e-3

        angular_coefficient = self._initial_guidance_angular_coefficient(
            reference
        )
        direction_tolerance = 1e-10
        if abs(angular_coefficient) > direction_tolerance:
            local_steering_direction = math.copysign(
                1.0,
                angular_coefficient,
            )
            steering_directions = (
                local_steering_direction,
                -local_steering_direction,
            )
            best_candidate: _TrajectoryEvaluation | None = None
            best_objective = math.inf
            maximum_steering_steps = min(
                self.config.horizon_steps, 15
            )
            translation_profiles = (
                (1.0, 1.0),
                (1.0, 0.75),
                (1.0, 0.5),
                (1.0, 0.25),
                (1.0, 0.0),
                (0.75, 0.75),
                (0.75, 0.5),
                (0.75, 0.25),
                (0.50, 0.50),
                (0.50, 0.25),
                (0.25, 0.25),
            )
            for steering_direction in steering_directions:
                angular_limit = (
                    self.config.maximum_angular_velocity
                    if steering_direction > 0.0
                    else abs(self.config.minimum_angular_velocity)
                )
                if angular_limit <= 0.0:
                    continue
                for steering_step_count in range(
                    1, maximum_steering_steps + 1
                ):
                    for angular_fraction in (1.0, 0.75, 0.5, 0.25):
                        steering_rate = (
                            steering_direction
                            * angular_fraction
                            * angular_limit
                        )
                        for (
                            linear_scale,
                            steering_linear_fraction,
                        ) in translation_profiles:
                            commands = np.array(
                                nominal_sequence,
                                dtype=np.float64,
                                copy=True,
                            )
                            commands[:, 0] *= linear_scale
                            commands[:steering_step_count, 0] = (
                                nominal_sequence[:steering_step_count, 0]
                                * steering_linear_fraction
                            )
                            commands[:steering_step_count, 1] = steering_rate
                            if self.config.enable_recovery_cost_pruning:
                                commands = self._clip_commands(commands)
                                # This recovery pool has no state-path cost.
                                # Its clipped command cost is exact; a tie would
                                # retain the earlier candidate in the old loop.
                                if self._objective_value(commands, nominal_sequence) >= best_objective:
                                    continue
                            candidate = self._evaluate_trajectory(
                                initial_state,
                                commands,
                                require_feasible=self.config.enable_recovery_feasibility_pruning,
                                configuration_space=configuration_space,
                                h_full=h_full,
                                guidance_fields=guidance_fields,
                                temporal_safety_derivative=(
                                    temporal_safety_derivative
                                ),
                            )
                            if (
                                candidate is None
                                or not self._trajectory_is_safe(candidate)
                                or not has_preferred_lateral_progress(candidate)
                            ):
                                continue
                            objective = self._objective_value(
                                candidate.commands,
                                nominal_sequence,
                                chassis_states=candidate.chassis_states,
                                target_chassis_states=None,
                            )
                            if objective < best_objective:
                                best_candidate = candidate
                                best_objective = objective

            if best_candidate is not None:
                return (
                    best_candidate,
                    "nominal_guidance_seed_feasibility_recovered",
                )

            # A pure rotation is safe in many scenes but is not a bypass
            # plan. Consider it only after moving candidates in both angular
            # directions fail. Its direction remains the local constraint's
            # best instantaneous correction.
            steering_step_count = min(
                self.config.horizon_steps,
                max(1, math.ceil(0.8 / self.config.time_step)),
            )
            steering_direction = local_steering_direction
            angular_limit = (
                self.config.maximum_angular_velocity
                if steering_direction > 0.0
                else abs(self.config.minimum_angular_velocity)
            )
            for angular_fraction in (1.0, 0.75, 0.5, 0.25):
                commands = np.zeros_like(reference.commands)
                commands[:steering_step_count, 1] = (
                    steering_direction * angular_fraction * angular_limit
                )
                candidate = self._evaluate_trajectory(
                    initial_state,
                    commands,
                    configuration_space=configuration_space,
                    h_full=h_full,
                    guidance_fields=guidance_fields,
                    temporal_safety_derivative=temporal_safety_derivative,
                )
                if candidate is not None and self._trajectory_is_safe(
                    candidate
                ):
                    return (
                        candidate,
                        "guidance_rotation_feasibility_recovered",
                    )

        for linear_scale in (0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125, 0.0):
            commands = np.array(reference.commands, dtype=np.float64, copy=True)
            commands[:, 0] *= linear_scale
            candidate = self._evaluate_trajectory(
                initial_state,
                commands,
                configuration_space=configuration_space,
                h_full=h_full,
                guidance_fields=guidance_fields,
                temporal_safety_derivative=temporal_safety_derivative,
            )
            if candidate is not None and self._trajectory_is_safe(candidate):
                return candidate, "translation_scaled_feasibility_recovered"

        stationary_commands = np.zeros_like(reference.commands)
        stationary = self._evaluate_trajectory(
            initial_state,
            stationary_commands,
            configuration_space=configuration_space,
            h_full=h_full,
            guidance_fields=guidance_fields,
            temporal_safety_derivative=temporal_safety_derivative,
        )
        if stationary is not None and self._trajectory_is_safe(stationary):
            return stationary, "stationary_feasibility_recovered"
        return None, None

    def _initial_guidance_angular_coefficient(
        self,
        reference: _TrajectoryEvaluation,
    ) -> float:
        """Return the current Safe-SAGE coefficient multiplying ``omega``.

        The look-ahead point velocity is
        ``(v cos(yaw) - ell omega sin(yaw),
        v sin(yaw) + ell omega cos(yaw))``. Pulling the sampled paper-space
        constraint normal back through that interface gives the coefficient
        below. Its sign is therefore the angular direction that increases
        the Safe-SAGE constraint margin; it is not a demo-specific side flag.
        """

        yaw = float(reference.chassis_states[0, 2])
        normal = reference.samples.guidance_normals[0]
        distance = self.config.lookahead_distance
        return float(
            -distance * math.sin(yaw) * normal[0]
            + distance * math.cos(yaw) * normal[1]
            + normal[2]
        )

    def _solve_exact_nonlinear_rescue(
        self,
        reference: _TrajectoryEvaluation,
        *,
        initial_state: np.ndarray,
        nominal_sequence: np.ndarray,
        target_chassis_states: np.ndarray | None = None,
        configuration_space: SafeSAGEConfigurationSpaceResult,
        h_full: SafeSAGEHFullResult,
        guidance_fields: Sequence[LaplaceGuidanceFieldResult],
        temporal_safety_derivative: TemporalSafetyDerivativeInput,
    ) -> tuple[_TrajectoryEvaluation | None, int, str]:
        """Rescue a stalled QP sequence using exact nonlinear DTCBF values.

        The objective gradient, unicycle rollout sensitivities, look-ahead
        mapping, and sampled ``h_full`` gradients are supplied analytically.
        This stage runs only after sparse SQP cannot restore exact feasibility.
        """

        horizon = self.config.horizon_steps
        variable_count = 2 * horizon
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
        invalid_constraints = np.full(5 * horizon, -1.0, dtype=np.float64)
        cached_commands: np.ndarray | None = None
        cached_evaluation: _TrajectoryEvaluation | None = None

        def evaluate(flattened_commands: np.ndarray) -> _TrajectoryEvaluation | None:
            nonlocal cached_commands, cached_evaluation
            commands = np.asarray(flattened_commands, dtype=np.float64).reshape(
                horizon,
                2,
            )
            if cached_commands is not None and np.array_equal(
                commands, cached_commands
            ):
                return cached_evaluation
            cached_commands = commands.copy()
            cached_evaluation = self._evaluate_trajectory(
                initial_state,
                commands,
                configuration_space=configuration_space,
                h_full=h_full,
                guidance_fields=guidance_fields,
                temporal_safety_derivative=temporal_safety_derivative,
                check_dense_geometry=False,
            )
            return cached_evaluation

        def objective(flattened_commands: np.ndarray) -> float:
            commands = np.asarray(flattened_commands, dtype=np.float64).reshape(
                horizon,
                2,
            )
            chassis_states = self._rollout(initial_state, commands)
            return self._objective_value(
                commands,
                nominal_sequence,
                chassis_states=chassis_states,
                target_chassis_states=target_chassis_states,
            )

        def objective_jacobian(flattened_commands: np.ndarray) -> np.ndarray:
            commands = np.asarray(flattened_commands, dtype=np.float64).reshape(
                horizon,
                2,
            )
            chassis_states = self._rollout(initial_state, commands)
            return self._objective_gradient(
                commands,
                nominal_sequence,
                chassis_states=chassis_states,
                target_chassis_states=target_chassis_states,
            ).reshape(-1)

        def constraint_values(flattened_commands: np.ndarray) -> np.ndarray:
            evaluation = evaluate(flattened_commands)
            if evaluation is None:
                return invalid_constraints.copy()
            values: list[float] = list(evaluation.margins)
            for state in evaluation.safety_states[1:]:
                for axis in (0, 1):
                    values.append(float(state[axis] - grid_minimum[axis]))
                    values.append(float(grid_maximum[axis] - state[axis]))
            return np.asarray(values, dtype=np.float64)

        def constraint_jacobian(flattened_commands: np.ndarray) -> np.ndarray:
            evaluation = evaluate(flattened_commands)
            if evaluation is None:
                return np.zeros((5 * horizon, variable_count), dtype=np.float64)
            state_sensitivities = self._rollout_state_sensitivities(
                evaluation.chassis_states,
                evaluation.commands,
            )
            safety_sensitivities = self._safety_state_sensitivities(
                evaluation.chassis_states,
                state_sensitivities,
            )
            rows: list[np.ndarray] = []
            gradients = evaluation.samples.exact_gradients
            for index in range(horizon):
                rows.append(
                    gradients[index + 1] @ safety_sensitivities[index + 1]
                    - self.config.dtcbf_decay
                    * gradients[index]
                    @ safety_sensitivities[index]
                )
            for index in range(1, horizon + 1):
                for axis in (0, 1):
                    rows.append(safety_sensitivities[index, axis])
                    rows.append(-safety_sensitivities[index, axis])
            return np.asarray(rows, dtype=np.float64)

        actuator_lower, actuator_upper = self._actuator_bounds()
        bounds = tuple(zip(actuator_lower, actuator_upper))
        optimization = minimize(
            objective,
            reference.commands.reshape(-1),
            method="SLSQP",
            jac=objective_jacobian,
            bounds=bounds,
            constraints={
                "type": "ineq",
                "fun": constraint_values,
                "jac": constraint_jacobian,
            },
            options={
                "ftol": self.config.solver_tolerance,
                "maxiter": self.config.maximum_nonlinear_rescue_iterations,
                "disp": False,
            },
        )
        iteration_count = int(getattr(optimization, "nit", 0))
        # The optimizer uses smooth endpoint constraints/Jacobians only. Dense
        # mask work cannot change those values; do it once on the final proposal
        # before accepting it, including when the optimizer reports failure.
        candidate = self._evaluate_trajectory(
            initial_state,
            np.asarray(optimization.x, dtype=np.float64).reshape(horizon, 2),
            configuration_space=configuration_space,
            h_full=h_full,
            guidance_fields=guidance_fields,
            temporal_safety_derivative=temporal_safety_derivative,
        )
        if candidate is not None and self._trajectory_is_safe(candidate):
            return candidate, iteration_count, str(optimization.message)
        minimum_margin = (
            math.nan if candidate is None else float(np.min(candidate.margins))
        )
        return (
            None,
            iteration_count,
            f"{optimization.message}; minimum_margin={minimum_margin:.3e}",
        )

    def _solve_sparse_subproblem(
        self,
        reference: _TrajectoryEvaluation,
        *,
        nominal_sequence: np.ndarray,
        target_chassis_states: np.ndarray | None = None,
        configuration_space: SafeSAGEConfigurationSpaceResult,
        derivative_mode: str,
    ) -> _SparseQPSolution:
        """Assemble and solve one sparse Safe-SAGE SQP subproblem."""

        if derivative_mode == "guidance":
            derivatives = reference.samples.guidance_normals
        elif derivative_mode == "gradient":
            derivatives = reference.samples.exact_gradients
        else:
            raise ValueError("derivative_mode must be guidance or gradient.")

        state_sensitivities = self._rollout_state_sensitivities(
            reference.chassis_states,
            reference.commands,
        )
        safety_sensitivities = self._safety_state_sensitivities(
            reference.chassis_states,
            state_sensitivities,
        )
        command_variable_count = 2 * self.config.horizon_steps
        slack_variable_count = self.config.horizon_steps
        reference_flat = reference.commands.reshape(-1)

        cbf_rows = np.empty(
            (self.config.horizon_steps, command_variable_count),
            dtype=np.float64,
        )
        cbf_lower = np.empty(
            self.config.horizon_steps,
            dtype=np.float64,
        )
        for index in range(self.config.horizon_steps):
            row = (
                derivatives[index + 1] @ safety_sensitivities[index + 1]
                - self.config.dtcbf_decay
                * derivatives[index]
                @ safety_sensitivities[index]
            )
            cbf_rows[index] = row
            cbf_lower[index] = float(row @ reference_flat) - reference.margins[index]

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
        for index in range(1, self.config.horizon_steps + 1):
            for axis in (0, 1):
                row = safety_sensitivities[index, axis]
                offset = float(row @ reference_flat)
                grid_rows.append(row)
                grid_lower.append(
                    grid_minimum[axis] - reference.safety_states[index, axis] + offset
                )
                grid_upper.append(
                    grid_maximum[axis] - reference.safety_states[index, axis] + offset
                )

        actuator_lower, actuator_upper = self._actuator_bounds()
        trust_radius = np.tile(
            np.array(
                [
                    self.config.linear_velocity_trust_region,
                    self.config.angular_velocity_trust_region,
                ],
                dtype=np.float64,
            ),
            self.config.horizon_steps,
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
        zero_grid_slack = sparse.csc_matrix((len(grid_rows), slack_variable_count))
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
                np.full(self.config.horizon_steps, np.inf),
                np.asarray(grid_upper, dtype=np.float64),
            )
        )
        objective_matrix, objective_vector = self._quadratic_objective(
            nominal_sequence,
            reference=reference,
            state_sensitivities=state_sensitivities,
            target_chassis_states=target_chassis_states,
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

        status = str(result.info.status)
        solved = status.casefold() in {"solved", "solved inaccurate"}
        if not solved or result.x is None:
            return _SparseQPSolution(
                success=False,
                commands=reference.commands,
                iteration_count=int(result.info.iter),
                status=f"osqp_{status}",
                maximum_cbf_slack=math.inf,
            )
        optimizer = np.asarray(result.x, dtype=np.float64)
        commands = self._clip_commands(
            optimizer[:command_variable_count].reshape(
                self.config.horizon_steps,
                2,
            )
        )
        maximum_cbf_slack = float(
            np.max(optimizer[command_variable_count:], initial=0.0)
        )
        return _SparseQPSolution(
            success=True,
            commands=commands,
            iteration_count=int(result.info.iter),
            status=status,
            maximum_cbf_slack=maximum_cbf_slack,
        )

    def _quadratic_objective(
        self,
        nominal_sequence: np.ndarray,
        *,
        reference: _TrajectoryEvaluation,
        state_sensitivities: np.ndarray,
        target_chassis_states: np.ndarray | None,
        slack_variable_count: int,
    ) -> tuple[sparse.csc_matrix, np.ndarray]:
        """Return the local command-and-state tracking QP objective."""

        horizon = self.config.horizon_steps
        weights = np.tile(
            np.array(
                [
                    self.config.linear_velocity_weight,
                    self.config.angular_velocity_weight,
                ],
                dtype=np.float64,
            ),
            horizon,
        )
        matrix = np.diag(2.0 * weights + 1e-10)
        vector = -2.0 * weights * nominal_sequence.reshape(-1)

        if target_chassis_states is not None:
            reference_flat = reference.commands.reshape(-1)
            state_weights = np.array(
                [
                    self.config.position_tracking_weight,
                    self.config.position_tracking_weight,
                    self.config.yaw_tracking_weight,
                ],
                dtype=np.float64,
            )
            state_weight_matrix = np.diag(state_weights)
            for state_index in range(1, horizon + 1):
                sensitivity = state_sensitivities[state_index]
                state_error = self._state_tracking_error(
                    reference.chassis_states[state_index],
                    target_chassis_states[state_index],
                )
                affine_offset = state_error - sensitivity @ reference_flat
                matrix += 2.0 * (
                    sensitivity.T @ state_weight_matrix @ sensitivity
                )
                vector += 2.0 * (
                    sensitivity.T @ state_weight_matrix @ affine_offset
                )

        smoothness = self.config.command_smoothness_weight
        if smoothness > 0.0:
            for stage in range(horizon - 1):
                for command_axis in (0, 1):
                    first = 2 * stage + command_axis
                    second = first + 2
                    matrix[first, first] += 2.0 * smoothness
                    matrix[second, second] += 2.0 * smoothness
                    matrix[first, second] -= 2.0 * smoothness
                    matrix[second, first] -= 2.0 * smoothness

        upper_triangle = sparse.csc_matrix(np.triu(matrix))
        slack_objective = (
            sparse.eye(
                slack_variable_count,
                format="csc",
            )
            * 1.0e-10
        )
        objective_matrix = sparse.block_diag(
            (upper_triangle, slack_objective),
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

    def _rollout_state_sensitivities(
        self,
        chassis_states: np.ndarray,
        commands: np.ndarray,
    ) -> np.ndarray:
        """Return analytical ``d(x,y,yaw)_k / d(commands)`` tensors."""

        horizon = self.config.horizon_steps
        variable_count = 2 * horizon
        sensitivities = np.zeros(
            (horizon + 1, 3, variable_count),
            dtype=np.float64,
        )
        for index in range(horizon):
            _, state_matrix, control_matrix = self._interval_dynamics(
                chassis_states[index],
                commands[index],
            )
            sensitivities[index + 1] = state_matrix @ sensitivities[index]
            column = 2 * index
            sensitivities[index + 1, :, column : column + 2] += control_matrix
        return sensitivities

    def _safety_state_sensitivities(
        self,
        chassis_states: np.ndarray,
        state_sensitivities: np.ndarray,
    ) -> np.ndarray:
        """Map chassis sensitivities to forward look-ahead safety states."""

        result = np.empty_like(state_sensitivities)
        distance = self.config.lookahead_distance
        for index, state in enumerate(chassis_states):
            yaw = float(state[2])
            output_matrix = np.array(
                [
                    [1.0, 0.0, -distance * math.sin(yaw)],
                    [0.0, 1.0, distance * math.cos(yaw)],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            result[index] = output_matrix @ state_sensitivities[index]
        return result

    def _select_restoration_step(
        self,
        reference: _TrajectoryEvaluation,
        candidate: _TrajectoryEvaluation | None,
        *,
        initial_state: np.ndarray,
        proposed_commands: np.ndarray,
        configuration_space: SafeSAGEConfigurationSpaceResult,
        h_full: SafeSAGEHFullResult,
        guidance_fields: Sequence[LaplaceGuidanceFieldResult],
        temporal_safety_derivative: TemporalSafetyDerivativeInput,
    ) -> _TrajectoryEvaluation | None:
        """Backtrack until the sampled nonlinear DTCBF violation improves."""

        reference_violation = self._maximum_violation(reference)
        if candidate is not None and (
            self._trajectory_is_safe(candidate)
            or self._maximum_violation(candidate) < reference_violation
        ):
            return candidate

        step_size = 1.0
        best: _TrajectoryEvaluation | None = None
        best_violation = reference_violation
        for _ in range(self.config.maximum_line_search_iterations):
            step_size *= self.config.line_search_contraction
            trial_commands = reference.commands + step_size * (
                proposed_commands - reference.commands
            )
            trial = self._evaluate_trajectory(
                initial_state,
                trial_commands,
                configuration_space=configuration_space,
                h_full=h_full,
                guidance_fields=guidance_fields,
                temporal_safety_derivative=temporal_safety_derivative,
            )
            if trial is None:
                continue
            violation = self._maximum_violation(trial)
            if violation < best_violation:
                best = trial
                best_violation = violation
            if self._trajectory_is_safe(trial):
                return trial
        return best

    def _evaluate_trajectory(
        self,
        initial_state: np.ndarray,
        commands: np.ndarray,
        *,
        configuration_space: SafeSAGEConfigurationSpaceResult,
        h_full: SafeSAGEHFullResult,
        guidance_fields: Sequence[LaplaceGuidanceFieldResult],
        temporal_safety_derivative: TemporalSafetyDerivativeInput,
        require_feasible: bool = False,
        check_dense_geometry: bool = True,
    ) -> _TrajectoryEvaluation | None:
        """Roll out and sample one complete command sequence."""

        clipped_commands = self._clip_commands(commands)
        chassis_states = self._rollout(initial_state, clipped_commands)
        safety_states = self._safety_states(chassis_states)
        try:
            samples = self._sample_reference(
                safety_states,
                configuration_space=configuration_space,
                h_full=h_full,
                guidance_fields=guidance_fields,
                temporal_safety_derivative=temporal_safety_derivative,
            )
        except (TypeError, ValueError):
            return None
        safety_values = self._time_adjusted_safety_values(samples)
        margins = safety_values[1:] - self.config.dtcbf_decay * safety_values[:-1]
        if require_feasible and (
            not np.isfinite(margins).all()
            or np.any(samples.unsafe_configuration_mask)
            or float(np.min(margins)) < -self.config.constraint_tolerance
        ):
            # Dense geometry cannot repair an already failed exact DTCBF
            # constraint. Recovery does not use infeasible evaluations for
            # gradients or restoration, unlike the main solver.
            return None
        if (
            check_dense_geometry
            and not np.any(samples.unsafe_configuration_mask)
            and (
                self.config.safety_substeps_per_interval > 1
                or self.config.configuration_position_padding_m > 0
                or self.config.configuration_yaw_padding_radians > 0
            )
        ):
            unsafe_mask = self._inter_sample_unsafe_mask(
                chassis_states,
                samples.unsafe_configuration_mask,
                commands=clipped_commands,
                configuration_space=configuration_space,
                h_full=h_full,
                stop_on_first_unsafe=require_feasible,
            )
            if require_feasible and np.any(unsafe_mask):
                return None
            samples = replace(
                samples,
                unsafe_configuration_mask=unsafe_mask,
            )
        return _TrajectoryEvaluation(
            commands=clipped_commands,
            chassis_states=chassis_states,
            safety_states=safety_states,
            samples=samples,
            safety_values=safety_values,
            margins=margins,
        )

    def _inter_sample_unsafe_mask(
        self,
        chassis_states: np.ndarray,
        endpoint_mask: np.ndarray,
        *,
        commands: np.ndarray,
        configuration_space: SafeSAGEConfigurationSpaceResult,
        h_full: SafeSAGEHFullResult,
        stop_on_first_unsafe: bool = False,
    ) -> np.ndarray:
        """Mark intervals whose predicted motion enters an unsafe cell.

        DTCBF constraints remain defined at MPC sample times.  This denser
        geometry check prevents a command from leaving and then re-entering
        the safe configuration space between two samples.  Any unsafe
        interior point marks the interval's endpoint unsafe, so the existing
        exact-feasibility and recovery paths reject that command horizon.
        """

        unsafe = np.array(endpoint_mask, dtype=bool, copy=True)
        substeps = self.config.safety_substeps_per_interval
        fractions = np.arange(1, substeps + 1) / substeps
        starts = chassis_states[:-1]
        integration_count = self.config.integration_substeps_per_interval
        if integration_count > 1:
            dt = self.config.time_step / integration_count
            headings = starts[:, 2, None] + commands[:, 1, None] * dt * np.arange(integration_count)
            increments = commands[:, 0, None, None] * dt * np.stack(
                (np.cos(headings), np.sin(headings)), axis=-1,
            )
            indices = np.rint(fractions * integration_count).astype(int) - 1
            positions = starts[:, None, :2] + np.cumsum(increments, axis=1)[:, indices]
            yaws = starts[:, 2, None] + commands[:, 1, None] * self.config.time_step * fractions
        else:
            positions = starts[:, None, :2] + fractions[None, :, None] * (
                chassis_states[1:, None, :2] - starts[:, None, :2]
            )
            yaw_changes = (chassis_states[1:, 2] - starts[:, 2] + math.pi) % math.tau - math.pi
            yaws = starts[:, 2, None] + yaw_changes[:, None] * fractions
        yaws = (yaws + math.pi) % math.tau - math.pi
        safety_queries = np.concatenate((
            positions + self.config.lookahead_distance * np.stack(
                (np.cos(yaws), np.sin(yaws)), axis=-1,
            ),
            yaws[..., None],
        ), axis=-1).reshape(-1, 3)
        grid = configuration_space.occupancy_grids[0]
        position_padding = self.config.configuration_position_padding_m
        yaw_padding = self.config.configuration_yaw_padding_radians

        def offsets(padding, spacing):
            if padding == 0:
                return (0.0,)
            # Sample densely enough to cover every crossed discrete cell/slice.
            count = 2 * max(1, math.ceil(2 * padding / spacing)) + 1
            return np.linspace(-padding, padding, count)

        spatial_offsets = offsets(position_padding, grid.config.resolution)
        yaw_offsets = offsets(yaw_padding, math.tau / h_full.yaw_slice_count)
        perturbations = np.array([
            [x, y, yaw] for x in spatial_offsets for y in spatial_offsets
            for yaw in yaw_offsets
        ])
        scales = np.ones((self.config.horizon_steps, substeps))
        initial_fraction = self.config.configuration_padding_initial_fraction
        scales[0] = initial_fraction + (1.0 - initial_fraction) * fractions
        queries = (
            np.asarray(safety_queries)[:, None, :]
            + scales.reshape(-1, 1, 1) * perturbations[None, :, :]
        )
        queries = queries.reshape(self.config.horizon_steps, substeps, len(perturbations), 3)
        # Recovery never uses an infeasible candidate for gradients. One
        # actually unsafe query suffices to reject it; safe candidates still
        # receive every original query before they can be selected.
        chunk = 4 if stop_on_first_unsafe else self.config.horizon_steps
        for lower in range(0, self.config.horizon_steps, chunk):
            upper = min(self.config.horizon_steps, lower + chunk)
            try:
                query_mask = self.paper_filter.unsafe_configuration_mask(
                    queries[lower:upper].reshape(-1, 3),
                    configuration_space=configuration_space, h_full=h_full,
                ).reshape(upper-lower, substeps, len(perturbations))
            except ValueError:
                # A padded query outside the field is never safe evidence.
                unsafe[1:] = True
            else:
                unsafe[lower+1:upper+1] |= query_mask.any(axis=(1, 2))
            if stop_on_first_unsafe and np.any(unsafe):
                break
        return np.ascontiguousarray(unsafe, dtype=bool)

    def _trajectory_is_safe(
        self,
        trajectory: _TrajectoryEvaluation,
    ) -> bool:
        """Return whether every sampled DTCBF margin is valid."""

        return bool(
            np.isfinite(trajectory.margins).all()
            and not np.any(trajectory.samples.unsafe_configuration_mask)
            and float(np.min(trajectory.margins)) >= -self.config.constraint_tolerance
        )

    @staticmethod
    def _maximum_violation(trajectory: _TrajectoryEvaluation) -> float:
        """Return the nonnegative maximum DTCBF violation."""

        if (
            not np.isfinite(trajectory.margins).all()
            or np.any(trajectory.samples.unsafe_configuration_mask)
        ):
            return math.inf
        return max(0.0, -float(np.min(trajectory.margins)))

    def _nominal_sequence(
        self,
        nominal_command: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Expand one nominal command or validate a complete sequence."""

        return self._command_sequence(
            nominal_command,
            name="nominal_command",
        )

    def _target_state_sequence(
        self,
        target_value: Sequence[Sequence[float]] | np.ndarray | None,
    ) -> np.ndarray | None:
        """Validate one optional chassis-state path over the full horizon."""

        if target_value is None:
            return None
        target = np.asarray(target_value, dtype=np.float64)
        expected_shape = (self.config.horizon_steps + 1, 3)
        if target.shape != expected_shape:
            raise ValueError(
                "target_chassis_states_grid must have shape "
                f"{expected_shape}."
            )
        if not np.isfinite(target).all():
            raise ValueError(
                "target_chassis_states_grid must contain finite values."
            )
        # Own the validated reference so a caller cannot mutate the MPC
        # objective while one synchronous planning call is in progress.
        return np.array(target, dtype=np.float64, order="C", copy=True)

    def _command_sequence(
        self,
        command_value: Sequence[float] | np.ndarray,
        *,
        name: str,
    ) -> np.ndarray:
        """Expand one command or validate a complete horizon sequence."""

        command = np.asarray(command_value, dtype=np.float64)
        if command.shape == (2,):
            command = np.repeat(
                command[np.newaxis, :],
                self.config.horizon_steps,
                axis=0,
            )
        expected_shape = (self.config.horizon_steps, 2)
        if command.shape != expected_shape:
            raise ValueError(f"{name} must have shape (2,) or {expected_shape}.")
        if not np.isfinite(command).all():
            raise ValueError(f"{name} must contain finite values.")
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

    def _actuator_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return flattened lower and upper command bounds."""

        lower = np.tile(
            np.array(
                [
                    self.config.minimum_linear_velocity,
                    self.config.minimum_angular_velocity,
                ],
                dtype=np.float64,
            ),
            self.config.horizon_steps,
        )
        upper = np.tile(
            np.array(
                [
                    self.config.maximum_linear_velocity,
                    self.config.maximum_angular_velocity,
                ],
                dtype=np.float64,
            ),
            self.config.horizon_steps,
        )
        return lower, upper

    def _interval_dynamics(self, state, command, *, duration=None, substeps=None):
        """Constant-command Euler substeps and their analytical Jacobians.

        A vectorized sum is equivalent to repeated smaller Euler steps, whose
        headings change while executing a turn. Derivatives include angular
        velocity's influence on translation. One substep preserves the old
        endpoint model. This remains a kinematic model, not wheel/slip physics.
        """
        duration = self.config.time_step if duration is None else duration
        count = (
            self.config.integration_substeps_per_interval
            if substeps is None
            else substeps
        )
        dt = duration / count
        velocity, angular = command
        indices = np.arange(count, dtype=np.float64)
        angles = state[2] + indices * angular * dt
        cosines, sines = np.cos(angles), np.sin(angles)
        cosine_sum, sine_sum = float(cosines.sum()), float(sines.sum())
        if count == 1:
            next_state = integrate_unicycle_euler(state, command, time_step=duration)
        else:
            next_state = np.array(
                [
                    state[0] + velocity * dt * cosine_sum,
                    state[1] + velocity * dt * sine_sum,
                    wrap_to_pi(float(state[2] + angular * duration)),
                ]
            )
        state_matrix = np.array(
            [
                [1.0, 0.0, -velocity * dt * sine_sum],
                [0.0, 1.0, velocity * dt * cosine_sum],
                [0.0, 0.0, 1.0],
            ]
        )
        control_matrix = np.array(
            [
                [dt * cosine_sum, -velocity * dt * dt * float(indices @ sines)],
                [dt * sine_sum, velocity * dt * dt * float(indices @ cosines)],
                [0.0, duration],
            ]
        )
        return next_state, state_matrix, control_matrix

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
            states[index + 1], _, _ = self._interval_dynamics(states[index], command)
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
        temporal_safety_derivative: TemporalSafetyDerivativeInput,
    ) -> _ReferenceSamples:
        """Sample ``h_full``, Safe-SAGE guidance, and exact gradients."""

        values: list[float] = []
        guidance_normals: list[tuple[float, float, float]] = []
        exact_gradients: list[tuple[float, float, float]] = []
        temporal_terms: list[float] = []
        unsafe_configurations: list[bool] = []
        evaluations = self.paper_filter.evaluate_constraints(
            safety_states,
            configuration_space=configuration_space,
            h_full=h_full,
            guidance_fields=guidance_fields,
            temporal_safety_derivative=temporal_safety_derivative,
        )
        for evaluation in evaluations:
            values.append(evaluation.sampled_safety_value)
            guidance_normals.append(evaluation.constraint_normal)
            exact_gradients.append(evaluation.interpolant_gradient)
            temporal_terms.append(evaluation.scaled_temporal_safety_derivative)
            unsafe_configurations.append(evaluation.unsafe_reason is not None)
        return _ReferenceSamples(
            safety_values=np.asarray(values, dtype=np.float64),
            guidance_normals=np.asarray(
                guidance_normals,
                dtype=np.float64,
            ),
            exact_gradients=np.asarray(
                exact_gradients,
                dtype=np.float64,
            ),
            scaled_temporal_derivatives=np.asarray(
                temporal_terms,
                dtype=np.float64,
            ),
            unsafe_configuration_mask=np.asarray(
                unsafe_configurations,
                dtype=bool,
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
        return samples.safety_values + times * (samples.scaled_temporal_derivatives)

    def _objective_value(
        self,
        commands: np.ndarray,
        nominal_sequence: np.ndarray,
        *,
        chassis_states: np.ndarray | None = None,
        target_chassis_states: np.ndarray | None = None,
    ) -> float:
        """Return the command and optional chassis-path tracking objective."""

        command_error = commands - nominal_sequence
        value = np.sum(
            self.config.linear_velocity_weight * np.square(command_error[:, 0])
            + self.config.angular_velocity_weight * np.square(command_error[:, 1])
        )
        if self.config.command_smoothness_weight > 0.0:
            command_delta = np.diff(commands, axis=0)
            value += self.config.command_smoothness_weight * float(
                np.sum(np.square(command_delta))
            )
        if target_chassis_states is not None:
            if chassis_states is None:
                raise ValueError(
                    "chassis_states are required for state-path tracking."
                )
            errors = np.stack(
                [
                    self._state_tracking_error(state, target)
                    for state, target in zip(
                        chassis_states[1:],
                        target_chassis_states[1:],
                    )
                ]
            )
            value += self.config.position_tracking_weight * float(
                np.sum(np.square(errors[:, :2]))
            )
            value += self.config.yaw_tracking_weight * float(
                np.sum(np.square(errors[:, 2]))
            )
        return float(value)

    def _objective_gradient(
        self,
        commands: np.ndarray,
        nominal_sequence: np.ndarray,
        *,
        chassis_states: np.ndarray,
        target_chassis_states: np.ndarray | None,
    ) -> np.ndarray:
        """Return the analytical objective gradient with respect to commands."""

        command_weights = np.array(
            [
                self.config.linear_velocity_weight,
                self.config.angular_velocity_weight,
            ],
            dtype=np.float64,
        )
        gradient = 2.0 * command_weights * (commands - nominal_sequence)
        smoothness = self.config.command_smoothness_weight
        if smoothness > 0.0:
            command_differences = np.diff(commands, axis=0)
            gradient[:-1] -= 2.0 * smoothness * command_differences
            gradient[1:] += 2.0 * smoothness * command_differences

        if target_chassis_states is not None:
            sensitivities = self._rollout_state_sensitivities(
                chassis_states,
                commands,
            )
            state_weights = np.array(
                [
                    self.config.position_tracking_weight,
                    self.config.position_tracking_weight,
                    self.config.yaw_tracking_weight,
                ],
                dtype=np.float64,
            )
            state_gradient = np.zeros(2 * self.config.horizon_steps)
            for state_index in range(1, self.config.horizon_steps + 1):
                state_error = self._state_tracking_error(
                    chassis_states[state_index],
                    target_chassis_states[state_index],
                )
                state_gradient += 2.0 * (
                    sensitivities[state_index].T
                    @ (state_weights * state_error)
                )
            gradient += state_gradient.reshape(self.config.horizon_steps, 2)
        return gradient

    @staticmethod
    def _state_tracking_error(
        chassis_state: np.ndarray,
        target_chassis_state: np.ndarray,
    ) -> np.ndarray:
        """Return position error and the shortest signed yaw error."""

        return np.array(
            [
                chassis_state[0] - target_chassis_state[0],
                chassis_state[1] - target_chassis_state[1],
                wrap_to_pi(
                    float(chassis_state[2] - target_chassis_state[2])
                ),
            ],
            dtype=np.float64,
        )

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
            restoration_used=False,
            status=status,
        )

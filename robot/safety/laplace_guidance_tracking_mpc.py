"""Explicit guidance-heading tracking ablation; never the primary objective.

Each plan freezes a heading reference from the current field sampled along the
unchanged obstacle-unaware nominal rollout. The added nonnegative quadratic
cost is used by QP, rescue, recovery and exact objective comparisons alike.
The reference is a preference, not an executable or certified-safe command.
"""

from dataclasses import dataclass, fields
import math
from numbers import Real

import numpy as np
from scipy import sparse

from robot.safety.safe_sage_unicycle_mpc import (
    SafeSAGEUnicycleMPC,
    SafeSAGEUnicycleMPCResult,
)


@dataclass(frozen=True)
class LaplaceGuidanceTrackingConfig:
    weight: float = 0.0
    activation_safety_value: float = 1.0
    maximum_heading_bias_radians: float = math.pi / 4

    def __post_init__(self):
        for name, value in (
            ("weight", self.weight),
            ("activation_safety_value", self.activation_safety_value),
            ("maximum_heading_bias_radians", self.maximum_heading_bias_radians),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite and real")
        if self.weight < 0 or self.activation_safety_value <= 0:
            raise ValueError(
                "weight must be nonnegative; activation_safety_value positive"
            )
        if not 0 < self.maximum_heading_bias_radians < math.pi / 2:
            raise ValueError("maximum heading bias must be in (0, pi/2)")


@dataclass(frozen=True)
class GuidanceTrackingDiagnostic:
    basis: str
    weight: float
    activation_safety_value: float
    maximum_heading_bias_radians: float
    nominal_terminal_heading_radians: float
    heading_reference_radians: tuple[float, ...]
    activation: tuple[float, ...]
    unavailable_reference_stages: tuple[int, ...]
    guidance_seed_used: bool = False
    nominal_unchanged: bool = True


@dataclass(frozen=True)
class GuidanceTrackingMPCResult(SafeSAGEUnicycleMPCResult):
    guidance_tracking: GuidanceTrackingDiagnostic


class LaplaceGuidanceTrackingMPC(SafeSAGEUnicycleMPC):
    """A separate opt-in objective with the primary MPC's safety machinery."""

    def __init__(self, paper_filter, config=None, *, guidance_config=None):
        super().__init__(paper_filter, config)
        self.guidance_config = guidance_config or LaplaceGuidanceTrackingConfig()
        if not isinstance(self.guidance_config, LaplaceGuidanceTrackingConfig):
            raise TypeError("guidance_config must be LaplaceGuidanceTrackingConfig")

    def plan(self, initial_chassis_state_grid, nominal_command, **kwargs):
        if self.guidance_config.weight == 0:
            return super().plan(initial_chassis_state_grid, nominal_command, **kwargs)
        initial = np.asarray(initial_chassis_state_grid, dtype=np.float64)
        if initial.shape != (3,) or not np.isfinite(initial).all():
            raise ValueError("initial_chassis_state_grid must be a finite three-vector")
        nominal = self._nominal_sequence(nominal_command)
        kwargs = dict(kwargs)
        kwargs["guidance_fields"] = tuple(kwargs["guidance_fields"])
        if "target_chassis_states_grid" in kwargs:
            kwargs["target_chassis_states_grid"] = self._target_state_sequence(
                kwargs["target_chassis_states_grid"]
            )
        nominal_states = self._rollout(initial, self._clip_commands(nominal))
        safety_states = self._safety_states(nominal_states)[:-1]
        grid = kwargs["configuration_space"].occupancy_grids[0].config
        # Only the auxiliary reference may omit out-of-domain nominal stages.
        # The inherited planner still validates every candidate and its field.
        valid = (
            (safety_states[:, 0] >= grid.minimum_x + grid.resolution / 2)
            & (
                safety_states[:, 0]
                <= grid.minimum_x + (grid.column_count - 0.5) * grid.resolution
            )
            & (safety_states[:, 0] < grid.maximum_x)
            & (safety_states[:, 1] >= grid.minimum_y + grid.resolution / 2)
            & (
                safety_states[:, 1]
                <= grid.minimum_y + (grid.row_count - 0.5) * grid.resolution
            )
            & (safety_states[:, 1] < grid.maximum_y)
        )
        targets = np.array(nominal_states[1:, 2], copy=True)
        route_yaw = float(nominal_states[-1, 2])
        activation = np.zeros(self.config.horizon_steps)
        unavailable = []
        indices = np.flatnonzero(valid)
        samples = (
            self.paper_filter.evaluate_constraints(
                safety_states[indices],
                configuration_space=kwargs["configuration_space"],
                h_full=kwargs["h_full"],
                guidance_fields=kwargs["guidance_fields"],
            )
            if len(indices)
            else ()
        )
        for i, sample in zip(indices, samples, strict=True):
            guidance = np.asarray(sample.sampled_guidance_grid_xy)
            norm = float(np.linalg.norm(guidance))
            if sample.unsafe_reason is not None or norm <= 1e-9 or nominal[i, 0] <= 0:
                unavailable.append(int(i))
                continue
            nominal_yaw = route_yaw
            bias = (
                math.atan2(guidance[1], guidance[0]) - nominal_yaw + math.pi
            ) % math.tau - math.pi
            targets[i] = nominal_yaw + np.clip(
                bias,
                -self.guidance_config.maximum_heading_bias_radians,
                self.guidance_config.maximum_heading_bias_radians,
            )
            activation[i] = (
                np.clip(
                    1.0
                    - sample.sampled_safety_value
                    / self.guidance_config.activation_safety_value,
                    0.0,
                    1.0,
                )
                ** 2
            )
        unavailable.extend(int(i) for i in np.flatnonzero(~valid))
        weights = self.guidance_config.weight * activation
        worker = _GuidanceObjectiveMPC(
            self.paper_filter, self.config, targets=targets, weights=weights
        )
        seed_used = False
        if np.any(weights > 0):
            supplied = kwargs.get("initial_command_guess")
            incumbent = worker._evaluate_trajectory(
                initial,
                worker._clip_commands(
                    nominal
                    if supplied is None
                    else worker._command_sequence(
                        supplied, name="initial_command_guess"
                    )
                ),
                configuration_space=kwargs["configuration_space"],
                h_full=kwargs["h_full"],
                guidance_fields=kwargs["guidance_fields"],
                temporal_safety_derivative=kwargs.get(
                    "temporal_safety_derivative", 0.0
                ),
            )
            best_cost = (
                worker._objective_value(
                    incumbent.commands,
                    nominal,
                    chassis_states=incumbent.chassis_states,
                    target_chassis_states=kwargs.get("target_chassis_states_grid"),
                )
                if incumbent is not None and worker._trajectory_is_safe(incumbent)
                else math.inf
            )
            for turn_bound in (0.4, 0.8):
                commands = self._guidance_rollout(
                    initial, nominal, route_yaw, turn_bound, kwargs
                )
                if commands is None:
                    continue
                candidate = worker._evaluate_trajectory(
                    initial,
                    commands,
                    require_feasible=True,
                    configuration_space=kwargs["configuration_space"],
                    h_full=kwargs["h_full"],
                    guidance_fields=kwargs["guidance_fields"],
                    temporal_safety_derivative=kwargs.get(
                        "temporal_safety_derivative", 0.0
                    ),
                )
                if candidate is None or not worker._trajectory_is_safe(candidate):
                    continue
                cost = worker._objective_value(
                    candidate.commands,
                    nominal,
                    chassis_states=candidate.chassis_states,
                    target_chassis_states=kwargs.get("target_chassis_states_grid"),
                )
                if cost < best_cost - self.config.trajectory_tolerance:
                    kwargs["initial_command_guess"] = candidate.commands
                    best_cost, seed_used = cost, True
        diagnostic = GuidanceTrackingDiagnostic(
            basis="laplace_guidance_nominal_rollout_bounded_heading_bias_ablation",
            weight=self.guidance_config.weight,
            activation_safety_value=self.guidance_config.activation_safety_value,
            maximum_heading_bias_radians=self.guidance_config.maximum_heading_bias_radians,
            nominal_terminal_heading_radians=route_yaw,
            heading_reference_radians=tuple(map(float, targets)),
            activation=tuple(map(float, activation)),
            unavailable_reference_stages=tuple(sorted(unavailable)),
            guidance_seed_used=seed_used,
        )
        result = worker.plan(initial, nominal, **kwargs)
        return GuidanceTrackingMPCResult(
            **{
                f.name: getattr(result, f.name)
                for f in fields(SafeSAGEUnicycleMPCResult)
            },
            guidance_tracking=diagnostic,
        )

    def _guidance_rollout(self, initial, nominal, route_yaw, turn_bound, field):
        """Build only an optimization seed; exact validation follows in plan."""
        state = initial.copy()
        commands = np.empty_like(nominal)
        clipped_nominal = self._clip_commands(nominal)
        grid = field["configuration_space"].occupancy_grids[0].config
        for i in range(self.config.horizon_steps):
            if nominal[i, 0] <= 0:
                commands[i] = clipped_nominal[i]
                state, _, _ = self._interval_dynamics(state, commands[i])
                continue
            safety = self._safety_states(state[None, :])[0]
            if not (
                grid.minimum_x + grid.resolution / 2
                <= safety[0]
                <= grid.minimum_x + (grid.column_count - 0.5) * grid.resolution
                and grid.minimum_y + grid.resolution / 2
                <= safety[1]
                <= grid.minimum_y + (grid.row_count - 0.5) * grid.resolution
                and safety[0] < grid.maximum_x
                and safety[1] < grid.maximum_y
            ):
                return None
            sample = self.paper_filter.evaluate_constraint(
                query_position_grid_xy=safety[:2],
                query_yaw_radians=float(safety[2]),
                configuration_space=field["configuration_space"],
                h_full=field["h_full"],
                guidance_fields=field["guidance_fields"],
            )
            g = sample.sampled_guidance_grid_xy
            if sample.unsafe_reason is not None or math.hypot(*g) <= 1e-9:
                return None
            bias = (math.atan2(g[1], g[0]) - route_yaw + math.pi) % math.tau - math.pi
            desired_yaw = route_yaw + np.clip(
                bias,
                -self.guidance_config.maximum_heading_bias_radians,
                self.guidance_config.maximum_heading_bias_radians,
            )
            error = (desired_yaw - state[2] + math.pi) % math.tau - math.pi
            commands[i] = (
                np.clip(
                    nominal[i, 0],
                    self.config.minimum_linear_velocity,
                    self.config.maximum_linear_velocity,
                ),
                np.clip(
                    error / self.config.time_step,
                    max(-turn_bound, self.config.minimum_angular_velocity),
                    min(turn_bound, self.config.maximum_angular_velocity),
                ),
            )
            state, _, _ = self._interval_dynamics(state, commands[i])
        return commands


class _GuidanceObjectiveMPC(SafeSAGEUnicycleMPC):
    """Own one plan's immutable quadratic data; no shared mutable context."""

    def __init__(self, paper_filter, config, *, targets, weights):
        super().__init__(paper_filter, config)
        self.targets = np.array(targets, dtype=np.float64, copy=True)
        self.weights = np.array(weights, dtype=np.float64, copy=True)
        self.targets.setflags(write=False)
        self.weights.setflags(write=False)
        self._additional_tracking_objective_active = bool(np.any(self.weights > 0))

    def _objective_value(self, commands, nominal_sequence, **kwargs):
        value = super()._objective_value(commands, nominal_sequence, **kwargs)
        states = kwargs.get("chassis_states")
        # Cheap recovery calls omit states and need a lower bound. The added
        # cost is nonnegative; exact candidate ranking always supplies states.
        if states is not None:
            value += float(np.sum(self.weights * self._heading_errors(states) ** 2))
        return value

    def _heading_errors(self, states):
        return (states[1:, 2] - self.targets + math.pi) % math.tau - math.pi

    def _objective_gradient(self, commands, nominal_sequence, **kwargs):
        gradient = super()._objective_gradient(commands, nominal_sequence, **kwargs)
        states = kwargs["chassis_states"]
        sensitivities = self._rollout_state_sensitivities(states, commands)[1:, 2, :]
        extra = 2.0 * sensitivities.T @ (self.weights * self._heading_errors(states))
        return gradient + extra.reshape(commands.shape)

    def _quadratic_objective(self, nominal_sequence, **kwargs):
        matrix, vector = super()._quadratic_objective(nominal_sequence, **kwargs)
        reference = kwargs["reference"]
        jacobian = kwargs["state_sensitivities"][1:, 2, :]
        offset = self._heading_errors(
            reference.chassis_states
        ) - jacobian @ reference.commands.reshape(-1)
        extra_matrix = 2.0 * jacobian.T @ (self.weights[:, None] * jacobian)
        extra_vector = 2.0 * jacobian.T @ (self.weights * offset)
        slack = kwargs["slack_variable_count"]
        addition = sparse.block_diag(
            (
                sparse.csc_matrix(np.triu(extra_matrix)),
                sparse.csc_matrix((slack, slack)),
            ),
            format="csc",
        )
        return matrix + addition, vector + np.concatenate(
            (extra_vector, np.zeros(slack))
        )

"""Separate MPC ablation tracking a checked, reachable local guidance path.

The waypoint nominal stays obstacle-unaware. Guidance changes only a local MPC
tracking reference, never physical evidence, field policy or safety acceptance.
"""

from dataclasses import dataclass, fields, replace
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
    maximum_heading_bias_radians: float = math.pi / 3
    reference_goal_radius_m: float = 0.04
    guidance_fade_distance_m: float = 0.5
    minimum_reference_progress_m: float = 0.01

    def __post_init__(self):
        for name, value in vars(self).items():
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite and real")
        if (
            self.weight < 0
            or self.activation_safety_value <= 0
            or self.reference_goal_radius_m <= 0
            or self.guidance_fade_distance_m <= self.reference_goal_radius_m
            or self.minimum_reference_progress_m < 0
        ):
            raise ValueError(
                "weight must be nonnegative; activation/goal radius positive; fade distance greater than goal radius"
            )
        if not 0 < self.maximum_heading_bias_radians < math.pi / 2:
            raise ValueError("maximum heading bias must be in (0, pi/2)")


@dataclass(frozen=True)
class GuidanceTrackingDiagnostic:
    basis: str
    active: bool
    reason: str
    weight: float
    activation: float
    active_nominal_goal_grid_xy: tuple[float, float]
    preferred_forward_speed_mps: float
    checked_reference_count: int
    selected_turn_rate_cap: float | None
    selected_turn_first: bool | None
    reference_goal_progress_m: float | None
    effective_position_weight: float
    effective_yaw_weight: float
    command_tracking_weights: tuple[float, float]
    reference_commands: tuple[tuple[float, float], ...]
    reference_chassis_states: tuple[tuple[float, float, float], ...]
    minimum_reference_progress_m: float
    rejected_no_progress_reference_count: int
    nominal_unchanged: bool = True


@dataclass(frozen=True)
class GuidanceTrackingMPCResult(SafeSAGEUnicycleMPCResult):
    guidance_tracking: GuidanceTrackingDiagnostic


def _wrap(angle):
    return (angle + math.pi) % math.tau - math.pi


class _ReferenceTrackingMPC(SafeSAGEUnicycleMPC):
    """One-solve owned command reference, shared by every cost path."""

    def __init__(self, paper_filter, config, commands, weights):
        super().__init__(paper_filter, config)
        self.reference_commands = np.array(commands, dtype=np.float64, copy=True)
        self.reference_weights = np.array(weights, dtype=np.float64, copy=True)
        self.reference_commands.setflags(write=False)
        self.reference_weights.setflags(write=False)

    def _objective_value(self, commands, nominal_sequence, **kwargs):
        return super()._objective_value(commands, nominal_sequence, **kwargs) + float(
            np.sum(self.reference_weights * (commands - self.reference_commands) ** 2)
        )

    def _objective_gradient(self, commands, nominal_sequence, **kwargs):
        return super()._objective_gradient(commands, nominal_sequence, **kwargs) + (
            2 * self.reference_weights * (commands - self.reference_commands)
        )

    def _quadratic_objective(self, nominal_sequence, **kwargs):
        matrix, vector = super()._quadratic_objective(nominal_sequence, **kwargs)
        weights = np.tile(self.reference_weights, self.config.horizon_steps)
        padding = np.zeros(kwargs["slack_variable_count"])
        matrix += sparse.diags(np.concatenate((2 * weights, padding)), format="csc")
        vector += np.concatenate(
            (-2 * weights * self.reference_commands.ravel(), padding)
        )
        return matrix, vector


class LaplaceGuidanceTrackingMPC(SafeSAGEUnicycleMPC):
    """Opt-in local reference; all execution uses inherited exact checks."""

    def __init__(self, paper_filter, config=None, *, guidance_config=None):
        super().__init__(paper_filter, config)
        self.guidance_config = guidance_config or LaplaceGuidanceTrackingConfig()
        if not isinstance(self.guidance_config, LaplaceGuidanceTrackingConfig):
            raise TypeError("guidance_config must be LaplaceGuidanceTrackingConfig")

    def plan(
        self,
        initial_chassis_state_grid,
        nominal_command,
        *,
        nominal_goal_grid_xy=None,
        **kwargs,
    ):
        if self.guidance_config.weight == 0:
            return super().plan(initial_chassis_state_grid, nominal_command, **kwargs)
        initial = np.asarray(initial_chassis_state_grid, dtype=np.float64)
        goal = np.asarray(nominal_goal_grid_xy, dtype=np.float64)
        if initial.shape != (3,) or not np.isfinite(initial).all():
            raise ValueError("initial_chassis_state_grid must be a finite three-vector")
        if goal.shape != (2,) or not np.isfinite(goal).all():
            raise ValueError("nominal_goal_grid_xy must be a finite two-vector")
        if kwargs.get("target_chassis_states_grid") is not None:
            raise ValueError(
                "guidance path ablation cannot silently replace a separate state-path target"
            )
        kwargs = dict(kwargs)
        kwargs["guidance_fields"] = tuple(kwargs["guidance_fields"])
        nominal = self._nominal_sequence(nominal_command)
        speed = float(
            np.clip(
                np.max(nominal[:, 0]),
                self.config.minimum_linear_velocity,
                self.config.maximum_linear_velocity,
            )
        )
        current = self._sample(initial, kwargs)
        activation = float(
            np.clip(
                1.0
                - current.sampled_safety_value
                / self.guidance_config.activation_safety_value,
                0.0,
                1.0,
            )
            ** 2
        )
        best, rank, checked = None, 0.0, 0
        rejected_no_progress = 0
        selected = (None, None)
        progress = None
        reason = "inactive_field_preference"
        if activation > 0 and speed > 0 and current.unsafe_reason is None:
            reason = "no_exact_safe_progress_reference"
            direction = self._desired_heading(
                initial, goal, current.sampled_guidance_grid_xy
            )
            initial_error = abs(_wrap(direction - initial[2]))
            distance = float(np.linalg.norm(goal - initial[:2]))
            for cap, turn_first in (
                (0.4, False),
                (0.8, False),
                (0.1, True),
                (0.2, True),
                (0.4, True),
                (0.8, True),
            ):
                commands = self._guidance_rollout(
                    initial, goal, speed, cap, turn_first, kwargs
                )
                if commands is None:
                    continue
                candidate = self._evaluate_trajectory(
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
                checked += 1
                if candidate is None or not self._trajectory_is_safe(candidate):
                    continue
                gain = distance - float(
                    np.linalg.norm(goal - candidate.chassis_states[-1, :2])
                )
                turning = initial_error - abs(
                    _wrap(direction - candidate.chassis_states[-1, 2])
                )
                score = gain + self.config.lookahead_distance * turning
                reached = (
                    np.linalg.norm(goal - candidate.chassis_states[-1, :2])
                    <= self.guidance_config.reference_goal_radius_m
                )
                # Heading improvement alone is not a useful path reference.
                # A turn-only horizon with a large tracking weight repeatedly
                # rewards postponing translation at each replanning step.
                # This is an auxiliary-reference quality gate, never a new
                # executable constraint: the primary can still turn or take
                # a detour when no useful checked reference exists.
                minimum_gain = min(self.guidance_config.minimum_reference_progress_m,
                    max(0., distance - self.guidance_config.reference_goal_radius_m))
                if (self.guidance_config.minimum_reference_progress_m > 0
                    and not reached and gain + 1e-9 < minimum_gain):
                    rejected_no_progress += 1
                    continue
                if score > rank + 1e-6 or (best is None and reached):
                    best, rank, selected, progress = (
                        candidate,
                        score,
                        (cap, turn_first),
                        gain,
                    )
            if best is not None:
                reason = "exact_safe_local_guidance_reference"
        effective = (
            self.guidance_config.weight * activation if best is not None else 0.0
        )
        command_weights = effective / np.square(
            [
                max(
                    1e-9,
                    abs(self.config.minimum_linear_velocity),
                    abs(self.config.maximum_linear_velocity),
                ),
                max(
                    1e-9,
                    abs(self.config.minimum_angular_velocity),
                    abs(self.config.maximum_angular_velocity),
                ),
            ]
        )
        if best is None:
            result = super().plan(initial, nominal, **kwargs)
        else:
            # Track the checked reference states and normalized actuator commands.
            # Exact execution acceptance remains the inherited control path.
            worker = _ReferenceTrackingMPC(
                self.paper_filter,
                replace(
                    self.config,
                    position_tracking_weight=effective,
                    yaw_tracking_weight=effective * self.config.lookahead_distance**2,
                ),
                best.commands,
                command_weights,
            )
            kwargs["target_chassis_states_grid"] = best.chassis_states
            kwargs["initial_command_guess"] = best.commands
            result = worker.plan(initial, nominal, **kwargs)
        diagnostic = GuidanceTrackingDiagnostic(
            basis="checked_unicycle_laplace_local_path_reference_ablation",
            active=best is not None,
            reason=reason,
            weight=self.guidance_config.weight,
            activation=activation,
            active_nominal_goal_grid_xy=tuple(map(float, goal)),
            preferred_forward_speed_mps=speed,
            checked_reference_count=checked,
            selected_turn_rate_cap=selected[0],
            selected_turn_first=selected[1],
            reference_goal_progress_m=progress,
            effective_position_weight=effective,
            effective_yaw_weight=effective * self.config.lookahead_distance**2,
            command_tracking_weights=tuple(map(float, command_weights)),
            reference_commands=()
            if best is None
            else tuple(tuple(map(float, r)) for r in best.commands),
            reference_chassis_states=()
            if best is None
            else tuple(tuple(map(float, r)) for r in best.chassis_states),
            minimum_reference_progress_m=self.guidance_config.minimum_reference_progress_m,
            rejected_no_progress_reference_count=rejected_no_progress,
        )
        return GuidanceTrackingMPCResult(
            **{
                f.name: getattr(result, f.name)
                for f in fields(SafeSAGEUnicycleMPCResult)
            },
            guidance_tracking=diagnostic,
        )

    def _sample(self, state, field):
        safety = self._safety_states(state[None, :])[0]
        return self.paper_filter.evaluate_constraint(
            query_position_grid_xy=safety[:2],
            query_yaw_radians=float(safety[2]),
            configuration_space=field["configuration_space"],
            h_full=field["h_full"],
            guidance_fields=field["guidance_fields"],
        )

    def _desired_heading(self, state, goal, guidance):
        delta = goal - state[:2]
        bearing = math.atan2(delta[1], delta[0])
        bias = _wrap(math.atan2(guidance[1], guidance[0]) - bearing)
        fade = (
            min(
                float(np.linalg.norm(delta))
                / self.guidance_config.guidance_fade_distance_m,
                1.0,
            )
            ** 2
        )
        return bearing + fade * float(
            np.clip(
                bias,
                -self.guidance_config.maximum_heading_bias_radians,
                self.guidance_config.maximum_heading_bias_radians,
            )
        )

    def _guidance_rollout(self, initial, goal, speed, cap, turn_first, field):
        state = initial.copy()
        commands = np.zeros((self.config.horizon_steps, 2))
        grid = field["configuration_space"].occupancy_grids[0].config
        for i in range(self.config.horizon_steps):
            distance = float(np.linalg.norm(goal - state[:2]))
            if distance <= self.guidance_config.reference_goal_radius_m:
                break
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
            sample = self._sample(state, field)
            if (
                sample.unsafe_reason is not None
                or math.hypot(*sample.sampled_guidance_grid_xy) <= 1e-9
            ):
                return None
            desired = self._desired_heading(
                state, goal, sample.sampled_guidance_grid_xy
            )
            error = _wrap(desired - state[2])
            goal_bearing = math.atan2(goal[1] - state[1], goal[0] - state[0])
            radial_alignment = max(math.cos(_wrap(goal_bearing - state[2])), 0.0)
            # The reference may turn in place; reduce translation when the
            # current heading points away from the configured goal.
            # This is reference generation, not a new executable constraint.
            forward = min(
                speed * max(math.cos(error), 0.0),
                distance * radial_alignment / self.config.time_step,
            )
            if turn_first and abs(error) > 0.1:
                forward = 0.0
            commands[i] = (
                np.clip(
                    forward,
                    self.config.minimum_linear_velocity,
                    self.config.maximum_linear_velocity,
                ),
                np.clip(
                    error / self.config.time_step,
                    max(-cap, self.config.minimum_angular_velocity),
                    min(cap, self.config.maximum_angular_velocity),
                ),
            )
            state, _, _ = self._interval_dynamics(state, commands[i])
        return commands

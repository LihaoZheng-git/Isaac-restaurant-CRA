"""Separate MPC ablation tracking a checked, reachable local guidance path.

The waypoint nominal stays obstacle-unaware. Guidance changes only a local MPC
tracking reference, never physical evidence, field policy or safety acceptance.
"""

from dataclasses import dataclass, fields, replace
import math
from numbers import Real

import numpy as np
from scipy import sparse

from robot.control.unicycle_model import lookahead_control_matrix
from robot.safety.safe_sage_cbf_filter import project_weighted_box_halfspace
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
    minimum_turn_first_rate_cap: float = 0.4
    reference_mode: str = "bounded_field_heading"
    reference_search_width: int = 0
    reference_search_maximum_candidates: int = 4800
    tangent_velocity_fraction: float = 0.0
    reference_tangent_basis: str = "planar_gradient"
    minimum_reference_speed_fraction: float = 1.0

    def __post_init__(self):
        if self.reference_mode not in {
            "bounded_field_heading", "filtered_goal_velocity", "planar_forecast_velocity",
        }:
            raise ValueError("unknown guidance reference mode")
        for name, value in vars(self).items():
            if name in {"reference_mode", "reference_tangent_basis"}:
                continue
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
        if not 0 < self.minimum_turn_first_rate_cap <= .8:
            raise ValueError("minimum turn-first pool cap must be in (0,.8]")
        if (not isinstance(self.reference_search_width, int) or not 0 <= self.reference_search_width <= 8
            or not isinstance(self.reference_search_maximum_candidates, int)
            or self.reference_search_maximum_candidates <= 0
            or self.reference_search_width > 0 and self.reference_mode != "planar_forecast_velocity"):
            raise ValueError("bounded search requires width 0..8, positive integer candidate budget and planar forecast mode")
        if not .25 <= self.minimum_reference_speed_fraction <= 1:
            raise ValueError("minimum reference speed fraction must be in [.25,1]")
        if not 0 <= self.tangent_velocity_fraction <= 1 or (
            self.tangent_velocity_fraction > 0 and (self.weight <= 0
            or self.reference_mode != "planar_forecast_velocity")
        ):
            raise ValueError("tangent fraction requires 0..1, positive tracking weight and planar forecast mode")
        if self.reference_tangent_basis not in {"planar_gradient", "selected_observed_obstacle"} or (
            self.reference_tangent_basis == "selected_observed_obstacle" and self.reference_search_width > 0):
            raise ValueError("selected observed tangent requires an explicit supported basis and no legacy reference search")

    def reference_speed_fractions(self):
        """At most four explicit candidate speeds; default retains one."""
        return tuple(dict.fromkeys(max(self.minimum_reference_speed_fraction, value)
            for value in (1., .75, .5, .25)))


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
    minimum_turn_first_rate_cap: float
    current_guidance_grid_xy: tuple[float, float]
    initial_chassis_state_grid: tuple[float, float, float]
    first_reference_translation_delay_seconds: float | None
    reference_mode: str
    reference_displacement_m: float | None
    reference_progress_basis: str
    reference_uses_temporal_forecast: bool
    reference_projection_fixed_angular_velocity: float | None
    reference_search: dict | None = None
    nominal_unchanged: bool = True
    tangent_velocity_fraction: float = 0.0
    tangent_reference_track_ids: tuple[int, ...] = ()
    reference_projection_basis: str = "legacy_reference_projection"
    reference_tangent_basis: str = "planar_gradient"
    minimum_reference_speed_fraction: float = 1.0
    selected_reference_speed_fraction: float | None = None


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
        selected_passing_tangents=(),
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
        references = tuple(selected_passing_tangents)
        if references:
            if (self.guidance_config.reference_tangent_basis != "selected_observed_obstacle"
                or len(references) != 1):
                raise ValueError("selected observed tangent requires one explicit current encounter")
            key, center, side = references[0]
            if (isinstance(key, bool) or not isinstance(key, int) or key < 0
                or np.asarray(center).shape != (2,) or not np.isfinite(center).all()
                or side not in {"left", "right"}):
                raise ValueError("selected observed tangent requires finite current geometry and explicit side")
        elif self.guidance_config.reference_tangent_basis == "selected_observed_obstacle" and self.guidance_config.tangent_velocity_fraction > 0:
            raise ValueError("selected observed tangent requires one explicit current encounter")
        kwargs["guidance_fields"] = tuple(kwargs["guidance_fields"])
        reference_kwargs = dict(kwargs, selected_passing_tangents=references)
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
        projected = self.guidance_config.reference_mode != "bounded_field_heading"
        best, rank, checked = None, -math.inf if projected else 0.0, 0
        rejected_no_progress = 0
        selected = (None, None, None)
        progress = None
        displacement = None
        reason = "inactive_field_preference"
        search_audit = None
        if activation > 0 and speed > 0 and current.unsafe_reason is None:
            reason = "no_exact_safe_progress_reference"
            distance = float(np.linalg.norm(goal - initial[:2]))
            # Candidate generation only: reduced speeds may fit a curved
            # exact-safe horizon where full-speed guidance does not.
            for speed_fraction in self.guidance_config.reference_speed_fractions():
                candidate_speed = speed * speed_fraction
                preference = self._reference_preference(initial, goal, candidate_speed, current, reference_kwargs)
                direction = initial[2] if preference is None else preference[0]
                initial_error = abs(_wrap(direction - initial[2]))
                for cap, turn_first in (
                    (0.4, False),
                    (0.8, False),
                    (0.1, True),
                    (0.2, True),
                    (0.4, True),
                    (0.8, True),
                ):
                    if turn_first and cap < self.guidance_config.minimum_turn_first_rate_cap:
                        continue
                    commands = self._guidance_rollout(
                        initial, goal, candidate_speed, cap, turn_first, reference_kwargs
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
                    moved = float(np.linalg.norm(candidate.chassis_states[-1, :2] - initial[:2]))
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
                    quality_progress = moved if projected else gain
                    minimum_gain = self.guidance_config.minimum_reference_progress_m
                    if not projected:
                        minimum_gain = min(minimum_gain,
                            max(0., distance - self.guidance_config.reference_goal_radius_m))
                    if (self.guidance_config.minimum_reference_progress_m > 0
                        and not reached and quality_progress + 1e-9 < minimum_gain):
                        rejected_no_progress += 1
                        continue
                    if score > rank + 1e-6 or (best is None and reached):
                        best, rank, selected, progress = (
                            candidate,
                            score,
                            (cap, turn_first, speed_fraction),
                            gain,
                        )
                        displacement = moved
            if best is not None:
                reason = "exact_safe_local_guidance_reference"
            elif self.guidance_config.reference_search_width > 0:
                from robot.safety.checked_guidance_reference_search import search_checked_guidance_reference
                search = search_checked_guidance_reference(self, initial, goal, speed,
                    {k: kwargs[k] for k in ("configuration_space", "h_full", "guidance_fields")}
                    | {"temporal_safety_derivative": kwargs.get("temporal_safety_derivative", 0.)},
                    width=self.guidance_config.reference_search_width,
                    candidate_limit=self.guidance_config.reference_search_maximum_candidates)
                search_audit = search.diagnostics
                best = search.trajectory
                if best is not None:
                    progress = distance - float(np.linalg.norm(goal-best.chassis_states[-1, :2]))
                    displacement = float(np.linalg.norm(best.chassis_states[-1, :2]-initial[:2]))
                    reason = "exact_safe_bounded_guidance_search_reference"
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
        translation_delay = None
        if best is not None:
            translating = np.flatnonzero(best.commands[:, 0] > self.config.intervention_tolerance)
            if len(translating):
                translation_delay = float(translating[0] * self.config.time_step)
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
            minimum_turn_first_rate_cap=self.guidance_config.minimum_turn_first_rate_cap,
            current_guidance_grid_xy=tuple(map(float, current.sampled_guidance_grid_xy)),
            initial_chassis_state_grid=tuple(map(float, initial)),
            first_reference_translation_delay_seconds=translation_delay,
            reference_mode=self.guidance_config.reference_mode,
            reference_displacement_m=displacement,
            reference_progress_basis="endpoint_displacement" if projected else "waypoint_distance_gain",
            reference_uses_temporal_forecast=self.guidance_config.reference_mode == "planar_forecast_velocity",
            reference_projection_fixed_angular_velocity=0.
            if self.guidance_config.reference_mode == "planar_forecast_velocity"
            and self.guidance_config.tangent_velocity_fraction == 0 else None,
            reference_search=search_audit,
            tangent_velocity_fraction=self.guidance_config.tangent_velocity_fraction,
            reference_projection_basis="lookahead_unicycle_affine_v_omega"
            if self.guidance_config.tangent_velocity_fraction > 0 else "legacy_reference_projection",
            reference_tangent_basis=self.guidance_config.reference_tangent_basis,
            minimum_reference_speed_fraction=self.guidance_config.minimum_reference_speed_fraction,
            selected_reference_speed_fraction=selected[2],
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

    def _reference_preference(self, state, goal, speed, sample, field, *, elapsed_seconds=0.):
        if self.guidance_config.reference_mode == "bounded_field_heading":
            return self._desired_heading(state, goal, sample.sampled_guidance_grid_xy), speed
        # v_sem is the spatial CBF normal, not a stand-alone travel heading.
        # Filter a goal-directed holonomic velocity first, then track its
        # direction with the unicycle. This projection is only a reference:
        # every converted horizon still needs full exact unicycle validation.
        delta = goal - state[:2]
        distance = float(np.linalg.norm(delta))
        if distance <= self.guidance_config.reference_goal_radius_m:
            return None
        safety = self._safety_states(state[None, :])[0]
        if self.guidance_config.reference_mode == "planar_forecast_velocity":
            constraint = self.paper_filter.evaluate_constraint(
                query_position_grid_xy=safety[:2], query_yaw_radians=float(safety[2]),
                configuration_space=field["configuration_space"], h_full=field["h_full"],
                guidance_fields=field["guidance_fields"],
                temporal_safety_derivative=field.get("temporal_safety_derivative", 0.))
            # Use the same local first-order forecast as exact horizon
            # evaluation. A reference cannot borrow feasibility from an angular
            # QP output which this planar-direction builder does not apply.
            forecast_h = (constraint.sampled_safety_value
                + elapsed_seconds * constraint.scaled_temporal_safety_derivative)
            if constraint.unsafe_reason is not None or forecast_h < -self.config.constraint_tolerance:
                return None
            c = self.paper_filter.config
            preferred = speed * delta / distance
            fraction = self.guidance_config.tangent_velocity_fraction
            if fraction > 0:
                # The semantic field is a CBF normal. Its component tangent to
                # the actual planar safety gradient can also supply a separate
                # proactive travel preference. Runtime gates this ablation by
                # a current observed-motion passing selection. This vector has
                # no execution authority: retain the projection and every
                # inherited exact unicycle/geometry/analytical check.
                if self.guidance_config.reference_tangent_basis == "selected_observed_obstacle":
                    _, center, side = field["selected_passing_tangents"][0]
                    normal = np.asarray(center)-state[:2]
                    norm = float(np.linalg.norm(normal))
                    if norm <= 1e-9:
                        return None
                    normal = normal / norm
                    tangent = np.array([-normal[1], normal[0]]) * (1 if side == "left" else -1)
                else:
                    gradient = np.asarray(constraint.interpolant_gradient[:2])
                    norm = float(np.linalg.norm(gradient))
                    normal = gradient / max(norm, 1e-9)
                    guide = np.asarray(constraint.sampled_guidance_grid_xy)
                    tangent = guide - (guide @ normal) * normal if norm > 1e-9 else np.zeros(2)
                tangent_norm = float(np.linalg.norm(tangent))
                if tangent_norm > 1e-9:
                    activation = math.sqrt(float(np.clip(1 - constraint.sampled_safety_value
                        / self.guidance_config.activation_safety_value, 0, 1)))
                    fade = min(1., distance / self.guidance_config.guidance_fade_distance_m) ** 2
                    preferred = preferred + speed * fraction * activation * fade * tangent / tangent_norm
                    preferred *= min(1., speed / max(1e-9, float(np.linalg.norm(preferred))))
                magnitude = float(np.linalg.norm(preferred))
                if magnitude <= 1e-9:
                    return None
                # This is only a desired travel vector. The rollout projects
                # the ACTUALLY REQUESTED (v,omega), including angular coupling,
                # instead of discarding feasible turns at a fictitious omega=0.
                return math.atan2(preferred[1], preferred[0]), min(speed, magnitude)
            velocity, feasible = project_weighted_box_halfspace(preferred,
                lower_bounds=np.full(2, c.minimum_planar_velocity),
                upper_bounds=np.full(2, c.maximum_planar_velocity),
                weights=np.full(2, c.planar_velocity_weight),
                constraint_normal=np.array(constraint.sampled_guidance_grid_xy),
                constraint_rhs=-self.config.gamma * max(0., forecast_h) - constraint.scaled_temporal_safety_derivative,
                numerical_tolerance=c.numerical_tolerance)
            magnitude = float(np.linalg.norm(velocity))
            if not feasible or magnitude <= 1e-9:
                return None
            # An online safety intervention may request more than the nominal
            # speed, still within the original actuator bound. Capping a
            # required outward projection at a slow nominal can invalidate it.
            # Conversion still has no safety authority: exact horizon and
            # execution checks remain mandatory, including angular coupling.
            return math.atan2(velocity[1], velocity[0]), min(self.config.maximum_linear_velocity, magnitude)
        result = self.paper_filter.filter_command(speed * delta / distance, 0.,
            query_position_grid_xy=safety[:2], query_yaw_radians=float(safety[2]),
            configuration_space=field["configuration_space"], h_full=field["h_full"],
            guidance_fields=field["guidance_fields"],
            temporal_safety_derivative=field.get("temporal_safety_derivative", 0.))
        velocity = np.asarray(result.safe_planar_velocity_grid_xy)
        magnitude = float(np.linalg.norm(velocity))
        if result.emergency_stop_active or magnitude <= 1e-9:
            return None
        return math.atan2(velocity[1], velocity[0]), min(speed, magnitude)

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
            preference = self._reference_preference(state, goal, speed, sample, field,
                elapsed_seconds=i * self.config.time_step)
            if preference is None:
                return None
            desired, preferred_speed = preference
            error = _wrap(desired - state[2])
            goal_bearing = math.atan2(goal[1] - state[1], goal[0] - state[0])
            radial_alignment = max(math.cos(_wrap(goal_bearing - state[2])), 0.0)
            # The reference may turn in place; reduce translation when the
            # current heading points away from the configured goal.
            # This is reference generation, not a new executable constraint.
            forward = preferred_speed * max(math.cos(error), 0.0)
            if self.guidance_config.reference_mode == "bounded_field_heading":
                forward = min(forward, distance * radial_alignment / self.config.time_step)
            else:
                forward = min(forward, distance / self.config.time_step)
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
            if self.guidance_config.tangent_velocity_fraction > 0:
                constraint = self.paper_filter.evaluate_constraint(
                    query_position_grid_xy=safety[:2], query_yaw_radians=float(safety[2]),
                    configuration_space=field["configuration_space"], h_full=field["h_full"],
                    guidance_fields=field["guidance_fields"],
                    temporal_safety_derivative=field.get("temporal_safety_derivative", 0.))
                forecast_h = constraint.sampled_safety_value + i * self.config.time_step * constraint.scaled_temporal_safety_derivative
                if constraint.unsafe_reason is not None or forecast_h < -self.config.constraint_tolerance:
                    return None
                normal = lookahead_control_matrix(float(safety[2]),
                    lookahead_distance=self.config.lookahead_distance).T @ np.asarray(constraint.constraint_normal)
                commands[i], feasible = project_weighted_box_halfspace(commands[i],
                    lower_bounds=np.array([self.config.minimum_linear_velocity, max(-cap, self.config.minimum_angular_velocity)]),
                    upper_bounds=np.array([self.config.maximum_linear_velocity, min(cap, self.config.maximum_angular_velocity)]),
                    weights=np.array([self.paper_filter.config.planar_velocity_weight, self.paper_filter.config.angular_velocity_weight]),
                    constraint_normal=normal,
                    constraint_rhs=-self.config.gamma * max(0., forecast_h) - constraint.scaled_temporal_safety_derivative,
                    numerical_tolerance=self.paper_filter.config.numerical_tolerance)
                if not feasible:
                    return None
            state, _, _ = self._interval_dynamics(state, commands[i])
        return commands

"""Opt-in static-obstacle planning reserve and checked longer-horizon fallback.

The original MPC/CBF execution gates remain authoritative. The reserve is a
planning heuristic based on actuator bounds over one execution interval, not a
formal reachable-set guarantee. No object label, oracle pose or saved path is
used. The caller supplies the same obstacle-unaware nominal at both horizons.
"""

from dataclasses import dataclass, fields, replace
import itertools

import numpy as np

from robot.safety.laplace_guidance_tracking_mpc import (
    GuidanceTrackingMPCResult,
    LaplaceGuidanceTrackingMPC,
)
from robot.safety.safe_sage_unicycle_cbf_filter import (
    SafeSAGEUnicycleCBFConfig,
    SafeSAGEUnicycleCBFFilter,
)
from robot.safety.safe_sage_unicycle_mpc import _frozen_array


@dataclass(frozen=True)
class DelayRobustBypassConfig:
    fallback_steps: int = 30
    reserve_seconds: float = 0.2

    def __post_init__(self):
        if isinstance(self.fallback_steps, bool) or self.fallback_steps != 30:
            raise ValueError("bounded static fallback requires 30 prediction steps")
        if (
            isinstance(self.reserve_seconds, bool)
            or not 0 < self.reserve_seconds <= 0.2
        ):
            raise ValueError("planning reserve must be finite and in (0, .2] seconds")


@dataclass(frozen=True)
class DelayRobustBypassResult(GuidanceTrackingMPCResult):
    delay_robust_bypass: dict


class DelayRobustBypassMPC(LaplaceGuidanceTrackingMPC):
    """Separate static-controller ablation; legacy/default control is untouched."""

    def __init__(self, paper_filter, config, *, guidance_config, bypass_config=None):
        super().__init__(paper_filter, config, guidance_config=guidance_config)
        self.bypass_config = bypass_config or DelayRobustBypassConfig()
        if config.horizon_steps != 6 or guidance_config.tangent_velocity_fraction != 0:
            raise ValueError(
                "delay-robust bypass is scoped to N6 static/no-human guidance"
            )
        if self.bypass_config.reserve_seconds > config.time_step:
            raise ValueError("reserve must fit the existing execution interval")

    @property
    def nominal_horizon_steps(self):
        return self.bypass_config.fallback_steps

    def _actuator_maximum(self):
        return np.array(
            [
                self.config.maximum_linear_velocity,
                max(
                    abs(self.config.minimum_angular_velocity),
                    abs(self.config.maximum_angular_velocity),
                ),
            ]
        )

    def _physics_step_allowance(self):
        return self._actuator_maximum() * (
            self.config.time_step / self.config.integration_substeps_per_interval
        )

    def _reserve(self, applied_command):
        maximum = self._actuator_maximum()
        if applied_command is None:
            return maximum * self.bypass_config.reserve_seconds
        command = np.asarray(applied_command, dtype=float)
        if (
            command.shape != (2,)
            or not np.isfinite(command).all()
            or not self.config.minimum_linear_velocity
            <= command[0]
            <= self.config.maximum_linear_velocity
            or not self.config.minimum_angular_velocity
            <= command[1]
            <= self.config.maximum_angular_velocity
        ):
            raise ValueError(
                "planning applied command must be finite and within original actuator bounds"
            )
        # Keep the spatial reserve fixed through turning and stopping. Heading
        # delay is checked explicitly at held-input receipt poses; inflating all
        # future headings by the last turn rate makes the feasible set oscillate.
        # The one-step yaw floor covers scheduling drift, not arbitrary motion.
        return np.array(
            [
                maximum[0] * self.bypass_config.reserve_seconds,
                self._physics_step_allowance()[1],
            ]
        )

    def _planner(self, steps, *, reserve, applied_command=None):
        allowances = self._reserve(applied_command) if reserve else np.zeros(2)
        extra_position = allowances[0] if reserve else 0.0
        extra_yaw = allowances[1] if reserve else 0.0
        return LaplaceGuidanceTrackingMPC(
            self.paper_filter,
            replace(
                self.config,
                horizon_steps=steps,
                configuration_position_padding_m=self.config.configuration_position_padding_m
                + extra_position,
                configuration_yaw_padding_radians=self.config.configuration_yaw_padding_radians
                + extra_yaw,
            ),
            guidance_config=self.guidance_config,
        )

    def revalidate_commands(self, initial, commands, **kwargs):
        """Check the ENTIRE returned horizon with original execution margins."""
        commands = np.asarray(commands)
        if commands.shape not in (
            (self.config.horizon_steps, 2),
            (self.nominal_horizon_steps, 2),
        ):
            return False
        return self._planner(len(commands), reserve=False).revalidate_commands(
            initial, commands, **kwargs
        )

    def _selection_cost(self, planner, commands, states, nominal, goal):
        # All candidate families use the same cost, including optimizer results
        # whose internal objective may also contain a guidance tracking term.
        return planner._objective_value(
            commands, nominal
        ) + self.config.position_tracking_weight * float(
            np.sum((states[-1, :2] - goal) ** 2)
        )

    def _delay_poses(self, initial, applied_command):
        """Small held-input receipt-pose probes, never replacement odometry."""
        if applied_command is None:
            return (initial,)
        v, omega = map(float, applied_command)
        duration = self.bypass_config.reserve_seconds
        yaw = initial[2]
        if abs(omega) < 1e-9:
            shift = np.array(
                [v * np.cos(yaw) * duration, v * np.sin(yaw) * duration, 0.0]
            )
        else:
            shift = np.array(
                [
                    v / omega * (np.sin(yaw + omega * duration) - np.sin(yaw)),
                    v / omega * (np.cos(yaw) - np.cos(yaw + omega * duration)),
                    omega * duration,
                ]
            )
        end = initial + shift
        floor = self._physics_step_allowance()
        drift = np.array([floor[0] * np.cos(end[2]), floor[0] * np.sin(end[2]), 0.0])
        turn = np.array([0.0, 0.0, floor[1]])
        return (initial, end, end + drift, end - drift, end + turn, end - turn)

    def _delay_checked(self, poses, commands, field):
        return all(self.revalidate_commands(pose, commands, **field) for pose in poses)

    def _moving_candidates(
        self,
        planner,
        initial,
        nominal,
        goal,
        field,
        template,
        *,
        delay_poses=None,
        stopping_tails=False,
    ):
        """Finite online arc proposals; safety checks, not a speed floor.

        Both directions are searched without fixture/semantic identity. A moving
        candidate has no authority unless its complete reserved horizon and its
        immediate original analytical command pass. Stops remain available.
        """
        speed = min(self.config.maximum_linear_velocity, float(np.max(nominal[:6, 0])))
        backup = SafeSAGEUnicycleCBFFilter(
            self.paper_filter,
            SafeSAGEUnicycleCBFConfig(
                lookahead_distance=self.config.lookahead_distance,
                minimum_linear_velocity=self.config.minimum_linear_velocity,
                maximum_linear_velocity=self.config.maximum_linear_velocity,
                minimum_angular_velocity=self.config.minimum_angular_velocity,
                maximum_angular_velocity=self.config.maximum_angular_velocity,
            ),
        )
        query = planner._safety_states(initial[None, :])[0]
        best = None
        seen = set()
        eligible = []
        for fraction, omega, turning_steps, cutoff in itertools.product(
            (0.5, 0.75, 1.0),
            (-0.8, -0.4, -0.2, 0.0, 0.2, 0.4, 0.8),
            (1, 3, 5, 8),
            (6, 12, 18, 24) if stopping_tails else (30,),
        ):
            if (
                not self.config.minimum_angular_velocity
                <= omega
                <= self.config.maximum_angular_velocity
            ):
                continue
            commands = np.tile(
                (speed * fraction, 0.0), (planner.config.horizon_steps, 1)
            )
            commands[:turning_steps, 1] = omega
            if cutoff < planner.config.horizon_steps:
                # Taper the last two moving inputs and predict a stop for the
                # entire remaining horizon. Never truncate the geometry/CBF test.
                commands[cutoff - 2] *= 2.0 / 3.0
                commands[cutoff - 1] *= 1.0 / 3.0
                commands[cutoff:] = 0.0
            key = commands.tobytes()
            if key in seen:
                continue
            seen.add(key)
            trajectory = planner._evaluate_trajectory(
                initial, commands, **field, require_feasible=True
            )
            if trajectory is None or not planner._trajectory_is_safe(trajectory):
                continue
            filtered = backup.filter_command(
                *commands[0],
                query_position_grid_xy=query[:2],
                query_yaw_radians=float(query[2]),
                **field,
            )
            if (
                filtered.emergency_stop_active
                or not filtered.constraint_satisfied
                or not filtered.constraint_feasible
                or not np.allclose(
                    commands[0],
                    (filtered.safe_linear_velocity, filtered.safe_angular_velocity),
                    atol=1e-9,
                    rtol=0,
                )
            ):
                continue
            score = self._selection_cost(
                planner, commands, trajectory.chassis_states, nominal, goal
            )
            eligible.append((score, commands, trajectory))
        # Exact cost order preserves the selected candidate while avoiding
        # expensive receipt checks for candidates that cannot improve it.
        receipt_checked = 0
        for score, commands, trajectory in sorted(eligible, key=lambda row: row[0]):
            receipt_checked += 1
            if delay_poses is None or self._delay_checked(delay_poses, commands, field):
                candidate = replace(
                    template,
                    guidance_tracking=replace(
                        template.guidance_tracking,
                        active=False,
                        reason="separate_checked_static_arc_search",
                    ),
                    command_sequence=_frozen_array(commands),
                    predicted_chassis_states=_frozen_array(trajectory.chassis_states),
                    predicted_safety_states=_frozen_array(trajectory.safety_states),
                    predicted_safety_values=_frozen_array(trajectory.safety_values),
                    dtcbf_margins=_frozen_array(trajectory.margins),
                    nominal_first_command=tuple(map(float, nominal[0])),
                    objective_value=planner._objective_value(commands, nominal),
                    success=True,
                    intervention_active=True,
                    fallback_active=False,
                    restoration_used=True,
                    sqp_iteration_count=0,
                    solver_iteration_count=0,
                    status="checked_static_moving_candidate_selected",
                )
                best = (score, "moving_arc", candidate, True)
                break
        return best, dict(
            tested=len(seen),
            stopping_tails=stopping_tails,
            reserved_geometry_and_cbf_passed=len(eligible),
            receipt_checked=receipt_checked,
            passed=int(best is not None),
            best_first_command=None if best is None else best[2].first_command,
            basis="Symmetric continuous and tapered-to-stop arcs; complete 30-step reserved geometry/DTCBF and receipt probes plus original first-command CBF; no mandatory minimum speed",
        )

    def plan(
        self,
        initial_chassis_state_grid,
        nominal_command,
        *,
        nominal_goal_grid_xy=None,
        initial_command_guess=None,
        planning_applied_command=None,
        **kwargs,
    ):
        # Scope validation prevents implicit application to the human experiments.
        if (
            self.predictive_geometry is not None
            or kwargs.get("selected_passing_tangents")
            or kwargs.get("observed_motion_forecast") is not None
        ):
            raise ValueError(
                "static fallback cannot consume a human passing/prediction policy"
            )
        initial = np.asarray(initial_chassis_state_grid, dtype=float)
        nominal = np.asarray(nominal_command, dtype=float)
        goal = np.asarray(nominal_goal_grid_xy, dtype=float)
        if (
            nominal.shape != (self.nominal_horizon_steps, 2)
            or not np.isfinite(nominal).all()
        ):
            raise ValueError("caller must supply the full pose-only 30-step nominal")
        if (
            initial.shape != (3,)
            or goal.shape != (2,)
            or not np.isfinite(initial).all()
            or not np.isfinite(goal).all()
        ):
            raise ValueError("finite current state and nominal goal required")
        warm = (
            None
            if initial_command_guess is None
            else np.asarray(initial_command_guess, dtype=float)
        )
        if warm is not None and (
            warm.shape not in ((6, 2), (30, 2)) or not np.isfinite(warm).all()
        ):
            raise ValueError(
                "warm horizon must be finite and match a published horizon"
            )
        short = self._planner(6, reserve=True, applied_command=planning_applied_command)
        common = dict(kwargs, nominal_goal_grid_xy=goal)
        result = short.plan(
            initial,
            nominal[:6],
            initial_command_guess=None if warm is None else warm[:6],
            **common,
        )
        original_short_reason = result.guidance_tracking.reason
        continuing = warm is not None and len(warm) == self.nominal_horizon_steps
        long = self._planner(
            self.nominal_horizon_steps,
            reserve=True,
            applied_command=planning_applied_command,
        )
        field = {
            key: kwargs[key]
            for key in (
                "configuration_space",
                "h_full",
                "guidance_fields",
                "temporal_safety_derivative",
            )
            if key in kwargs
        }
        direct_safe = (
            long.revalidate_commands(initial, nominal, **field) if continuing else None
        )
        delay_poses = self._delay_poses(initial, planning_applied_command)
        short_progress = float(
            np.linalg.norm(initial[:2] - goal)
            - np.linalg.norm(result.predicted_chassis_states[-1, :2] - goal)
        )
        short_moves = (
            result.success
            and not result.fallback_active
            and result.linear_velocity >= 0.02
            and short_progress >= self.guidance_config.minimum_reference_progress_m
            and self._delay_checked(delay_poses, result.command_sequence, field)
        )
        # An absent preferred reference triggers a longer search, but a checked
        # moving short result remains available if the search can only stop.
        trigger = (
            not result.success
            or not result.guidance_tracking.active
            or nominal[0, 0] > 0.05
            and result.linear_velocity < 0.02
        )
        use_long = trigger or continuing and not direct_safe
        attempts = []
        moving_search = None
        checked_continuation = False
        short_progress_preferred = False
        stopping_search = None
        repaired_stall = False
        stopping_search_reason = None
        if use_long:

            def optimized(label, guess):
                candidate = long.plan(
                    initial, nominal, initial_command_guess=guess, **common
                )
                valid = (
                    candidate.success
                    and not candidate.fallback_active
                    and self._delay_checked(
                        delay_poses, candidate.command_sequence, field
                    )
                )
                score = (
                    self._selection_cost(
                        long,
                        candidate.command_sequence,
                        candidate.predicted_chassis_states,
                        nominal,
                        goal,
                    )
                    if valid
                    else float("inf")
                )
                attempts.append((score, label, candidate, valid))
                return candidate, valid

            def advances(candidate):
                return (
                    np.linalg.norm(initial[:2] - goal)
                    - np.linalg.norm(candidate.predicted_chassis_states[-1, :2] - goal)
                    >= self.guidance_config.minimum_reference_progress_m
                )

            if continuing:
                candidate, valid = optimized("shifted", warm)
                checked_continuation = bool(
                    valid and candidate.linear_velocity >= 0.05 and advances(candidate)
                )
                repeated_low_progress = bool(
                    planning_applied_command is not None
                    and nominal[0, 0] > 0.05
                    and abs(planning_applied_command[0]) < 0.02
                    and candidate.linear_velocity < 0.02
                )
                if valid and (not advances(candidate) or repeated_low_progress):
                    stopping_search_reason = (
                        "no_horizon_goal_progress"
                        if not advances(candidate)
                        else "repeated_low_forward_speed"
                    )
                    moving, stopping_search = self._moving_candidates(
                        long,
                        initial,
                        nominal,
                        goal,
                        field,
                        result,
                        delay_poses=delay_poses,
                        stopping_tails=True,
                    )
                    if moving is not None and advances(moving[2]):
                        attempts.append(moving)
                        repaired_stall = True
                    elif short_moves:
                        short_progress_preferred = True
            if not (checked_continuation or repaired_stall or short_progress_preferred):
                optimized("cold", None)
                moving, moving_search = self._moving_candidates(
                    long, initial, nominal, goal, field, result, delay_poses=delay_poses
                )
                if moving is not None:
                    attempts.append(moving)
                if stopping_search is None and not any(
                    row[3] and advances(row[2]) for row in attempts
                ):
                    stopping_search_reason = "no_advancing_long_candidate"
                    moving, stopping_search = self._moving_candidates(
                        long,
                        initial,
                        nominal,
                        goal,
                        field,
                        result,
                        delay_poses=delay_poses,
                        stopping_tails=True,
                    )
                    if moving is not None and advances(moving[2]):
                        attempts.append(moving)
                        repaired_stall = True
            passing = [row for row in attempts if row[3]]
            progressing = [
                row
                for row in passing
                if np.linalg.norm(initial[:2] - goal)
                - np.linalg.norm(row[2].predicted_chassis_states[-1, :2] - goal)
                >= self.guidance_config.minimum_reference_progress_m
            ]
            if repaired_stall:
                # The current and proposed commands were stalled, or the
                # original horizon had no goal progress. Prefer the checked
                # moving repair even if progress deferred to its tail would
                # have a slightly lower nominal tracking cost.
                result = moving[2]
            elif short_moves and not progressing:
                short_progress_preferred = True
                use_long = False
            elif passing:
                if progressing:
                    passing = progressing
                result = min(passing, key=lambda row: row[0])[2]
            elif result.success and not result.fallback_active:
                # Retain a checked short result; no invalid long trajectory gets
                # an execution exception and the next request retries online.
                use_long = False
        diagnostic = dict(
            enabled=True,
            short_reference_reason=original_short_reason,
            triggered=trigger,
            continuing_long_horizon=continuing,
            direct_long_nominal_safe=direct_safe,
            selected_steps=len(result.command_sequence),
            moving_search=moving_search,
            stopping_tail_search=stopping_search,
            stopping_tail_repaired_stall=repaired_stall,
            stopping_tail_search_reason=stopping_search_reason,
            checked_moving_continuation=checked_continuation,
            checked_short_progress=short_moves,
            short_progress_preferred=short_progress_preferred,
            reserve_seconds=self.bypass_config.reserve_seconds,
            receipt_pose_probe_count=len(delay_poses),
            applied_command=None
            if planning_applied_command is None
            else list(map(float, planning_applied_command)),
            additional_position_reserve_m=float(
                self._reserve(planning_applied_command)[0]
            ),
            additional_yaw_reserve_radians=float(
                self._reserve(planning_applied_command)[1]
            ),
            attempts=[
                dict(
                    start=label,
                    success=valid,
                    status=row.status,
                    first_command=row.first_command,
                    selection_cost=score if np.isfinite(score) else None,
                )
                for score, label, row, valid in attempts
            ],
            scope="Static online ablation; bounded planning reserve, original execution gates, no oracle/identity/phi change",
        )
        return DelayRobustBypassResult(
            **{
                f.name: getattr(result, f.name)
                for f in fields(GuidanceTrackingMPCResult)
            },
            delay_robust_bypass=diagnostic,
        )

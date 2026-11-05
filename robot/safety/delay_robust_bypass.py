"""Opt-in static-obstacle planning reserve and checked longer-horizon fallback.

The original MPC/CBF execution gates remain authoritative. The reserve is a
planning heuristic based on actuator bounds over one execution interval, not a
formal reachable-set guarantee. No object label, oracle pose or saved path is
used. The caller supplies the same obstacle-unaware nominal at both horizons.
"""

from dataclasses import dataclass, fields, replace

import numpy as np

from robot.safety.laplace_guidance_tracking_mpc import (
    GuidanceTrackingMPCResult,
    LaplaceGuidanceTrackingMPC,
)


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

    def _planner(self, steps, *, reserve):
        extra_position = (
            self.config.maximum_linear_velocity * self.bypass_config.reserve_seconds
            if reserve
            else 0.0
        )
        extra_yaw = (
            max(
                abs(self.config.minimum_angular_velocity),
                abs(self.config.maximum_angular_velocity),
            )
            * self.bypass_config.reserve_seconds
            if reserve
            else 0.0
        )
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

    def plan(
        self,
        initial_chassis_state_grid,
        nominal_command,
        *,
        nominal_goal_grid_xy=None,
        initial_command_guess=None,
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
        short = self._planner(6, reserve=True)
        common = dict(kwargs, nominal_goal_grid_xy=goal)
        result = short.plan(
            initial,
            nominal[:6],
            initial_command_guess=None if warm is None else warm[:6],
            **common,
        )
        original_short_reason = result.guidance_tracking.reason
        trigger = (
            not result.success
            or not result.guidance_tracking.active
            or nominal[0, 0] > 0.05
            and result.linear_velocity < 0.02
        )
        continuing = warm is not None and len(warm) == self.nominal_horizon_steps
        long = self._planner(self.nominal_horizon_steps, reserve=True)
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
        use_long = trigger or continuing and not direct_safe
        attempts = []
        if use_long:
            for label, guess in (
                ("cold", None),
                ("shifted", warm if continuing else None),
            ):
                if label == "shifted" and not continuing:
                    continue
                candidate = long.plan(
                    initial, nominal, initial_command_guess=guess, **common
                )
                valid = (
                    candidate.success
                    and not candidate.fallback_active
                    and self.revalidate_commands(
                        initial, candidate.command_sequence, **field
                    )
                )
                score = float("inf")
                if valid:
                    # Explicit goal-progress selection between complete checked
                    # optimizer candidates, with the existing state-cost weight.
                    score = (
                        candidate.objective_value
                        + self.config.position_tracking_weight
                        * float(
                            np.sum(
                                (candidate.predicted_chassis_states[-1, :2] - goal) ** 2
                            )
                        )
                    )
                attempts.append((score, label, candidate, valid))
            passing = [row for row in attempts if row[3]]
            if passing:
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
            reserve_seconds=self.bypass_config.reserve_seconds,
            additional_position_reserve_m=self.config.maximum_linear_velocity
            * self.bypass_config.reserve_seconds,
            additional_yaw_reserve_radians=max(
                abs(self.config.minimum_angular_velocity),
                abs(self.config.maximum_angular_velocity),
            )
            * self.bypass_config.reserve_seconds,
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

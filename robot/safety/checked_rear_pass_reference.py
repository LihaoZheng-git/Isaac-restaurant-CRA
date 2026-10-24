"""Bounded turn/translate reference synthesis for a separate rear-pass ablation.

The caller must supply a current verified crossing and original field/source
context. This component neither creates identity nor publishes wheel commands.
"""

from dataclasses import dataclass, asdict
from itertools import product
import math
import time

import numpy as np

from robot.control.unicycle_model import lookahead_control_matrix, wrap_to_pi
from robot.safety.encounter_motion import validated_route_direction


@dataclass(frozen=True)
class CheckedRearPassReferenceConfig:
    offset_m: float = 0.7
    lookahead_m: float = 1.0
    goal_fade_distance_m: float = 0.8
    goal_radius_m: float = 0.1
    minimum_lead_m: float = 1.4
    existing_trailing_offset_m: float = 0.15
    existing_turn_heading_radians: float | None = None
    release_lateral_m: float = 1.2
    heading_gain_m_per_radian: float = 0.4
    lateral_gain: float = 0.1
    minimum_translation_m: float = 0.01
    candidate_limit: int = 576
    minimum_first_translation_m: float = 0.0
    allow_turn_first_fallback: bool = False

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key == "allow_turn_first_fallback":
                if not isinstance(value, bool):
                    raise ValueError("turn-first fallback must be an explicit boolean")
                continue
            if key == "existing_turn_heading_radians" and value is None:
                continue
            if key == "minimum_first_translation_m" and not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value == 0.0:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{key} must be finite and positive")
        if (
            not isinstance(self.candidate_limit, int)
            or self.candidate_limit > 576
            or self.goal_fade_distance_m <= self.goal_radius_m
        ):
            raise ValueError(
                "candidate budget must be an integer <=576; fade exceeds goal radius"
            )
        if (
            self.existing_turn_heading_radians is not None
            and self.existing_turn_heading_radians >= math.pi / 2
        ):
            raise ValueError("existing trailing turn threshold must be below pi/2")


def rear_pass_context(initial, goal, obstacle, velocity, route_direction, config=None):
    """Engineering behavior gates only; these distances are not safety evidence."""
    cfg = config or CheckedRearPassReferenceConfig()
    if not isinstance(cfg, CheckedRearPassReferenceConfig):
        raise TypeError("checked rear-pass configuration required")
    initial = np.asarray(initial, float)
    vectors = [np.asarray(x, float) for x in (goal, obstacle, velocity)]
    if (
        initial.shape != (3,)
        or not np.isfinite(initial).all()
        or any(x.shape != (2,) or not np.isfinite(x).all() for x in vectors)
    ):
        raise ValueError("finite current state/goal/position/velocity required")
    goal, obstacle, velocity = vectors
    route = validated_route_direction(route_direction)
    speed = float(np.linalg.norm(velocity))
    transverse = velocity - (velocity @ route) * route
    transverse_speed = float(np.linalg.norm(transverse))
    result = dict(
        mode="none",
        reason="motion_outside_crossing_scope",
        configuration=asdict(cfg),
        scope="configured maneuver preference only; not physical risk, free space or a safety margin",
    )
    if not 0.2 <= speed <= 2.0 or transverse_speed < 0.6 * speed:
        return result
    rear = -transverse / transverse_speed
    distance = float(np.linalg.norm(goal - initial[:2]))
    lateral = float((obstacle - goal) @ (-rear))
    lead = float((obstacle - initial[:2]) @ route)
    offset = float((initial[:2] - goal) @ rear)
    heading = np.array([math.cos(initial[2]), math.sin(initial[2])])
    trailing_heading = math.atan2(float(heading @ rear), float(heading @ route))
    continuing_turn = (
        cfg.existing_turn_heading_radians is not None
        and cfg.existing_turn_heading_radians <= trailing_heading < math.pi / 2
    )
    result.update(
        current_lead_m=lead,
        current_trailing_offset_m=offset,
        current_human_lateral_progress_m=lateral,
        current_trailing_heading_radians=trailing_heading,
        continuing_existing_trailing_turn=continuing_turn,
        rear_direction=rear.tolist(),
    )
    if distance <= cfg.goal_radius_m:
        result["reason"] = "nominal_waypoint_region"
        return result
    if lateral >= cfg.release_lateral_m:
        result["reason"] = "current_human_has_cleared_behavior_window"
        return result
    if (
        lead < cfg.minimum_lead_m
        and offset < cfg.existing_trailing_offset_m
        and not continuing_turn
    ):
        result.update(mode="wait", reason="insufficient_maneuver_lead")
        return result
    along = float((goal - initial[:2]) @ route)
    target = goal.copy()
    if along > 0:
        target += route * (min(along, cfg.lookahead_m) - along)
    target += cfg.offset_m * min(1.0, distance / cfg.goal_fade_distance_m) ** 2 * rear
    result.update(
        mode="rear",
        reason="continue_existing_trailing_turn"
        if continuing_turn
        else "current_crossing_with_maneuver_room",
        target_grid_xy=target.tolist(),
    )
    return result


@dataclass(frozen=True)
class CheckedRearPassReferenceResult:
    trajectory: object | None
    diagnostics: dict


def search_checked_rear_pass_reference(
    mpc,
    initial,
    goal,
    obstacle,
    velocity,
    route_direction,
    field,
    check_first_command,
    *,
    config=None,
):
    cfg = config or CheckedRearPassReferenceConfig()
    context = rear_pass_context(initial, goal, obstacle, velocity, route_direction, cfg)
    counts = dict(
        tested=0,
        exact_feasible=0,
        future_analytical_rejections=0,
        translation_rejections=0,
        first_translation_rejections=0,
        first_command_checks=0,
        turn_first_fallback_candidates=0,
    )
    audit = dict(
        context=context,
        reason=context["reason"],
        counts=counts,
        selected=False,
        candidate_limit=cfg.candidate_limit,
        scope="complete exact/dense/forecast-analytical reference only; current identity, original source clocks, live revalidation and execution remain caller gates",
    )
    if context["mode"] != "rear":
        return CheckedRearPassReferenceResult(None, audit)
    if mpc.config.minimum_linear_velocity != 0.0:
        raise ValueError(
            "rear-pass reference is scoped to the forward-only actuator model"
        )
    started = time.perf_counter()
    initial = np.asarray(initial, float)
    target = np.asarray(context["target_grid_xy"])
    rear = np.asarray(context["rear_direction"])
    delta = target - initial[:2]
    angle = math.atan2(delta[1], delta[0])
    error = abs(wrap_to_pi(angle - initial[2]))
    turn = float(np.sign(wrap_to_pi(angle - initial[2])))
    angular_limit = (
        mpc.config.maximum_angular_velocity
        if turn >= 0
        else abs(mpc.config.minimum_angular_velocity)
    )
    best, rank = None, 0.0
    turn_first_best, turn_first_rank = None, 0.0
    pool = product(
        (0.2, 0.4, 0.8, 1.0),
        (0.25, 0.5, 0.75, 1.0),
        range(1, min(6, mpc.config.horizon_steps) + 1),
        (0.0, 0.5, -1.0),
        (0.0, 1.0),
    )
    for speed_fraction, rate_fraction, split, tail, turn_fraction in pool:
        if counts["tested"] >= cfg.candidate_limit:
            break
        counts["tested"] += 1
        speed = speed_fraction * mpc.config.maximum_linear_velocity
        rate = turn * rate_fraction * angular_limit
        commands = np.tile([speed, rate * tail], (mpc.config.horizon_steps, 1))
        commands[:split, 1] = rate
        commands[:split, 0] *= turn_fraction
        value = mpc._evaluate_trajectory(
            initial, commands, require_feasible=True, **field
        )
        if value is None or not mpc._trajectory_is_safe(value):
            continue
        counts["exact_feasible"] += 1
        future_ok = True
        for j in range(mpc.config.horizon_steps):
            normal = (
                lookahead_control_matrix(
                    value.safety_states[j, 2],
                    lookahead_distance=mpc.config.lookahead_distance,
                ).T
                @ value.samples.guidance_normals[j]
            )
            rhs = (
                -mpc.paper_filter.config.gamma * max(0.0, value.safety_values[j])
                - value.samples.scaled_temporal_derivatives[j]
            )
            if (
                normal @ value.commands[j]
                < rhs - mpc.paper_filter.config.numerical_tolerance
            ):
                future_ok = False
                break
        if not future_ok:
            counts["future_analytical_rejections"] += 1
            continue
        end = value.chassis_states[-1]
        displacement = end[:2] - initial[:2]
        if np.linalg.norm(displacement) < cfg.minimum_translation_m:
            counts["translation_rejections"] += 1
            continue
        # Optional reference-quality rule, never an executable velocity floor:
        # unavailable translating references retain primary/wait safety behavior.
        first_displacement = value.chassis_states[1, :2] - initial[:2]
        first_quality = np.linalg.norm(first_displacement) + 1e-12 >= cfg.minimum_first_translation_m
        if not first_quality:
            counts["first_translation_rejections"] += 1
            if not cfg.allow_turn_first_fallback:
                continue
            counts["turn_first_fallback_candidates"] += 1
        gain = float(np.linalg.norm(delta) - np.linalg.norm(target - end[:2]))
        end_angle = math.atan2(target[1] - end[1], target[0] - end[0])
        heading = error - abs(wrap_to_pi(end_angle - end[2]))
        utility = (
            gain
            + cfg.heading_gain_m_per_radian * heading
            + cfg.lateral_gain * float(displacement @ rear)
        )
        candidate_rank = rank if first_quality else turn_first_rank
        if utility <= candidate_rank + 1e-9:
            continue
        counts["first_command_checks"] += 1
        filtered = check_first_command(value.commands[0])
        if filtered.emergency_stop_active or not np.allclose(
            value.commands[0],
            [filtered.safe_linear_velocity, filtered.safe_angular_velocity],
            rtol=0,
            atol=1e-6,
        ):
            continue
        if first_quality:
            best, rank = value, utility
        else:
            turn_first_best, turn_first_rank = value, utility
    selection_tier = "translating_first" if best is not None else None
    if best is None and turn_first_best is not None:
        best, rank = turn_first_best, turn_first_rank
        selection_tier = "checked_turn_first_fallback"
    audit.update(
        selected=best is not None,
        reason="checked_reference_selected"
        if best is not None
        else "no_complete_checked_reference",
        selected_utility=rank if best is not None else None,
        selection_tier=selection_tier,
        budget_exhausted=counts["tested"] >= cfg.candidate_limit,
        wall_seconds=time.perf_counter() - started,
    )
    return CheckedRearPassReferenceResult(best, audit)

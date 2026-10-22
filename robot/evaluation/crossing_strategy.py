"""Explicit crossing ablation policy and independent strategy-specific scoring."""

from dataclasses import replace

import numpy as np

from robot.safety.safe_sage_guidance_policy import SafeSAGEGuidancePolicy


def zero_tangential_bias(policy, applied, audit):
    """Copy applied and fallback policies; retain normal flux and physical CRA.

    Only the explicitly selected wait-and-go experiment calls this helper.
    The physical policies supplied to CRA and their fixed lambda stay intact.
    """
    copied = SafeSAGEGuidancePolicy(
        {name: replace(value, tangential_strength=0.)
         for name, value in policy.class_parameters.items()},
        unknown_class=policy.unknown_class, risk_flux_mapping=policy.risk_flux_mapping)
    parameters = {key: replace(value, tangential_strength=0.) for key, value in applied.items()}
    rows = [dict(row, physical_CRA_lambda=row["lambda"], **{
        "lambda": 0., "tangential_bias_ablation": "yield_then_go_zero_lambda"}) for row in audit]
    return copied, parameters, rows


def summarize_crossing_strategy(legacy, truth, *, strategy, nominal_goal_world_xy):
    """Score the first crossing of the human's walking plane, not closest wait.

    Actor positions/phase are evaluation-only. No metric changes old results.
    Arrival, clearance, stopping and current human perception are separate
    mandatory demo gates. This report cannot turn a stopped route into a pass.
    """
    if strategy not in {"rear-pass-or-yield", "yield-then-go"}:
        raise ValueError("unknown crossing strategy")
    times = np.array([row["time"] for row in truth], dtype=float)
    robot = np.array([row["robot_xy"] for row in truth], dtype=float)
    actors = np.array([row["actor_xyz"][:2] for row in truth], dtype=float)
    if len(times) < 3 or np.any(np.diff(times) <= 0) or not all(
        np.isfinite(value).all() for value in (times, robot, actors)
    ):
        raise ValueError("finite ordered independent trace required")
    goal = np.asarray(nominal_goal_world_xy, dtype=float)
    if goal.shape != (2,) or not np.isfinite(goal).all():
        raise ValueError("finite nominal goal required")
    forward = goal - robot[0]
    distance = np.linalg.norm(forward)
    if distance < 1e-9:
        raise ValueError("nonzero route chord required")
    forward /= distance
    phases = np.array([row.get("phase") for row in truth])
    if not set(phases) <= {"pickup", "idle_before", "move", "stopped_after", "aborted"}:
        raise ValueError("complete recognized walking-phase metadata required")
    walking = phases == "move"
    velocity = np.gradient(actors, times, axis=0)
    walking &= np.linalg.norm(velocity, axis=1) >= .3
    checks = {key: value for key, value in legacy["checks"].items() if key not in {
        "visible_trailing_side_arc", "passes_behind_moving_actor"}}
    checks["independent_walking_window"] = bool(walking.any())
    projection, crossing_time, excursion = None, None, 0.
    if walking.any():
        motion = np.median(velocity[walking], axis=0)
        motion /= max(np.linalg.norm(motion), 1e-9)
        plane = float(np.median(actors[walking] @ forward))
        progress = robot @ forward - plane
        candidates = np.flatnonzero((progress[:-1] < 0) & (progress[1:] >= 0))
        if len(candidates):
            i = candidates[0]
            fraction = -progress[i] / (progress[i + 1] - progress[i])
            crossing_time = float(times[i] + fraction * (times[i + 1] - times[i]))
            r = robot[i] + fraction * (robot[i + 1] - robot[i])
            a = actors[i] + fraction * (actors[i + 1] - actors[i])
            projection = float((r - a) @ motion)
        # Mirror the trailing lateral side from independent walking direction.
        lateral_axis = np.array([-forward[1], forward[0]])
        trailing_sign = -np.sign(motion @ lateral_axis)
        active = walking & (np.linalg.norm(robot - actors, axis=1) <= 2.)
        if active.any():
            excursion = float(np.max((robot[active] - robot[0]) @ lateral_axis * trailing_sign))
    checks["crossed_walking_plane_behind_actor"] = projection is not None and projection <= -.10
    if strategy == "rear-pass-or-yield":
        checks["visible_trailing_side_arc"] = excursion >= .20
    return dict(strategy=strategy, checks=checks, passed=all(checks.values()),
        behind_projection_at_route_crossing_m=projection,
        route_crossing_timestamp_seconds=crossing_time,
        maximum_trailing_excursion_during_walking_m=excursion,
        legacy_passing_passed=legacy["passed"],
        scope="separate declared crossing-plane evaluation; wait-and-go does not claim a curved bypass; independent actor truth never enters control")

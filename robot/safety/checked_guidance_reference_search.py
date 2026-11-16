"""Bounded candidate search for the separate guidance-tracking ablation.

Every prefix retains exact DTCBF/dense geometry and mapped analytical guidance
checks. Only a complete original-length horizon may become a reference. The
caller still solves, revalidates and filters with the original source clocks.
"""

import copy
from dataclasses import dataclass, replace
import itertools
import math
import time

import numpy as np

from robot.control.unicycle_model import lookahead_control_matrix


@dataclass(frozen=True)
class CheckedGuidanceSearchResult:
    trajectory: object | None
    diagnostics: dict


def reference_motion_rejection(value, mpc, goal, *, terminal=False):
    """Reference quality only: safety waits remain available to the primary.

    Turn-first references may rotate at the existing useful-turn cap, then
    translate. A delayed idle interval or move-then-stop dead end must not
    attract the receding-horizon objective. Stopping at the goal is valid.
    """
    minimum = mpc.guidance_config.minimum_reference_progress_m
    if minimum <= 0:
        return None
    states = value.chassis_states
    n = min(2, len(value.commands))
    start, end = states[-n-1], states[-1]
    radius = mpc.guidance_config.reference_goal_radius_m
    at_goal = np.linalg.norm(end[:2]-goal) <= radius + 1e-9
    if at_goal and np.linalg.norm(start[:2]-goal) <= radius + 1e-9:
        return None
    moved = float(np.linalg.norm(end[:2]-start[:2]))
    if terminal and not at_goal and moved + 1e-9 < minimum:
        return "stationary_terminal_reference_outside_goal"
    turned = abs(math.atan2(math.sin(end[2]-start[2]), math.cos(end[2]-start[2])))
    useful_turn = mpc.guidance_config.minimum_turn_first_rate_cap * n * mpc.config.time_step
    # The first one-step window uses the same progress rate as the two-step
    # window. Pure rotations retain the existing turn-first cap.
    if moved + 1e-9 < minimum*n/2 and turned + 1e-9 < useful_turn:
        return "postponed_reference_motion_outside_goal"
    return None


def search_checked_guidance_reference(mpc, initial, goal, speed, field, *, width, candidate_limit):
    if (isinstance(width, bool) or not isinstance(width, int) or not 1 <= width <= 8
        or isinstance(candidate_limit, bool) or not isinstance(candidate_limit, int)
        or candidate_limit <= 0):
        raise ValueError("bounded reference search needs integer width 1..8 and positive candidate limit")
    initial, goal = np.asarray(initial, float), np.asarray(goal, float)
    if (initial.shape != (3,) or goal.shape != (2,) or not np.isfinite(initial).all()
        or not np.isfinite(goal).all() or not math.isfinite(speed) or speed <= 0):
        raise ValueError("finite state/goal and positive reference speed required")
    started = time.perf_counter()
    grid = field["configuration_space"].occupancy_grids[0].config
    position_separation = 1.2 * grid.resolution
    yaw_separation = min(.3, 2 * mpc.config.time_step * max(
        abs(mpc.config.minimum_angular_velocity), abs(mpc.config.maximum_angular_velocity)))
    speeds = np.unique(np.clip([0., speed/2, speed],
        mpc.config.minimum_linear_velocity, mpc.config.maximum_linear_velocity))
    turn_rates = np.unique([mpc.config.minimum_angular_velocity,
        mpc.config.minimum_angular_velocity/2, 0., mpc.config.maximum_angular_velocity/2,
        mpc.config.maximum_angular_velocity])
    translating_rates = np.unique([*turn_rates,
        -min(.1, abs(mpc.config.minimum_angular_velocity)/4),
        min(.1, abs(mpc.config.maximum_angular_velocity)/4)])
    pool = tuple((v, rate) for v in speeds for rate in
                 (turn_rates if v == 0 else translating_rates))
    beams = [np.empty((0, 2))]
    distance = float(np.linalg.norm(goal-initial[:2]))
    goal_direction = (goal-initial[:2]) / max(distance, 1e-12)
    goal_lateral = np.array([-goal_direction[1], goal_direction[0]])
    guidance = np.asarray(mpc._sample(initial, field).sampled_guidance_grid_xy)
    lateral_preference = float(guidance @ goal_lateral / max(np.linalg.norm(guidance), 1e-12))
    travel_scale = max(speed*mpc.config.horizon_steps*mpc.config.time_step, grid.resolution)
    tested = exact_feasible = semantic_rejections = quality_rejections = depth_done = 0
    full_candidates = []
    truncated = False
    for depth in range(1, mpc.config.horizon_steps+1):
        partial = copy.copy(mpc)
        partial.config = replace(mpc.config, horizon_steps=depth)
        candidates = []
        for prefix, command in itertools.product(beams, pool):
            if tested >= candidate_limit:
                truncated = True
                break
            tested += 1
            commands = np.vstack((prefix, command))
            value = partial._evaluate_trajectory(initial, commands, require_feasible=True, **field)
            if value is None or not partial._trajectory_is_safe(value):
                continue
            exact_feasible += 1
            # Forecast time remains relative to the ORIGINAL initial state.
            # Earlier intervals already passed on their own exact prefixes.
            normal = lookahead_control_matrix(value.chassis_states[-2, 2],
                lookahead_distance=mpc.config.lookahead_distance).T @ value.samples.guidance_normals[-2]
            rhs = (-mpc.paper_filter.config.gamma * max(0., value.safety_values[-2])
                - value.samples.scaled_temporal_derivatives[-2])
            if (not np.isfinite(normal).all() or not math.isfinite(rhs)
                or normal @ commands[-1] < rhs - mpc.paper_filter.config.numerical_tolerance):
                semantic_rejections += 1
                continue
            if reference_motion_rejection(value, mpc, goal,
                terminal=depth == mpc.config.horizon_steps) is not None:
                quality_rejections += 1
                continue
            state = value.chassis_states[-1]
            error = math.atan2(goal[1]-state[1], goal[0]-state[0])-state[2]
            score = distance - np.linalg.norm(goal-state[:2]) - mpc.config.lookahead_distance * abs(
                math.atan2(math.sin(error), math.cos(error)))
            # Use only the field's lateral component relative to the goal;
            # treating the complete CBF normal as a travel heading can flee
            # backwards. This bounded SOFT ranking term is not a side constraint.
            score += (mpc.config.lookahead_distance * lateral_preference
                * float((state[:2]-initial[:2]) @ goal_lateral) / travel_scale)
            candidates.append((float(score), state, commands, value))
        candidates.sort(key=lambda item: item[0], reverse=True)
        if depth == mpc.config.horizon_steps:
            full_candidates = candidates
        if not candidates:
            break
        depth_done = depth
        # Preserve actually different left/right exploration even before its
        # small lateral displacement exceeds one cell. Pure score pruning can
        # otherwise keep only straight approaches ending at the same obstacle.
        direction = np.array([-math.sin(initial[2]), math.cos(initial[2])])
        def lateral_key(item):
            yaw = math.atan2(math.sin(item[1][2]-initial[2]), math.cos(item[1][2]-initial[2]))
            return float((item[1][:2]-initial[:2]) @ direction), yaw, item[0]
        def negative_lateral_key(item):
            lateral, yaw, score = lateral_key(item)
            return -lateral, -yaw, score
        reserved = [candidates[0], max(candidates, key=lateral_key), max(candidates, key=negative_lateral_key)]
        winners = []
        for item in reserved:
            if not any(np.allclose(item[1], other[1], rtol=0, atol=1e-12) for other in winners):
                winners.append(item)
            if len(winners) == width:
                break
        for item in candidates:
            if len(winners) == width:
                break
            if all(np.linalg.norm(item[1][:2]-other[1][:2]) >= position_separation
                or abs(math.atan2(math.sin(item[1][2]-other[1][2]),
                    math.cos(item[1][2]-other[1][2]))) >= yaw_separation for other in winners):
                winners.append(item)
            if len(winners) == width:
                break
        beams = [item[2] for item in winners]
        if truncated:
            break
    selected = None
    for _, state, _, value in full_candidates:
        if (np.linalg.norm(state[:2]-initial[:2]) + 1e-9 >= mpc.guidance_config.minimum_reference_progress_m
            or np.linalg.norm(goal-state[:2]) <= mpc.guidance_config.reference_goal_radius_m):
            selected = value
            break
    return CheckedGuidanceSearchResult(selected, dict(
        basis="bounded_exact_and_analytical_guidance_constrained_reference_search_ablation",
        width=width, candidate_limit=candidate_limit, tested_candidates=tested,
        exact_feasible_prefixes=exact_feasible, semantic_rejections=semantic_rejections,
        quality_rejections=quality_rejections,
        quality_basis="continuous_reference_activity_and_translating_tail_or_goal; primary may still wait",
        completed_depth=depth_done, original_horizon_steps=mpc.config.horizon_steps,
        full_horizon_candidates=len(full_candidates), truncated=truncated,
        position_diversity_m=position_separation, yaw_diversity_radians=yaw_separation,
        control_pool_size=len(pool), reserved_lateral_exploration=True,
        initial_guidance_lateral_preference=lateral_preference,
        guidance_ranking_scale_m=mpc.config.lookahead_distance,
        wall_seconds=time.perf_counter()-started,
        scope="candidate generation only; no global convergence, real-time or paper equivalence; no source-clock change"))

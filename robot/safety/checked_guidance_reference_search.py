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
    rates = np.unique([mpc.config.minimum_angular_velocity,
        mpc.config.minimum_angular_velocity/2, 0., mpc.config.maximum_angular_velocity/2,
        mpc.config.maximum_angular_velocity])
    pool = tuple(itertools.product(speeds, rates))
    beams = [np.empty((0, 2))]
    distance = float(np.linalg.norm(goal-initial[:2]))
    tested = exact_feasible = semantic_rejections = depth_done = 0
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
            state = value.chassis_states[-1]
            error = math.atan2(goal[1]-state[1], goal[0]-state[0])-state[2]
            score = distance - np.linalg.norm(goal-state[:2]) - mpc.config.lookahead_distance * abs(
                math.atan2(math.sin(error), math.cos(error)))
            candidates.append((float(score), state, commands, value))
        candidates.sort(key=lambda item: item[0], reverse=True)
        if depth == mpc.config.horizon_steps:
            full_candidates = candidates
        if not candidates:
            break
        depth_done = depth
        winners = []
        for item in candidates:
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
        completed_depth=depth_done, original_horizon_steps=mpc.config.horizon_steps,
        full_horizon_candidates=len(full_candidates), truncated=truncated,
        position_diversity_m=position_separation, yaw_diversity_radians=yaw_separation,
        wall_seconds=time.perf_counter()-started,
        scope="candidate generation only; no global convergence, real-time or paper equivalence; no source-clock change"))

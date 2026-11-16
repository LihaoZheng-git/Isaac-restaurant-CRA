"""Bounded online reference candidates, never a replacement safety controller."""

import math

import numpy as np

from robot.control.unicycle_model import lookahead_control_matrix


def search_checked_moving_arc(mpc, initial, goal, speed, field):
    """Rank full-horizon arcs with the existing progress/heading reference score.

    Candidates depend only on the current state, field, goal and actuator limits.
    Their additional forecast half-space gate matches the projected reference
    builder. Passing here never authorizes execution against a later field.
    """
    audit = dict(
        checked=0,
        exact_safe=0,
        forecast_safe=0,
        useful=0,
        selected=False,
        selected_command=None,
        selected_score=None,
        minimum_forecast_margin=None,
        basis="current_field_full_horizon_constant_moving_arcs",
    )
    current = mpc._sample(initial, field)
    best, rank, fraction = None, -math.inf, None
    distance = float(np.linalg.norm(goal - initial[:2]))
    kwargs = {k: field[k] for k in ("configuration_space", "h_full", "guidance_fields")}
    kwargs["temporal_safety_derivative"] = field.get("temporal_safety_derivative", 0.0)
    for speed_fraction in mpc.guidance_config.reference_speed_fractions():
        candidate_speed = speed * speed_fraction
        preference = mpc._reference_preference(
            initial, goal, candidate_speed, current, field
        )
        if preference is None:
            continue
        direction = preference[0]
        initial_error = abs((direction - initial[2] + math.pi) % math.tau - math.pi)
        for omega in np.linspace(
            mpc.config.minimum_angular_velocity, mpc.config.maximum_angular_velocity, 33
        ):
            commands = np.tile([candidate_speed, omega], (mpc.config.horizon_steps, 1))
            audit["checked"] += 1
            candidate = mpc._evaluate_trajectory(
                initial, commands, require_feasible=True, **kwargs
            )
            if candidate is None or not mpc._trajectory_is_safe(candidate):
                continue
            audit["exact_safe"] += 1
            margins = []
            for i, command in enumerate(commands):
                normal = (
                    lookahead_control_matrix(
                        candidate.chassis_states[i, 2],
                        lookahead_distance=mpc.config.lookahead_distance,
                    ).T
                    @ candidate.samples.guidance_normals[i]
                )
                rhs = (
                    -mpc.paper_filter.config.gamma
                    * max(0.0, candidate.safety_values[i])
                    - candidate.samples.scaled_temporal_derivatives[i]
                )
                margins.append(float(normal @ command - rhs))
            if (
                not np.isfinite(margins).all()
                or min(margins) < -mpc.paper_filter.config.numerical_tolerance
            ):
                continue
            audit["forecast_safe"] += 1
            remaining = float(np.linalg.norm(goal - candidate.chassis_states[-1, :2]))
            moved = float(
                np.linalg.norm(candidate.chassis_states[-1, :2] - initial[:2])
            )
            if (
                remaining > mpc.guidance_config.reference_goal_radius_m
                and moved + 1e-9 < mpc.guidance_config.minimum_reference_progress_m
            ):
                continue
            audit["useful"] += 1
            final_error = abs(
                (direction - candidate.chassis_states[-1, 2] + math.pi) % math.tau
                - math.pi
            )
            score = (
                distance
                - remaining
                + mpc.config.lookahead_distance * (initial_error - final_error)
            )
            if score > rank + 1e-6:
                best, rank, fraction = candidate, score, speed_fraction
                audit.update(
                    selected_command=list(map(float, commands[0])),
                    selected_score=score,
                    minimum_forecast_margin=min(margins),
                )
    return best, rank, fraction, audit

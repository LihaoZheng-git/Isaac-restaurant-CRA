"""Independent timing/location gate for the new person-relative demonstration.

This stricter, separate evaluation never rewrites earlier rear-pass reports or
feeds actor truth to planning. Arrival, clearance and execution stay separate.
"""

import numpy as np


def summarize_person_relative_rear(truth, *, nominal_goal_world_xy):
    times = np.array([r["time"] for r in truth], float)
    robot = np.array([r["robot_xy"] for r in truth], float)
    actor = np.array([r["actor_xyz"][:2] for r in truth], float)
    goal = np.asarray(nominal_goal_world_xy, float)
    if (
        len(times) < 3
        or np.any(np.diff(times) <= 0)
        or goal.shape != (2,)
        or not all(np.isfinite(a).all() for a in (times, robot, actor, goal))
    ):
        raise ValueError("complete finite ordered evaluation trace required")
    if any(
        r.get("phase")
        not in {
            "pickup",
            "idle_before",
            "move",
            "stopped_after",
            "aborted",
            "walk_away",
            "away_endpoint",
        }
        for r in truth
    ):
        raise ValueError("complete native phase scope required for independent scoring")
    forward = goal - robot[0]
    if np.linalg.norm(forward) < 1e-9:
        raise ValueError("nonzero nominal route chord required")
    forward /= np.linalg.norm(forward)
    velocity = np.gradient(actor, times, axis=0)
    walking = (np.array([r["phase"] for r in truth]) == "move") & (
        np.linalg.norm(velocity, axis=1) >= 0.3
    )
    checks = dict(
        independent_transverse_window=bool(walking.any()),
        crossing_line_during_transverse_walk=False,
        behind_current_person_at_crossing=False,
        curved_near_crossing_line_during_walk=False,
    )
    crossing_time, behind, excursion, line_distance = None, None, None, None
    if walking.any():
        motion = np.median(velocity[walking], axis=0)
        motion /= np.linalg.norm(motion)
        plane = float(np.median(actor[walking] @ forward))
        progress = robot @ forward - plane
        hits = np.flatnonzero((progress[:-1] < 0) & (progress[1:] >= 0))
        if len(hits):
            i = int(hits[0])
            fraction = -progress[i] / (progress[i + 1] - progress[i])
            crossing_time = float(times[i] + fraction * (times[i + 1] - times[i]))
            relative = (robot[i] - actor[i]) + fraction * (
                (robot[i + 1] - actor[i + 1]) - (robot[i] - actor[i])
            )
            behind = float(relative @ motion)
            checks["crossing_line_during_transverse_walk"] = bool(
                walking[i] and walking[i + 1]
            )
            checks["behind_current_person_at_crossing"] = behind <= -0.10
        lateral = np.array([-forward[1], forward[0]])
        trailing = -np.sign(motion @ lateral)
        # New explicit LOCATION criterion, not a reduced old arc/safety margin.
        near = (
            walking
            & (np.abs(progress) <= 0.75)
            & (np.linalg.norm(robot - actor, axis=1) <= 2.0)
        )
        if near.any():
            values = (robot - robot[0]) @ lateral * trailing
            index = np.flatnonzero(near)[np.argmax(values[near])]
            excursion = float(values[index])
            line_distance = float(abs(progress[index]))
            checks["curved_near_crossing_line_during_walk"] = excursion >= 0.20
    return dict(
        passed=all(checks.values()),
        checks=checks,
        crossing_timestamp_seconds=crossing_time,
        behind_projection_m=behind,
        trailing_arc_near_line_m=excursion,
        arc_distance_from_line_m=line_distance,
        maximum_arc_line_distance_m=0.75,
        required_arc_m=0.20,
        scope="Separate stricter timed/person-location component, actor truth evaluation-only; original rear results unchanged; physical/arrival/stopping/execution gates separately required",
    )

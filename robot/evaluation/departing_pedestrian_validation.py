"""Independent acceptance of the separate crossing-and-departure fixture."""

import numpy as np


def summarize_departing_pedestrian(records, *, parked_robot, crossing_direction="left-to-right", crossing_y_m=8.9, crossing_layout="legacy"):
    from robot.scene.rear_pass_encounter import validate_crossing_layout
    open_layout = validate_crossing_layout(crossing_layout, direction=crossing_direction,
        extension=0., crossing_y=8.9 if crossing_layout == 'lower-aisle' else crossing_y_m)
    if open_layout:
        crossing_y_m = 7.0
    if crossing_direction not in {"left-to-right", "right-to-left"}:
        raise ValueError("unsupported independent crossing direction")
    if isinstance(crossing_y_m, bool) or not np.isfinite(crossing_y_m) or not (crossing_y_m == 7.0 if open_layout else 8.4 <= crossing_y_m <= 8.9):
        raise ValueError("explicit authored crossing-line evaluation scope required")
    if len(records) < 30:
        return dict(passed=False, checks=dict(enough_actual_samples=False))
    t = np.array([r["time"] for r in records], dtype=float)
    xyz = np.array([r["actor_xyz"] for r in records], dtype=float)
    if not np.isfinite(t).all() or not np.isfinite(xyz).all() or np.any(np.diff(t) <= 0):
        raise ValueError("finite ordered actual actor trace required")
    phases = np.array([r["phase"] for r in records])
    recognized = {"idle_before", "move", "walk_away", "away_endpoint", "aborted"}
    if not set(phases) <= recognized:
        raise ValueError("explicit crossing/departure phase metadata required")
    left_to_right = crossing_direction == "left-to-right"
    crossed_side = xyz[:, 0] >= 29.5 if left_to_right else xyz[:, 0] <= 28.1
    crossed = (phases == "walk_away") & crossed_side & (np.abs(xyz[:, 1] - crossing_y_m) <= .4)
    departing = phases == "walk_away"
    speed = np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1) / np.diff(t)
    walking = np.isin(phases, ["move", "walk_away"])
    steady = speed[walking[1:] & walking[:-1] & (speed >= .25)]
    median = float(np.median(steady)) if len(steady) else None
    upper = np.flatnonzero(xyz[:, 1] >= 11.5)
    feet = np.array([np.array(r["left_foot_xyz"]) - r["right_foot_xyz"] for r in records])
    geometry = np.array([[r["actor_layout_clearance_m"], r["collider_alignment_error_m"]] for r in records])
    robot = np.array([r["robot_xy"] for r in records])
    checks = dict(
        actual_upper_aisle_departure=departing.sum() >= 30 and len(upper) > 0,
        no_return_to_robot_route=bool(len(upper)) and np.all(xyz[upper[0]:, 1] >= 11.2),
        requested_half_metre_speed_observed=len(steady) >= 10 and .325 <= (median or 0) <= .7,
        articulated_limbs=np.isfinite(feet).all() and walking.any()
            and np.max(np.ptp(feet[walking], axis=0)) >= .1,
        bounded_actor_region=bool(((xyz[:, 0] >= (25.5 if open_layout else 27.3)) & (xyz[:, 0] <= 30.4)
            & (xyz[:, 1] >= (6.7 if open_layout else 8.2)) & (xyz[:, 1] <= 13.3)).all()),
        grounded_actor=np.max(np.abs(xyz[:, 2])) <= .05,
        independent_geometry=np.isfinite(geometry).all() and geometry[:, 0].min() > .02
            and geometry[:, 1].max() <= .03,
        robot_zero_wheels=not parked_robot or (np.isfinite(robot).all()
            and np.linalg.norm(robot - robot[0], axis=1).max() <= .02
            and all(r["nonzero_wheel_command_count"] == 0 for r in records)),
    )
    checks["actual_left_to_right_crossing" if left_to_right else "actual_right_to_left_crossing"] = (
        (xyz[0, 0] <= 28.1 if left_to_right else xyz[0, 0] >= 29.5) and crossed.any())
    checks = {k: bool(v) for k, v in checks.items()}
    return dict(passed=all(checks.values()), checks=checks,
        median_moving_speed_mps=median, maximum_actor_y_m=float(xyz[:, 1].max()),
        minimum_actor_layout_clearance_m=float(geometry[:, 0].min()),
        maximum_collider_alignment_error_m=float(geometry[:, 1].max()),
        authored_crossing_layout=crossing_layout,
        scope="separate one-crossing then upper-aisle departure; actual root/articulation/geometry only; no human endpoint stopping, CRA or robot arrival claim")

"""Actual continuous actor/collider acceptance; no endpoint-stop claim."""

import numpy as np


def summarize_continuous_server(records, *, requested_speed_mps, parked_robot):
    if len(records) < 30:
        return {"passed": False, "checks": {"enough_actual_samples": False}}
    t = np.asarray([r["time"] for r in records])
    xyz = np.asarray([r["actor_xyz"] for r in records])
    if (
        not np.isfinite(t).all()
        or not np.isfinite(xyz).all()
        or np.any(np.diff(t) <= 0)
    ):
        raise ValueError("finite distinct ordered actor observations required")
    moving = np.array([r["phase"] == "move" for r in records])
    speed = np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1) / np.diff(t)
    both = moving[1:] & moving[:-1]
    steady = speed[both & (speed > 0.5 * requested_speed_mps)]
    median = float(np.median(steady)) if len(steady) else None
    tail = xyz[t >= t[-1] - 1.0, :2]
    feet = np.array(
        [np.array(r["left_foot_xyz"]) - r["right_foot_xyz"] for r in records]
    )
    geometry = np.array(
        [
            [r["actor_layout_clearance_m"], r["collider_alignment_error_m"]]
            for r in records
        ]
    )
    robot = np.array([r["robot_xy"] for r in records])
    # This circuit turns at x=28.0 with a .25 m handoff radius. x<=28.3
    # places the .4 m body envelope left of the x=28.8 robot centerline.
    crossed = np.flatnonzero(
        moving & (xyz[:, 0] <= 28.3) & (np.abs(xyz[:, 1] - 8.9) <= 0.4)
    )
    checks = {
        "actual_route_crossing": xyz[0, 0] >= 29.5 and len(crossed) > 0,
        "continuous_actor_mode": bool(moving.any())
        and all(r["phase"] == "move" for r in records[np.flatnonzero(moving)[0] :]),
        "actual_terminal_motion": len(tail) >= 10
        and np.linalg.norm(tail[-1] - tail[0]) >= 0.12,
        "requested_speed_observed": len(steady) >= 10
        and 0.65 * requested_speed_mps <= median <= 1.4 * requested_speed_mps,
        "articulated_limbs": np.isfinite(feet).all()
        and moving.any()
        and np.max(np.ptp(feet[moving], axis=0)) >= 0.1,
        "bounded_actor_region": bool(
            (
                (xyz[:, 0] >= 27.5)
                & (xyz[:, 0] <= 30.4)
                & (xyz[:, 1] >= 8.2)
                & (xyz[:, 1] <= 12.1)
            ).all()
        ),
        "grounded_actor": np.max(np.abs(xyz[:, 2])) <= 0.05,
        "independent_geometry": np.isfinite(geometry).all()
        and geometry[:, 0].min() > 0.02
        and geometry[:, 1].max() <= 0.03,
        "robot_zero_wheels": not parked_robot
        or (
            np.isfinite(robot).all()
            and np.linalg.norm(robot - robot[0], axis=1).max() <= 0.02
            and all(r["nonzero_wheel_command_count"] == 0 for r in records)
        ),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": {k: bool(v) for k, v in checks.items()},
        "median_moving_speed_mps": median,
        "terminal_one_second_displacement_m": float(np.linalg.norm(tail[-1] - tail[0])),
        "minimum_actor_layout_clearance_m": float(geometry[:, 0].min()),
        "scope": "explicit continuous native server circuit; actual crossing/articulation/collider geometry; no human endpoint stopping, robot arrival, perception or real-time claim",
    }

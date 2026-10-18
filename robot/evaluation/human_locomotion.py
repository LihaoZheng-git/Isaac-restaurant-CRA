"""Independent actor-only acceptance; root translation alone is not animation."""

import numpy as np


def summarize_human_locomotion(
    records, *, goal_world_xyz, requested_speed_mps, task_succeeded
):
    if len(records) < 20:
        return {"passed": False, "checks": {"enough_samples": False}}
    times = np.asarray([r["time"] for r in records])
    xyz = np.asarray([r["actor_xyz"] for r in records])
    if (
        not np.isfinite(times).all()
        or not np.isfinite(xyz).all()
        or (np.diff(times) <= 0).any()
    ):
        raise ValueError("locomotion samples require finite ordered epochs")
    for key in (
        "left_foot_xyz",
        "right_foot_xyz",
        "robot_xy",
        "collider_alignment_error_m",
        "actor_layout_clearance_m",
    ):
        if not np.isfinite(np.asarray([r[key] for r in records])).all():
            raise ValueError("locomotion geometry must be finite")
    phases = np.array([r["phase"] for r in records])
    speed = np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1) / np.diff(times)
    moving = (phases[1:] == "move") & (phases[:-1] == "move")
    # Separate acceleration/braking from the steady moving speed, retaining all
    # samples in the raw record and rejecting an episode that never gets moving.
    steady = speed[moving & (speed > 0.5 * requested_speed_mps)]
    stop_xyz = xyz[phases == "stopped_after"]
    foot = np.asarray(
        [
            np.asarray(r["left_foot_xyz"]) - np.asarray(r["right_foot_xyz"])
            for r in records
        ]
    )
    articulation = (
        float(np.max(np.ptp(foot[phases == "move"], axis=0)))
        if (phases == "move").any()
        else 0.0
    )
    collider_errors = [r["collider_alignment_error_m"] for r in records]
    robot_xy = np.asarray([r["robot_xy"] for r in records])
    stop_drift = (
        float(np.linalg.norm(stop_xyz[:, :2] - stop_xyz[0, :2], axis=1).max())
        if len(stop_xyz) >= 10
        else None
    )
    median = float(np.median(steady)) if len(steady) else None
    checks = {
        "task_arrival": bool(task_succeeded)
        and float(np.linalg.norm(xyz[-1, :2] - goal_world_xyz[:2])) <= 0.25,
        "requested_speed_observed": len(steady) >= 10
        and 0.65 * requested_speed_mps <= median <= 1.4 * requested_speed_mps,
        "articulated_limb_motion": articulation >= 0.1,
        "actor_stopped": stop_drift is not None and stop_drift <= 0.03,
        "actor_root_grounded": bool(np.max(np.abs(xyz[:, 2])) <= 0.05),
        "collider_follows_actor": all(np.isfinite(collider_errors))
        and max(collider_errors) <= 0.03,
        "robot_stationary": float(np.linalg.norm(robot_xy - robot_xy[0], axis=1).max())
        <= 0.02
        and all(r["nonzero_wheel_command_count"] == 0 for r in records),
        "independent_clearance": min(r["actor_layout_clearance_m"] for r in records)
        > 0.02,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "median_moving_speed_mps": median,
        "relative_foot_motion_range_m": articulation,
        "actor_stop_drift_m": stop_drift,
        "maximum_actor_root_height_error_m": float(np.max(np.abs(xyz[:, 2]))),
        "max_collider_alignment_error_m": max(collider_errors),
        "minimum_independent_actor_clearance_m": min(
            r["actor_layout_clearance_m"] for r in records
        ),
        "scope": "native articulated actor and parked-robot diagnostic; not CRA, sensor tracking or navigation acceptance",
    }

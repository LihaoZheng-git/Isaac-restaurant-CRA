"""Independent parked-robot human sensor benchmark acceptance."""

import numpy as np

from robot.perception.human_mass_prior import HUMAN_MASS_PRIOR_SOURCE


def summarize_human_sensor(records, truth):
    """Independent oracle match; no native actor data enters the estimators."""
    rows = []
    if len(truth) >= 3:
        times = np.array([r["time"] for r in truth])
        xy = np.array([r["actor_xyz"][:2] for r in truth])
        if (
            not np.isfinite(times).all()
            or not np.isfinite(xy).all()
            or np.any(np.diff(times) <= 0)
        ):
            raise ValueError("human evaluation truth requires finite ordered epochs")
        for r in records:
            t = r["time"]
            if not times[0] + 0.1 <= t <= times[-1] - 0.1:
                continue
            center = np.array([np.interp(t, times, xy[:, i]) for i in range(2)])
            velocity = np.array(
                [
                    (
                        np.interp(t + 0.1, times, xy[:, i])
                        - np.interp(t - 0.1, times, xy[:, i])
                    )
                    / 0.2
                    for i in range(2)
                ]
            )
            matched = [
                o
                for o in r["objects"]
                if o["diagnostics"]["instance_centroid_world_xyz"] is not None
                and np.linalg.norm(
                    np.array(o["diagnostics"]["instance_centroid_world_xyz"][:2])
                    - center
                )
                <= 0.45
            ]
            if len(matched) != 1:
                continue
            o = matched[0]
            attrs = o["attributes"]
            mass = attrs["mass"]["mass_kg"]
            motion = attrs["motion"]
            if mass["source"] != HUMAN_MASS_PRIOR_SOURCE or mass["status"] != "prior":
                continue
            if (
                not o["diagnostics"]["accepted_instance"]
                or o["diagnostics"]["policy_return_count"] <= 0
                or abs(mass["timestamp_seconds"] - r["camera_time"]) > 1e-6
                or any(abs(v["timestamp_seconds"] - t) > 1e-6 for v in motion.values())
            ):
                continue
            if any(
                motion[k]["value"] is None
                for k in ("velocity_x_mps", "velocity_y_mps", "relative_speed_mps")
            ):
                continue
            speed = motion["relative_speed_mps"]["value"]
            if any(
                motion[k]["status"] not in {"measured", "inferred"}
                for k in ("velocity_x_mps", "velocity_y_mps", "relative_speed_mps")
            ):
                continue
            estimated_v = np.array(
                [motion[k]["value"] for k in ("velocity_x_mps", "velocity_y_mps")]
            )
            rows.append(
                {
                    "time": t,
                    "phase": r["phase"],
                    "track_id": attrs["track_id"],
                    "speed_mps": speed,
                    "oracle_speed_mps": float(np.linalg.norm(velocity)),
                    "velocity_error_mps": float(np.linalg.norm(estimated_v - velocity)),
                    "mass_status": mass["status"],
                    "mass_kg_prior": mass["value"],
                    "rough_relative_energy_j": 0.5 * mass["value"] * speed**2,
                    "phi": o["phi"],
                    "factor_scores": o["diagnostics"]["factor_scores"],
                }
            )
    idle = [r for r in rows if r["phase"] == "idle_before"]
    moving = [r for r in rows if r["phase"] == "move" and r["oracle_speed_mps"] >= 0.4]
    error = (
        float(np.median([r["velocity_error_mps"] for r in moving])) if moving else None
    )
    idle_speed = float(np.median([r["speed_mps"] for r in idle])) if idle else None
    checks = {
        "stationary_visual_person_mass_prior": len(idle) >= 3 and idle_speed < 0.12,
        "moving_visual_person_mass_prior": len(moving) >= 5,
        "bounded_velocity_error": error is not None and error <= 0.25,
        "four_numeric_scores": bool(rows)
        and all(
            len(r["factor_scores"]) == 4
            and 0 <= r["phi"] <= 1
            and all(0 <= f["score"] <= 1 for f in r["factor_scores"].values())
            for r in rows
        ),
        "no_property_borrowing": all(
            f["mode"] != "persistence_prior"
            for r in rows
            for f in r["factor_scores"].values()
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "matched_fresh_prior_rows": rows,
        "stationary_count": len(idle),
        "steady_moving_count": len(moving),
        "median_stationary_speed_mps": idle_speed,
        "median_moving_velocity_error_mps": error,
        "maximum_result_completion_age_sim_seconds": max(
            (r["completion_age_sim_seconds"] for r in records), default=None
        ),
        "scope": "parked-robot asynchronous RGB/LiDAR component diagnostic; robot pose/extrinsics are oracle references; RGB time is read-time approximation; mass is a configured adult prior; no temperature/sharpness, moving-object deskew, real-time or robot crossing acceptance",
    }

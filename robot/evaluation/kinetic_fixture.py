"""Authored bounded actor motion for independent sensor evaluation only."""

import math

import numpy as np


def actor_offset_velocity(seconds):
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("actor time must be finite and nonnegative")
    if seconds < 2:
        return 0.0, 0.0, "stopped_before"
    if seconds < 5:
        return 0.12 * (seconds - 2), 0.12, "slow"
    if seconds < 7:
        return 0.36, 0.0, "stopped_middle"
    if seconds < 10:
        return 0.36 - 0.24 * (seconds - 7), -0.24, "fast"
    return -0.36, 0.0, "stopped_after"


def summarize_kinetic_records(records):
    groups = {}
    for record in records:
        truth = record["actor_monitor"]
        elapsed = truth["elapsed_seconds"]
        if any(abs(elapsed - boundary) < 0.4 for boundary in (0, 2, 5, 7, 10, 12)):
            continue  # explicitly exclude scan/filter transition intervals
        matches = [
            item
            for item in record["objects"]
            if np.linalg.norm(
                np.asarray(item["diagnostics"]["centroid_world_xy"])
                - truth["center_world_xy"]
            )
            < 0.4
        ]
        if len(matches) != 1:
            continue
        item = matches[0]
        speed = item["attributes"]["motion"]["relative_speed_mps"]
        factor = item["cra_control"]["cra"]["factors"]["kinetic"]
        velocity = np.asarray(item["diagnostics"]["velocity_world_xy"])
        if speed["value"] is None or factor["confidence"] <= 0:
            continue
        if item["diagnostics"]["policy_return_count"] == 0:
            continue
        error = float(np.linalg.norm(velocity - truth["velocity_world_xy"]))
        groups.setdefault(truth["phase"], []).append(
            (speed["value"], error, factor["severity"], factor["nominal_risk"])
        )
    rows = {
        name: {
            "usable_updates": len(values),
            "median_relative_speed_mps": float(np.median(np.array(values)[:, 0])),
            "median_velocity_error_mps": float(np.median(np.array(values)[:, 1])),
            "median_energy_severity": float(np.median(np.array(values)[:, 2])),
            "median_nominal_kinetic": float(np.median(np.array(values)[:, 3])),
        }
        for name, values in groups.items()
    }
    checks = {
        "enough_updates": all(
            rows.get(name, {}).get("usable_updates", 0) >= 5
            for name in ("stopped_before", "slow", "fast", "stopped_after")
        )
    }
    if checks["enough_updates"]:
        checks.update(
            stationary_speed=max(
                rows[name]["median_relative_speed_mps"]
                for name in ("stopped_before", "stopped_after")
            )
            < 0.06,
            bounded_velocity_error=max(
                row["median_velocity_error_mps"] for row in rows.values()
            )
            < 0.10,
            speed_ordering=rows["fast"]["median_relative_speed_mps"]
            > rows["slow"]["median_relative_speed_mps"] + 0.06,
            energy_ordering=rows["fast"]["median_energy_severity"]
            > rows["slow"]["median_energy_severity"]
            > rows["stopped_before"]["median_energy_severity"],
        )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "phases": rows,
        "scope": "stationary robot, actual simulated LiDAR/RGB; independently authored actor motion is evaluation truth only; excludes .4 s phase transitions; not crowded or real-time validation",
    }


def summarize_live_relative_motion(records):
    """Audit simultaneous motion without supplying truth to the estimator."""
    errors = []
    seen_sources = set()
    for record in records:
        source = (record["sensor_frame_id"], record["sensor_timestamp"])
        if source in seen_sources:
            continue
        seen_sources.add(source)
        truth = record["actor_monitor"]
        elapsed = truth["elapsed_seconds"]
        if any(abs(elapsed - boundary) < 0.4 for boundary in (0, 2, 5, 7, 10)):
            continue
        actor_velocity = np.asarray(truth["velocity_world_xy"])
        robot_velocity = np.asarray(
            record["diagnostics"]["measured_robot_velocity_world_xy"]
        )
        if (
            np.linalg.norm(actor_velocity) == 0
            or np.linalg.norm(robot_velocity) <= 0.02
        ):
            continue
        matches = [
            item
            for item in record["objects"]
            if np.linalg.norm(
                np.asarray(item["diagnostics"]["centroid_world_xy"])
                - truth["center_world_xy"]
            )
            < 0.4
        ]
        if len(matches) != 1:
            continue
        item = matches[0]
        speed = item["attributes"]["motion"]["relative_speed_mps"]
        if (
            item["diagnostics"]["policy_return_count"] == 0
            or speed["value"] is None
            or speed["status"] not in {"measured", "inferred"}
            or item["diagnostics"]["factor_scores"]["kinetic"]["mode"]
            != "evidence_supported"
        ):
            continue
        expected = float(np.linalg.norm(actor_velocity - robot_velocity))
        errors.append(
            (
                abs(speed["value"] - expected),
                float(
                    np.linalg.norm(
                        np.asarray(item["diagnostics"]["velocity_world_xy"])
                        - actor_velocity
                    )
                ),
            )
        )
    relative_error = float(np.median(np.asarray(errors)[:, 0])) if errors else None
    velocity_error = float(np.median(np.asarray(errors)[:, 1])) if errors else None
    checks = {
        "enough_simultaneous_updates": len(errors) >= 5,
        "bounded_relative_speed_error": relative_error is not None
        and relative_error < 0.10,
        "bounded_world_velocity_error": velocity_error is not None
        and velocity_error < 0.10,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "usable_updates": len(errors),
        "median_relative_speed_error_mps": relative_error,
        "median_world_velocity_error_mps": velocity_error,
        "scope": "actual simultaneous robot/actor sensor updates; actor truth is independent evaluation metadata; ego velocity uses declared oracle-pose references; .4 s transitions excluded; not crowded or real-time acceptance",
    }

"""Independent acceptance for one native adult crossing a waypoint route.

Actor root truth is evaluation-only. These checks never provide CRA evidence.
"""

import numpy as np

from robot.perception.human_mass_prior import HUMAN_MASS_PRIOR_SOURCE


def human_stationary_checks(records):
    fresh = []
    for r in records:
        for o in r["objects"]:
            m = o["attributes"]["mass"]["mass_kg"]
            d = o["diagnostics"]
            if (
                m["source"] == HUMAN_MASS_PRIOR_SOURCE
                and m["status"] == "prior"
                and d["accepted_instance"]
                and d["policy_return_count"] > 0
                and d["property_evidence"]["mass"]["mode"] == "fresh"
            ):
                fresh.append(o)
    return {
        "three_fresh_visual_person_mass_priors": len(fresh) >= 3,
        "current_supported_motion": sum(
            o["diagnostics"]["factor_scores"]["kinetic"]["mode"] == "evidence_supported"
            for o in fresh
        )
        >= 3,
        "observed_contact_proxy": any(
            o["diagnostics"]["factor_scores"]["contact"]["mode"] == "evidence_supported"
            for o in fresh
        ),
    }


def summarize_human_navigation(records, truth, *, mode):
    if mode not in {"standing", "crossing", "continuous-crossing"}:
        raise ValueError("unknown human navigation mode")
    rows = []
    if len(truth) >= 3:
        times = np.array([r["time"] for r in truth])
        xy = np.array([r["actor_xyz"][:2] for r in truth])
        if (
            not np.isfinite(times).all()
            or not np.isfinite(xy).all()
            or np.any(np.diff(times) <= 0)
        ):
            raise ValueError("finite ordered actor truth required")
        for r in records:
            t = r["sensor_timestamp"]
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
            matches = [
                o
                for o in r["objects"]
                if np.linalg.norm(
                    np.array(o["diagnostics"]["centroid_world_xy"]) - center
                )
                <= 0.45
            ]
            if len(matches) != 1:
                continue
            o = matches[0]
            motion = o["attributes"]["motion"]
            if any(
                motion[k]["value"] is None
                or motion[k]["status"] not in {"measured", "inferred"}
                or abs(motion[k]["timestamp_seconds"] - t) > 1e-6
                for k in ("velocity_x_mps", "velocity_y_mps", "relative_speed_mps")
            ):
                continue
            est = np.array(
                [motion[k]["value"] for k in ("velocity_x_mps", "velocity_y_mps")]
            )
            rows.append(
                {
                    "time": t,
                    "track_id": o["track_id"],
                    "oracle_speed_mps": float(np.linalg.norm(velocity)),
                    "world_velocity_error_mps": float(np.linalg.norm(est - velocity)),
                    "relative_speed_mps": motion["relative_speed_mps"]["value"],
                    "robot_speed_mps": float(
                        np.linalg.norm(
                            r["diagnostics"]["measured_robot_velocity_world_xy"]
                        )
                    ),
                    "relative_speed_error_mps": abs(
                        motion["relative_speed_mps"]["value"]
                        - float(
                            np.linalg.norm(
                                velocity
                                - np.array(
                                    r["diagnostics"]["measured_robot_velocity_world_xy"]
                                )
                            )
                        )
                    ),
                    "positive_owned_returns": o["diagnostics"]["policy_return_count"],
                    "phi": o["cra_control"]["safe_sage"]["risk"],
                    "factor_scores": o["diagnostics"]["factor_scores"],
                    "mass": o["attributes"]["mass"]["mass_kg"],
                    "properties": o["diagnostics"]["property_evidence"],
                }
            )
    steady = [r for r in rows if r["oracle_speed_mps"] >= 0.4]
    simultaneous = [
        r
        for r in steady
        if r["robot_speed_mps"] > 0.02
        and r["positive_owned_returns"] > 0
        and r["factor_scores"]["kinetic"]["mode"] == "evidence_supported"
    ]
    supported_moving = [
        r
        for r in steady
        if r["positive_owned_returns"] > 0
        and r["factor_scores"]["kinetic"]["mode"] == "evidence_supported"
    ]
    error = (
        float(np.median([r["world_velocity_error_mps"] for r in steady]))
        if steady
        else None
    )
    checks = {
        "current_human_motion_rows": len(rows) >= 5,
        "supported_human_kinetic_uses_adult_mass_prior": all(
            r["mass"]["source"] == HUMAN_MASS_PRIOR_SOURCE
            and r["mass"]["status"] == "prior"
            and r["mass"]["value"] is not None
            and np.isfinite(r["mass"]["value"])
            and r["mass"]["value"] > 0
            for r in rows
            if r["factor_scores"]["kinetic"]["mode"] == "evidence_supported"
        ),
        "bounded_relative_speed_error": bool(rows)
        and float(np.median([r["relative_speed_error_mps"] for r in rows])) <= 0.25,
        "only_mass_may_persist": all(
            p["mode"] != "persistence_prior"
            for r in rows
            for name, p in r["properties"].items()
            if name != "mass"
        ),
        "bounded_mass_source_age": all(
            -0.15 - 1e-9 <= r["time"] - r["mass"]["timestamp_seconds"] <= 3 + 1e-9
            for r in rows
            if r["mass"]["value"] is not None
        ),
        "fresh_committed_fields": bool(records)
        and all(
            -1e-9 <= r["commit_simulation_time"] - r["sensor_timestamp"] <= 0.35 + 1e-9
            for r in records
        ),
        "numeric_scores": bool(rows)
        and all(
            0 <= r["phi"] <= 1
            and len(r["factor_scores"]) == 4
            and all(
                np.isfinite(f["score"]) and 0 <= f["score"] <= 1
                for f in r["factor_scores"].values()
            )
            for r in rows
        ),
        "all_track_numeric_policies_fixed_lambda": bool(records)
        and all(
            0 <= o["cra_control"]["safe_sage"]["risk"] <= 1
            and abs(o["cra_control"]["safe_sage"]["tangential_strength"] - 1.25) < 1e-9
            and set(o["diagnostics"]["factor_scores"])
            == {"potential", "kinetic", "contact", "thermal"}
            and all(
                np.isfinite(f["score"]) and 0 <= f["score"] <= 1
                for f in o["diagnostics"]["factor_scores"].values()
            )
            for r in records
            for o in r["objects"]
        ),
    }
    if mode in {"crossing", "continuous-crossing"}:
        checks.update(
            supported_moving_kinetic_rows=len(supported_moving) >= 5,
            simultaneous_motion_observed=len(simultaneous) >= 1,
            bounded_steady_world_velocity_error=error is not None and error <= 0.3,
            actual_crossing=bool(truth)
            and truth[-1]["actor_xyz"][0] <= 28.05
            and truth[-1]["phase"] == "stopped_after",
        )
        if mode == "continuous-crossing":
            from robot.evaluation.continuous_server_validation import summarize_continuous_server
            checks.pop("actual_crossing")
            checks["actual_continuous_crossing"] = summarize_continuous_server(
                truth, requested_speed_mps=.5, parked_robot=False)["passed"]
    else:
        checks["stationary_world_velocity_error"] = (
            bool(rows)
            and float(np.median([r["world_velocity_error_mps"] for r in rows])) <= 0.15
        )
        checks["bounded_native_standing_idle"] = (
            len(truth) >= 3
            and max(
                np.linalg.norm(np.array(r["actor_xyz"][:2]) - truth[0]["actor_xyz"][:2])
                for r in truth
            )
            <= 0.25
            and float(
                np.median(
                    np.linalg.norm(
                        np.diff(xy, axis=0) / np.diff(times)[:, None], axis=1
                    )
                )
            )
            <= 0.12
        )
    checks = {k: bool(v) for k, v in checks.items()}
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "matched_current_motion_rows": len(rows),
        "simultaneous_rows": len(simultaneous),
        "supported_moving_kinetic_rows": len(supported_moving),
        "median_steady_world_velocity_error_mps": error,
        "median_relative_speed_error_mps": float(
            np.median([r["relative_speed_error_mps"] for r in rows])
        )
        if rows
        else None,
        "rows": rows,
        "scope": "one adult; configured human mass, potential/thermal priors, visible contact proxy; root-motion velocity reference is not physical COM; no crowds, real time, calibrated injury or moving plume acceptance",
    }

"""Independent post-hoc crossing-policy and visible-bypass acceptance.

Authored actor truth is scoring only and never enters the runtime policy.
"""

import numpy as np


def summarize_motion_passing(
    records, truth, *, moving_robot=False, nominal_goal_world_xy=None
):
    times = np.array([row["time"] for row in truth], dtype=float)
    if len(times) < 3 or np.any(np.diff(times) <= 0):
        raise ValueError("ordered independent actor trace required")
    actors = np.array([row["actor_xyz"][:2] for row in truth], dtype=float)
    selected, errors, ages, wrong = [], [], [], []
    policy_preserved = True
    for record in records:
        detail = record.get("diagnostics", record)
        audit = detail.get("motion_aware_passing") or {}
        t = record.get("sensor_timestamp", record.get("time"))
        if t is None or not times[0] <= t <= times[-1]:
            continue
        center = np.array([np.interp(t, times, actors[:, i]) for i in range(2)])
        for key, side in audit.get("preferences", {}).items():
            objects = [
                o
                for o in record["objects"]
                if o.get("track_id", o.get("attributes", {}).get("track_id")) == int(key)
            ]
            if len(objects) != 1:
                wrong.append("selected identity absent from current assessment")
                continue
            obj = objects[0]
            error = float(
                np.linalg.norm(
                    np.array(obj["diagnostics"]["centroid_world_xy"]) - center
                )
            )
            errors.append(error)
            ages.append(t - audit["encounter_selected_timestamp"])
            velocity = np.array(audit["selection_velocity_world_xy"])
            source = audit["encounter_selected_timestamp"]
            if times[0] <= source <= times[-1]:
                original_actor = np.array(
                    [np.interp(source, times, actors[:, i]) for i in range(2)]
                )
                original_robot = np.array(
                    [
                        np.interp(source, times, [row["robot_xy"][i] for row in truth])
                        for i in range(2)
                    ]
                )
                normal = original_actor - original_robot
                tangent = np.array([-normal[1], normal[0]])
                if side == "right":
                    tangent = -tangent
                if side not in {"left", "right"} or tangent @ velocity >= 0:
                    wrong.append(
                        "selected tangent does not trail original crossing motion"
                    )
            else:
                wrong.append(
                    "original selection is outside independent observation history"
                )
            p = np.array(obj["diagnostics"]["centroid_world_xy"])
            # The selected original velocity supplies the trailing orientation;
            # current actor truth supplies only independent identity scoring.
            selected.append(
                {
                    "time": t,
                    "track_id": int(key),
                    "side": side,
                    "centroid_error_m": error,
                    "selection_velocity_world_xy": velocity.tolist(),
                    "centroid_world_xy": p.tolist(),
                }
            )
        for row in detail.get("applied_field_policies", ()):
            policy_preserved &= (
                row["lambda"] == 1.25
                and abs(row["boundary_flux"] + 0.5 + 1.2 * row["applied_field_risk"])
                < 1e-9
            )
    checks = {
        "three_observed_policy_rows": len(selected) >= 3,
        "selected_target_is_the_observed_human": bool(errors)
        and max(errors) <= 0.55
        and not wrong,
        "original_selection_clock_bounded": bool(ages)
        and min(ages) >= -1e-8
        and max(ages) <= 8.0 + 1e-8,
        "fixed_lambda_and_flux_mapping": bool(records) and bool(policy_preserved),
    }
    result = {
        "checks": checks,
        "selected_rows": selected,
        "rejections": wrong,
        "maximum_selected_centroid_error_m": max(errors, default=None),
        "maximum_original_selection_age_seconds": max(ages, default=None),
        "scope": "independent one-crossing scoring; not runtime actor truth or a prediction guarantee",
    }
    if moving_robot:
        robot = np.array([row["robot_xy"] for row in truth], dtype=float)
        goal = np.asarray(nominal_goal_world_xy, dtype=float)
        if goal.shape != (2,) or not np.isfinite(goal).all():
            raise ValueError("configured nominal goal required for bypass scoring")
        direction = goal - robot[0]
        distance = float(np.linalg.norm(direction))
        direction = direction / max(distance, 1e-9)
        left = np.array([-direction[1], direction[0]])
        lateral = (robot - robot[0]) @ left
        velocity = np.gradient(actors, times, axis=0)
        moving = np.linalg.norm(velocity, axis=1) >= 0.3
        separations = np.linalg.norm(robot - actors, axis=1)
        encounter = moving & (separations <= 2.0)
        if encounter.any():
            index = np.flatnonzero(encounter)[np.argmin(separations[encounter])]
            behind = float(
                (robot[index] - actors[index])
                @ velocity[index]
                / np.linalg.norm(velocity[index])
            )
            # For the bounded left-to-right fixture the trailing side is left
            # of the original northbound route. A large wrong-side bow fails.
            excursion = float(np.max(lateral[encounter]))
        else:
            behind, excursion = None, 0.0
        checks.update(
            visible_trailing_side_arc=excursion >= 0.20,
            passes_behind_moving_actor=behind is not None and behind <= -0.10,
            completed_route_progress=float(np.linalg.norm(robot[-1] - robot[0])) >= 3.5,
            rejoined_original_route_goal=float(np.linalg.norm(robot[-1] - goal))
            <= 0.15,
        )
        result.update(
            maximum_left_excursion_during_encounter_m=excursion,
            behind_projection_at_closest_encounter_m=behind,
            nominal_route_chord_m=distance,
            endpoint_goal_error_m=float(np.linalg.norm(robot[-1] - goal)),
        )
    result["passed"] = all(checks.values())
    return result

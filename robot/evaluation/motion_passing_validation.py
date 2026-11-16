"""Independent post-hoc crossing-policy and visible-bypass acceptance.

Authored actor truth is scoring only and never enters the runtime policy.
"""

import numpy as np


def summarize_motion_passing(
    records, truth, *, moving_robot=False, nominal_goal_world_xy=None, expected_lambda=1.25,
    continuing_away=False, encounter_guidance=False
):
    if expected_lambda not in {0., 1.25}:
        raise ValueError("only primary or explicit zero-bias ablation lambda supported")
    times = np.array([row["time"] for row in truth], dtype=float)
    if len(times) < 3 or np.any(np.diff(times) <= 0):
        raise ValueError("ordered independent actor trace required")
    actors = np.array([row["actor_xyz"][:2] for row in truth], dtype=float)
    selected, errors, ages, wrong = [], [], [], []
    policy_preserved = True
    prior_rows, prior_provenance_valid, original_priors = 0, True, {}
    for record in records:
        detail = record.get("diagnostics", record)
        audit = detail.get("motion_aware_passing") or {}
        t = record.get("sensor_timestamp", record.get("time"))
        if t is None or not times[0] <= t <= times[-1]:
            continue
        center = np.array([np.interp(t, times, actors[:, i]) for i in range(2)])
        for key, side in audit.get("preferences", {}).items():
            prior = audit.get("original_selection_identity_prior")
            if prior is not None:
                from robot.perception.human_body_envelope import HumanIdentitySelectionPrior, HUMAN_ENVELOPE_SOURCE
                try:
                    evidence = HumanIdentitySelectionPrior(**{
                        name: prior[name] for name in HumanIdentitySelectionPrior.__dataclass_fields__})
                    cap = audit.get("configuration", {}).get("maximum_identity_prior_selection_age_seconds", 0.)
                    signature = (int(key), audit["encounter_selected_timestamp"])
                    original = original_priors.setdefault(signature, prior)
                    current = [row for row in detail.get("human_body_envelope_priors", ())
                               if row["track_id"] == int(key)]
                    valid = (0 < cap <= .5 and evidence.age_seconds <= cap + 1e-9
                        and evidence.track_id == int(key)
                        and evidence.lidar_timestamp_seconds == audit["encounter_selected_timestamp"]
                        and prior.get("source") == HUMAN_ENVELOPE_SOURCE and prior.get("status") == "prior"
                        and audit.get("original_selection_visual_person_verified") is False
                        and prior == original and len(current) == 1
                        and current[0]["lidar_timestamp_seconds"] == t
                        and current[0]["source"] == HUMAN_ENVELOPE_SOURCE
                        and current[0]["status"] == "prior" and current[0]["current_support_count"] >= 3)
                except (ValueError, TypeError, KeyError):
                    valid = False
                prior_rows += 1
                prior_provenance_valid &= valid
            elif (audit.get("configuration", {}).get("maximum_identity_prior_selection_age_seconds", 0.) > 0
                  and audit.get("original_selection_visual_person_verified") is not True):
                prior_provenance_valid = False
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
            expected = expected_lambda
            if encounter_guidance:
                cue = row.get('encounter_guidance_ablation', {})
                expected = 0. if cue.get('lambda_suppressed') else 1.25
                policy_preserved &= (cue.get('source_timestamp_seconds') == t
                    and row.get('physical_CRA_lambda') == 1.25
                    and (not cue.get('lambda_suppressed') or (
                        cue.get('classification') == 'crossing'
                        and cue.get('parent_track_id') in detail.get('crossing_yield_track_ids', ()))))
            policy_preserved &= (
                row["lambda"] == expected
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
        "identity_prior_selection_provenance_bounded": bool(prior_provenance_valid),
    }
    result = {
        "checks": checks,
        "selected_rows": selected,
        "identity_prior_selected_rows": prior_rows,
        "rejections": wrong,
        "maximum_selected_centroid_error_m": max(errors, default=None),
        "maximum_original_selection_age_seconds": max(ages, default=None),
        "scope": "independent one-crossing scoring; not runtime actor truth or a prediction guarantee",
        "expected_field_lambda": 'encounter-dependent 0/1.25' if encounter_guidance else expected_lambda,
    }
    if moving_robot:
        robot = np.array([row["robot_xy"] for row in truth], dtype=float)
        goal = np.asarray(nominal_goal_world_xy, dtype=float)
        if goal.shape != (2,) or not np.isfinite(goal).all():
            raise ValueError("configured nominal goal required for bypass scoring")
        direction = goal - robot[0]
        distance = float(np.linalg.norm(direction))
        if distance <= .15:
            raise ValueError('nontrivial configured nominal route required for bypass scoring')
        direction = direction / distance
        left = np.array([-direction[1], direction[0]])
        lateral = (robot - robot[0]) @ left
        velocity = np.gradient(actors, times, axis=0)
        moving = np.linalg.norm(velocity, axis=1) >= 0.3
        phase_available = ["phase" in row for row in truth]
        encounter_basis = "actor_root_velocity_only; walking phase unavailable"
        excluded_nonwalking_rows = 0
        if any(phase_available):
            if not all(phase_available):
                raise ValueError("crossing evaluation requires complete walking-phase metadata")
            phases = np.array([row["phase"] for row in truth])
            recognized = {"pickup", "idle_before", "move", "stopped_after", "aborted"}
            if continuing_away:
                recognized |= {"walk_away", "away_endpoint"}
            if not set(phases) <= recognized:
                raise ValueError("unknown native human evaluation phase")
            # Idle skeletal animation can exceed the root-speed threshold.
            # Authored task phase is independent scoring truth, never a runtime
            # velocity input. It restricts the crossing window; the original
            # observed-speed, distance, arc and behind thresholds still apply.
            walking = phases == "move"
            excluded_nonwalking_rows = int(np.count_nonzero(moving & ~walking))
            moving &= walking
            encounter_basis = "native walking phase AND actor root speed >= .3 m/s; evaluation truth only"
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
            completed_route_progress=float((robot[-1] - robot[0]) @ direction) >= distance - .15,
            rejoined_original_route_goal=float(np.linalg.norm(robot[-1] - goal))
            <= 0.15,
        )
        result.update(
            maximum_left_excursion_during_encounter_m=excursion,
            behind_projection_at_closest_encounter_m=behind,
            nominal_route_chord_m=distance,
            required_route_progress_m=distance - .15,
            endpoint_goal_error_m=float(np.linalg.norm(robot[-1] - goal)),
            encounter_evaluation_basis=encounter_basis,
            excluded_nonwalking_speed_threshold_rows=excluded_nonwalking_rows,
            closest_moving_encounter_timestamp_seconds=float(times[index]) if encounter.any() else None,
        )
    result["passed"] = all(checks.values())
    return result

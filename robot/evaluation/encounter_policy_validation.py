"""Independent source/policy audit, separate from route and physical perception gates."""


def summarize_encounter_policy(records, *, require_transition=False):
    zero_rows = parallel_rows = 0
    errors = []
    previous = None
    for record in records:
        t = record["sensor_timestamp"]
        detail = record["diagnostics"]
        classification = detail.get("encounter_motion_classification") or {}
        tracks = {int(k): v for k, v in classification.get("tracks", {}).items()}
        passing = detail.get("motion_aware_passing") or {}
        selected = {int(k) for k in passing.get("preferences", {})}
        yield_ids = set(detail.get("crossing_yield_track_ids") or ())
        expected_yield = {
            k for k in selected if tracks.get(k, {}).get("classification") == "crossing"
        }
        if (
            previous is not None
            and t <= previous
            or classification.get("source_timestamp_seconds") != t
            or yield_ids != expected_yield
            or detail.get("physical_CRA_lambda") != 1.25
        ):
            errors.append(
                dict(time=t, reason="source_selection_or_physical_lambda_mismatch")
            )
        previous = t
        for key, row in tracks.items():
            if row.get("source_timestamp_seconds") != t or row.get("track_id") != key:
                errors.append(dict(time=t, reason="classification_source_mismatch"))
            state = row.get("classification")
            if state != "uncertain" and row.get("consistent_observations", 0) < 3:
                errors.append(dict(time=t, reason="unconfirmed_classification"))
            if state == "parallel":
                parallel_rows += 1
                if row.get("acute_route_angle_degrees", 91) > 40.0 + 1e-8:
                    errors.append(
                        dict(time=t, reason="parallel_angle_outside_hold_scope")
                    )
            if (
                state == "crossing"
                and row.get("acute_route_angle_degrees", 0) < 50.0 - 1e-8
            ):
                errors.append(dict(time=t, reason="crossing_angle_outside_hold_scope"))
        zero_rows += bool(yield_ids)
        for row in detail.get("applied_field_policies", ()):
            cue = row.get("encounter_guidance_ablation") or {}
            owner = cue.get("parent_track_id")
            expected = 0.0 if owner in expected_yield else 1.25
            if (
                row.get("lambda") != expected
                or cue.get("source_timestamp_seconds") != t
                or row.get("physical_CRA_lambda") != 1.25
                or abs(row["boundary_flux"] + 0.5 + 1.2 * row["applied_field_risk"])
                > 1e-9
            ):
                errors.append(
                    dict(
                        time=t,
                        track=row.get("track_or_group_id"),
                        reason="applied_policy_mismatch",
                    )
                )
    checks = dict(
        current_ordered_sources_and_bounded_policy=bool(records) and not errors,
        crossing_zero_and_parallel_normal_observed=not require_transition
        or zero_rows >= 3
        and parallel_rows >= 3,
    )
    return dict(
        passed=all(checks.values()),
        checks=checks,
        selected_crossing_zero_fields=zero_rows,
        current_parallel_classifications=parallel_rows,
        rejections=errors,
        scope="per-row source/motion/lambda/flux audit only; route, body containment, physical perception and actual execution are separate",
    )

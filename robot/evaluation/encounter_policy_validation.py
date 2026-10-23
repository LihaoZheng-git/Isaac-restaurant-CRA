"""Independent source/policy audit, separate from route and physical perception gates."""


def summarize_variant_passing(
    passing, *, require_encounter=False, encounter=None, strategy=None
):
    """Compare shared policy behavior without requiring a baseline's proposed feature.

    Arrival, clearance, braking, current perception and containment are separate
    mandatory caller gates. An actual selection must still have valid identity
    and source clocks. The encounter variant additionally needs its unchanged
    switching and intentional-wait acceptance; absent selections never pass it.
    """
    original = passing.get("checks", {})
    required = (
        "fixed_lambda_and_flux_mapping",
        "identity_prior_selection_provenance_bounded",
        "completed_route_progress",
        "rejoined_original_route_goal",
    )
    checks = {key: original.get(key) is True for key in required}
    checks["no_rejected_selection_provenance"] = not passing.get("rejections", ())
    if passing.get("selected_rows") or require_encounter:
        for key in (
            "selected_target_is_the_observed_human",
            "original_selection_clock_bounded",
        ):
            checks[key] = original.get(key) is True
    if require_encounter:
        checks.update(
            three_observed_policy_rows=original.get("three_observed_policy_rows")
            is True,
            encounter_switching_passed=(encounter or {}).get("passed") is True,
            intentional_wait_strategy_passed=(strategy or {}).get("passed") is True,
        )
    return dict(
        passed=all(checks.values()),
        checks=checks,
        original_passing_passed=passing.get("passed"),
        original_strategy_passed=(strategy or {}).get("passed"),
        selected_rows=len(passing.get("selected_rows", ())),
        scope="controller-variant passing comparison only; baseline side selection/curved bypass/intentional waiting are measured, not required; arrival, independent clearance, stopping, current perception and containment remain mandatory separately",
    )


def summarize_encounter_policy(records, *, require_transition=False, checked_wait=False):
    zero_rows = parallel_rows = 0
    rejected_wait_rows = 0
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
        if checked_wait and expected_yield:
            audit = detail.get("checked_wait_transition") or {}
            source_valid = (
                audit.get("source_timestamp_seconds") == t
                and set(audit.get("candidate_track_ids", ())) == expected_yield
                and audit.get("original_selection_timestamp_seconds")
                == passing.get("encounter_selected_timestamp")
            )
            if audit.get("accepted") is True:
                admission = detail.get("checked_wait_current_pose_admission") or {}
                source_valid &= (
                    audit.get("selected_policy") == "checked_zero_bias_wait_field"
                    and all(audit.get(k) is True for k in (
                        "exact_dense_accepted", "future_analytical_accepted", "first_analytical_accepted"))
                    and admission.get("accepted") is True
                    and admission.get("source_timestamp_seconds") == t
                )
            elif audit.get("accepted") is False:
                source_valid &= audit.get("selected_policy") == "normal_guidance_current_field"
                expected_yield = set()
                rejected_wait_rows += 1
            else:
                source_valid = False
            if not source_valid:
                errors.append(dict(time=t, reason="checked_wait_admission_mismatch"))
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
        rejected_wait_normal_fields=rejected_wait_rows,
        checked_wait_transition_enabled=checked_wait,
        rejections=errors,
        scope="per-row source/motion/lambda/flux audit only; route, body containment, physical perception and actual execution are separate",
    )

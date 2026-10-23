import copy

import pytest

from robot.evaluation.encounter_policy_validation import (
    summarize_encounter_policy,
    summarize_variant_passing,
)


def records():
    result = []
    for i in range(6):
        crossing = i < 3
        t = 1.0 + i * 0.1
        result.append(
            dict(
                sensor_timestamp=t,
                diagnostics=dict(
                    physical_CRA_lambda=1.25,
                    crossing_yield_track_ids=[7] if crossing else [],
                    motion_aware_passing=dict(preferences={7: "left"}),
                    encounter_motion_classification=dict(
                        source_timestamp_seconds=t,
                        tracks={
                            7: dict(
                                track_id=7,
                                source_timestamp_seconds=t,
                                classification="crossing" if crossing else "parallel",
                                consistent_observations=3,
                                acute_route_angle_degrees=90.0 if crossing else 0.0,
                            )
                        },
                    ),
                    applied_field_policies=[
                        dict(
                            track_or_group_id=7,
                            physical_CRA_lambda=1.25,
                            **{"lambda": 0.0 if crossing else 1.25},
                            boundary_flux=-1.7,
                            applied_field_risk=1.0,
                            encounter_guidance_ablation=dict(
                                parent_track_id=7, source_timestamp_seconds=t
                            ),
                        )
                    ],
                ),
            )
        )
    return result


def test_transition_audit_requires_current_crossing_zero_then_normal_parallel():
    r = summarize_encounter_policy(records(), require_transition=True)
    assert (
        r["passed"]
        and r["selected_crossing_zero_fields"] == 3
        and r["current_parallel_classifications"] == 3
    )
    assert not summarize_encounter_policy(records()[:3], require_transition=True)[
        "passed"
    ]


def guarded_records():
    result = records()
    for row in result[:3]:
        t, d = row["sensor_timestamp"], row["diagnostics"]
        d["motion_aware_passing"]["encounter_selected_timestamp"] = 1.
        d["checked_wait_transition"] = dict(
            accepted=True, source_timestamp_seconds=t, candidate_track_ids=[7],
            original_selection_timestamp_seconds=1.,
            selected_policy="checked_zero_bias_wait_field", exact_dense_accepted=True,
            future_analytical_accepted=True, first_analytical_accepted=True)
        d["checked_wait_current_pose_admission"] = dict(accepted=True, source_timestamp_seconds=t)
    return result


def test_guarded_variant_requires_both_source_and_current_pose_admission():
    r = guarded_records()
    assert summarize_encounter_policy(r, require_transition=True, checked_wait=True)["passed"]
    for key in ("checked_wait_transition", "checked_wait_current_pose_admission"):
        broken = copy.deepcopy(r)
        broken[0]["diagnostics"][key]["accepted"] = False
        assert not summarize_encounter_policy(broken, checked_wait=True)["passed"]


@pytest.mark.parametrize("corruption", [None, "source", "ids", "clock", "wrong_lambda"])
def test_guarded_normal_alternative_requires_original_rejected_candidate_audit(corruption):
    r = guarded_records()
    d = r[0]["diagnostics"]
    d["crossing_yield_track_ids"] = []
    d["applied_field_policies"][0]["lambda"] = 1.25
    audit = d["checked_wait_transition"]
    audit.update(accepted=False, selected_policy="normal_guidance_current_field")
    if corruption == "source":
        audit["source_timestamp_seconds"] = 0.
    elif corruption == "ids":
        audit["candidate_track_ids"] = [8]
    elif corruption == "clock":
        audit["original_selection_timestamp_seconds"] = 0.
    elif corruption == "wrong_lambda":
        d["applied_field_policies"][0]["lambda"] = 0.
    assert summarize_encounter_policy(r, checked_wait=True)["passed"] == (corruption is None)
    assert not summarize_encounter_policy(r)["passed"]  # Strict historical variant is unchanged.


@pytest.mark.parametrize("failure", ["stale", "parallel_zero", "flux", "unconfirmed"])
def test_transition_audit_rejects_stale_wrong_or_unconfirmed_policy(failure):
    r = copy.deepcopy(records())
    d = r[-1]["diagnostics"]
    if failure == "stale":
        d["encounter_motion_classification"]["source_timestamp_seconds"] -= 0.1
    if failure == "parallel_zero":
        d["applied_field_policies"][0]["lambda"] = 0.0
    if failure == "flux":
        d["applied_field_policies"][0]["boundary_flux"] = -2.0
    if failure == "unconfirmed":
        d["encounter_motion_classification"]["tracks"][7]["consistent_observations"] = 1
    assert not summarize_encounter_policy(r, require_transition=True)["passed"]


def baseline_passing():
    return dict(
        passed=False,
        selected_rows=[],
        rejections=[],
        checks=dict(
            fixed_lambda_and_flux_mapping=True,
            identity_prior_selection_provenance_bounded=True,
            completed_route_progress=True,
            rejoined_original_route_goal=True,
            three_observed_policy_rows=False,
            selected_target_is_the_observed_human=False,
            original_selection_clock_bounded=False,
            visible_trailing_side_arc=False,
            passes_behind_moving_actor=False,
        ),
    )


def test_comparison_accepts_baseline_without_proposed_selection_but_keeps_raw_failure():
    passing = baseline_passing()
    original = copy.deepcopy(passing)
    result = summarize_variant_passing(passing)
    assert result["passed"] and not result["original_passing_passed"]
    assert passing == original
    # The same missing feature cannot pass the proposed encounter variant.
    assert not summarize_variant_passing(
        passing,
        require_encounter=True,
        encounter=dict(passed=True),
        strategy=dict(passed=True),
    )["passed"]


@pytest.mark.parametrize(
    "key",
    [
        "fixed_lambda_and_flux_mapping",
        "identity_prior_selection_provenance_bounded",
        "completed_route_progress",
        "rejoined_original_route_goal",
    ],
)
@pytest.mark.parametrize("missing", [True, False])
def test_comparison_still_rejects_missing_or_failed_shared_checks(key, missing):
    passing = baseline_passing()
    if missing:
        del passing["checks"][key]
    else:
        passing["checks"][key] = False
    assert not summarize_variant_passing(passing)["passed"]


def test_any_actual_baseline_selection_still_requires_identity_and_original_clock():
    passing = baseline_passing()
    passing["selected_rows"] = [dict(track_id=7)]
    assert not summarize_variant_passing(passing)["passed"]
    passing["checks"].update(
        selected_target_is_the_observed_human=True,
        original_selection_clock_bounded=True,
    )
    assert summarize_variant_passing(passing)["passed"]
    passing["rejections"] = ["selected tangent does not trail original motion"]
    assert not summarize_variant_passing(passing)["passed"]


@pytest.mark.parametrize("failed", [None, "encounter", "strategy", "selection"])
def test_encounter_comparison_preserves_switching_wait_and_selection_gates(failed):
    passing = baseline_passing()
    passing["selected_rows"] = [dict(track_id=7)] * 3
    passing["checks"].update(
        three_observed_policy_rows=failed != "selection",
        selected_target_is_the_observed_human=True,
        original_selection_clock_bounded=True,
    )
    result = summarize_variant_passing(
        passing,
        require_encounter=True,
        encounter=dict(passed=failed != "encounter"),
        strategy=dict(passed=failed != "strategy"),
    )
    assert result["passed"] is (failed is None)

import copy

import pytest

from robot.evaluation.encounter_policy_validation import summarize_encounter_policy


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

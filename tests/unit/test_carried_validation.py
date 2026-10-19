"""Controlled report validation, not native sensor/perception acceptance."""

import copy

import pytest

from robot.evaluation.carried_object_validation import summarize_carried_groups
from robot.perception.human_mass_prior import HUMAN_MASS_PRIOR_SOURCE
from robot.perception.visual_mass_runtime import VISUAL_MASS_RUNTIME_SOURCE


def report_fixture():
    records = []
    truth = []
    for frame, t in enumerate((1.0, 1.2, 1.4)):

        def member(track, source, status, mass, phi):
            return {
                "attributes": {
                    "track_id": track,
                    "mass": {
                        "mass_kg": {
                            "value": mass,
                            "source": source,
                            "status": status,
                        }
                    },
                },
                "phi": phi,
                "diagnostics": {
                    "accepted_instance": True,
                    "policy_return_count": 12,
                    "source_time": t,
                    "source_frame_id": frame,
                    "factor_scores": {
                        name: {"score": 0.25}
                        for name in ("potential", "kinetic", "contact", "thermal")
                    },
                    "instance_centroid_world_xyz": [t * 0.5, 0.2, 1.1],
                },
            }

        records.append(
            {
                "time": t,
                "camera_time": t,
                "phase": "move",
                "completion_age_sim_seconds": 0.2,
                "carried_object_groups": {
                    "members": {"human": 1, "object": 2},
                    "confirmed_groups": [{"members": [1, 2], "phi": 0.8}],
                    "relationship": {"confirmed": True},
                },
                "occupancy_policies": [
                    {"track_or_group_id": k, "phi": 0.8} for k in (1, 2)
                ],
                "objects": [
                    member(1, HUMAN_MASS_PRIOR_SOURCE, "prior", 70.0, 0.6),
                    member(2, VISUAL_MASS_RUNTIME_SOURCE, "inferred", 2.0, 0.8),
                ],
            }
        )
        truth.append({"time": t, "bowl_center_xyz": [t * 0.5, 0.2, 1.1]})
    return records, truth


def test_independent_group_gate_accepts_fresh_separate_maximum_policy():
    records, truth = report_fixture()
    original = copy.deepcopy(records)
    result = summarize_carried_groups(records, truth)
    assert result["passed"]
    assert result["confirmed_row_count"] == 3
    assert records == original


@pytest.mark.parametrize(
    "case,check",
    [
        ("mixed_mass", "separate_mass_provenance"),
        ("wrong_field", "maximum_shared_field_policy"),
        ("stale_epoch", "fresh_distinct_member_geometry"),
        ("old_report", "fresh_distinct_member_geometry"),
        ("stationary", "moving_confirmed_relationship"),
        ("late", "fresh_group_completion"),
        ("negative_age", "fresh_group_completion"),
        ("wrong_bowl", "independent_bowl_geometry"),
        ("missing_truth", "independent_bowl_geometry"),
        ("missing_factor", "four_numeric_factors_each"),
    ],
)
def test_independent_gate_rejects_incomplete_or_incorrect_reports(case, check):
    records, truth = report_fixture()
    row = records[0]
    obj = row["objects"][1]
    if case == "mixed_mass":
        obj["attributes"]["mass"]["mass_kg"]["source"] = HUMAN_MASS_PRIOR_SOURCE
    elif case == "wrong_field":
        row["occupancy_policies"][0]["phi"] = 0.6
    elif case == "stale_epoch":
        obj["diagnostics"]["source_time"] = 0.9
    elif case == "old_report":
        obj["diagnostics"].pop("source_time")
    elif case == "stationary":
        row["phase"] = "stationary"
    elif case == "late":
        row["completion_age_sim_seconds"] = 0.36
    elif case == "negative_age":
        row["completion_age_sim_seconds"] = -0.1
    elif case == "wrong_bowl":
        obj["diagnostics"]["instance_centroid_world_xyz"] = [9, 9, 9]
    elif case == "missing_truth":
        truth = []
    elif case == "missing_factor":
        obj["diagnostics"]["factor_scores"].pop("potential")
    result = summarize_carried_groups(records, truth)
    assert not result["passed"]
    assert not result["checks"][check]

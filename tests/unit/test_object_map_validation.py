from copy import deepcopy

import pytest

from robot.evaluation.object_map_validation import (
    prior_sensitivity,
    summarize_fixture_assessments,
    bounded_fixture_checks,
    kinetic_actor_fixture_checks,
)
from robot.risk.cra import PhysicalRiskProfile, RiskFactorEstimate
from robot.safety.cra_safe_sage_policy import CRASafeSAGEPolicy


def object_row(track_id=1):
    control = (
        CRASafeSAGEPolicy()
        .evaluate(
            PhysicalRiskProfile(
                thermal=RiskFactorEstimate(0.8, 1, 0.8, "observed_plume_proxy"),
                contact=RiskFactorEstimate.unknown(source="test_missing"),
                potential=RiskFactorEstimate.unknown(source="test_missing"),
                kinetic=RiskFactorEstimate.unknown(source="test_missing"),
            )
        )
        .as_dict()
    )
    return {
        "track_id": track_id,
        "cra_control": control,
        "diagnostics": {"centroid_world_xy": [1, 0], "accepted_instance": True},
    }


def test_prior_sensitivity_keeps_high_evidence_and_does_not_mutate_sources():
    control = object_row()["cra_control"]
    original = deepcopy(control)
    values = prior_sensitivity(control)
    assert control == original
    assert values["0.1"]["phi"] < values["0.25"]["phi"] < values["0.5"]["phi"]
    assert all(
        row["factors"]["thermal"]["conservative_risk"] == pytest.approx(0.8)
        for row in values.values()
    )


def test_fixture_report_rejects_ambiguous_matches_and_uses_observed_height_for_matching_only():
    fixture = {
        "name": "raised",
        "center_world_xy": [1, 0],
        "center_world_xyz": [1, 0, 0.6],
    }
    cube, sphere = object_row(1), object_row(2)
    cube["diagnostics"]["instance_centroid_world_xyz"] = [1, 0, 0.15]
    sphere["diagnostics"]["instance_centroid_world_xyz"] = [1, 0, 0.6]
    report = summarize_fixture_assessments(
        [{"objects": [cube, sphere]}], [fixture], 0.25
    )
    assert report["fixtures"]["raised"]["track_ids"] == [2]
    sphere.pop(
        "track_id"
    )  # malformed objects must fail, not produce a partial success report
    with pytest.raises(KeyError):
        summarize_fixture_assessments([{"objects": [sphere]}], [fixture], 0.25)
    rows = summarize_fixture_assessments(
        [{"objects": [object_row(1), object_row(2)]}], [fixture]
    )
    assert rows["fixtures"]["raised"]["latest"] is None
    assert rows["fixtures"]["raised"]["ambiguous_updates"] == 1


@pytest.mark.parametrize(
    "status, accepted",
    [
        ("inferred", True),
        ("prior", True),
        ("oracle", False),
        ("fallback", False),
        ("unavailable", False),
    ],
)
def test_fixture_evidence_gate_does_not_promote_oracle_or_fallback_mass(
    status, accepted
):
    row = object_row()
    row["attributes"] = {
        "mass": {"mass_kg": {"value": 1, "status": status}},
        "geometry": {"fall_height_m": {"value": 0, "status": "prior"}},
    }
    row["diagnostics"]["factor_scores"] = {
        "contact": {"mode": "evidence_supported"},
        "potential": {"mode": "evidence_supported"},
    }
    report = {"fixtures": {"ground": {"latest": row, "accepted_visual_updates": 3}}}
    assert bounded_fixture_checks(report)["ground_mass_contact"] is accepted


@pytest.mark.parametrize(
    "status, accepted",
    [("inferred", True), ("prior", False), ("oracle", False), ("unavailable", False)],
)
def test_kinetic_actor_gate_requires_fresh_visual_inputs_and_labels_missing_potential(
    status, accepted
):
    row = object_row()
    row["attributes"] = {
        "mass": {"mass_kg": {"value": 1, "status": status}},
        "motion": {"relative_speed_mps": {"value": 0.01, "status": "inferred"}},
    }
    row["diagnostics"].update(
        policy_return_count=10,
        factor_scores={
            "contact": {
                "mode": "evidence_supported",
                "basis": "visual_silhouette_geometry_proxy",
            },
            "kinetic": {"mode": "evidence_supported"},
            "potential": {"mode": "configured_prior"},
        },
    )
    report = {"fixtures": {"ground": {"latest": row, "accepted_visual_updates": 3}}}
    assert all(kinetic_actor_fixture_checks(report).values()) is accepted
    row["diagnostics"]["factor_scores"]["potential"]["mode"] = "evidence_supported"
    assert not kinetic_actor_fixture_checks(report)[
        "unsupported_potential_explicit_prior"
    ]
    row["diagnostics"]["factor_scores"]["contact"]["basis"] = "physical"
    assert not kinetic_actor_fixture_checks(report)["visual_contact_proxy"]

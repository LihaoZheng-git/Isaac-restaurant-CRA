"""Independent fixture matching and offline prior sensitivity diagnostics.

Authored centers identify report rows only; they never supply physical inputs.
Ambiguous cluster matches are rejected instead of selecting a convenient score.
"""

import math

import numpy as np

from robot.risk.cra import (
    CRAAssessmentConfig,
    CRARiskAssessor,
    PhysicalRiskProfile,
    PhysicalRiskVector,
    RiskFactorEstimate,
)


def prior_sensitivity(control_record, values=(0.10, 0.25, 0.50)):
    factors = control_record["cra"]["factors"]
    profile = PhysicalRiskProfile(
        **{
            name: RiskFactorEstimate(
                severity=item["severity"],
                interaction_likelihood=item["interaction_likelihood"],
                confidence=item["confidence"],
                source=item["source"],
                evidence=tuple(item["evidence"]),
            )
            for name, item in factors.items()
        }
    )
    return {
        str(value): CRARiskAssessor(
            CRAAssessmentConfig(
                fallback_risks=PhysicalRiskVector(value, value, value, value)
            )
        )
        .assess(profile)
        .as_dict()
        for value in values
    }


def summarize_fixture_assessments(records, fixtures, maximum_distance_m=0.40):
    if not math.isfinite(maximum_distance_m) or maximum_distance_m <= 0:
        raise ValueError("match distance must be positive and finite")
    rows = {}
    for fixture in fixtures:
        accepted, ambiguous = [], 0
        for record in records:
            matches = []
            for item in record["objects"]:
                diagnostic = item["diagnostics"]
                measured = diagnostic.get("instance_centroid_world_xyz")
                target = fixture.get("center_world_xyz")
                if measured is None or target is None:
                    measured, target = (
                        diagnostic["centroid_world_xy"],
                        fixture["center_world_xy"],
                    )
                if np.linalg.norm(np.asarray(measured) - target) <= maximum_distance_m:
                    matches.append(item)
            if len(matches) > 1:
                ambiguous += 1
            elif len(matches) == 1:
                accepted.append(matches[0])
        latest = accepted[-1] if accepted else None
        rows[fixture["name"]] = {
            "uniquely_matched_updates": len(accepted),
            "ambiguous_updates": ambiguous,
            "accepted_visual_updates": sum(
                item["diagnostics"]["accepted_instance"] for item in accepted
            ),
            "track_ids": sorted({item["track_id"] for item in accepted}),
            "phi_range": [
                min(item["cra_control"]["cra"]["phi"] for item in accepted),
                max(item["cra_control"]["cra"]["phi"] for item in accepted),
            ]
            if accepted
            else None,
            "latest": latest,
            "offline_prior_sensitivity": prior_sensitivity(latest["cra_control"])
            if latest
            else None,
        }
    return {
        "scope": "oracle fixture-center matching for evaluation only; unchanged recorded factor estimates; prior sensitivity is offline without wheel motion",
        "fixtures": rows,
    }


def bounded_fixture_checks(report):
    checks = {}
    for name, row in report["fixtures"].items():
        latest = row["latest"]
        accepted_statuses = {"measured", "inferred", "prior"}
        checks[name + "_mass_contact"] = bool(
            latest
            and row["accepted_visual_updates"] >= 3
            and latest["attributes"]["mass"]["mass_kg"]["value"] is not None
            and latest["attributes"]["mass"]["mass_kg"]["status"] in accepted_statuses
            and latest["diagnostics"]["factor_scores"]["contact"]["mode"]
            == "evidence_supported"
        )
        if latest and name in {"ground", "raised", "plume"}:
            drop = latest["attributes"]["geometry"]["fall_height_m"]["value"]
            checks[name + "_fall"] = (
                latest["diagnostics"]["factor_scores"]["potential"]["mode"]
                == "evidence_supported"
                and (
                    latest["attributes"]["geometry"]["fall_height_m"]["status"]
                    in accepted_statuses
                )
                and drop is not None
                and (drop > 0.1 if name == "raised" else drop <= 0.05)
            )
        if name == "plume":
            cue = (
                latest["attributes"]["thermal"]["heat_source_likelihood"]["value"]
                if latest
                else None
            )
            checks["positive_plume_cue"] = (
                cue is not None
                and cue > 0.2
                and latest["attributes"]["thermal"]["heat_source_likelihood"]["status"]
                in accepted_statuses
            )
    return checks

from copy import deepcopy

import pytest

from test_carried_validation import report_fixture
from robot.evaluation.human_body_contact_validation import summarize_human_body_contact
from robot.perception.human_body_contact import HUMAN_BODY_CONTACT_SOURCE


def rows():
    records, _ = report_fixture()
    for r in records:
        obj = r["objects"][0]
        obj["phi"] = 1 - 0.9 * 0.75**3
        obj["diagnostics"]["factor_scores"]["contact"].update(
            score=0.1,
            mode="configured_human_body_contact_prior",
            confidence=0,
            raw_contact_policy_score=1.0,
            raw_aggregate_phi=1.0,
            human_body_prior={
                "value": 0.1,
                "confidence": 0,
                "status": "prior",
                "source": HUMAN_BODY_CONTACT_SOURCE,
                "timestamp_seconds": r["camera_time"],
                "evidence": ["track_id=1"],
            },
        )
        # The other fixture factors need only numeric scores for this audit.
        r["objects"][1]["diagnostics"]["factor_scores"]["contact"]["mode"] = (
            "configured_prior"
        )
    return records


def test_contact_audit_accepts_fresh_body_priors_and_original_phi_arithmetic():
    records = rows()
    original = deepcopy(records)
    report = summarize_human_body_contact(records, 0.1)
    assert report["passed"] and report["raw_saturated_rows"] == 3
    assert records == original


@pytest.mark.parametrize(
    "problem", ["leak", "stale", "measured", "phi", "raw_phi", "score"]
)
def test_contact_audit_rejects_wrong_provenance_or_arithmetic(problem):
    records = rows()
    obj = records[0]["objects"][0]
    d = obj["diagnostics"]["factor_scores"]["contact"]
    if problem == "leak":
        obj["attributes"]["mass"]["mass_kg"]["source"] = "catalog_mass"
    elif problem == "stale":
        d["human_body_prior"]["timestamp_seconds"] -= 0.1
    elif problem == "measured":
        d["human_body_prior"]["status"] = "measured"
    elif problem == "phi":
        obj["phi"] = 0.2
    elif problem == "raw_phi":
        d["raw_aggregate_phi"] = 0.5
    elif problem == "score":
        d["score"] = 0
    assert not summarize_human_body_contact(records, 0.1)["passed"]

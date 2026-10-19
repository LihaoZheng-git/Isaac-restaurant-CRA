"""Audit declared human contact priors without treating them as measurements."""

import math

from robot.perception.human_body_contact import HUMAN_BODY_CONTACT_SOURCE
from robot.perception.human_mass_prior import HUMAN_MASS_PRIOR_SOURCE


def summarize_human_body_contact(records, expected_score):
    body, leaks, mismatches = [], [], []
    for row in records:
        for obj in row["objects"]:
            details = obj["diagnostics"]["factor_scores"]
            contact = details["contact"]
            human = (
                obj["attributes"]["mass"]["mass_kg"]["source"]
                == HUMAN_MASS_PRIOR_SOURCE
            )
            if contact["mode"] != "configured_human_body_contact_prior":
                continue
            prior = contact["human_body_prior"]
            if not human:
                leaks.append(
                    {"time": row["time"], "track_id": obj["attributes"]["track_id"]}
                )
            fresh = (
                obj["diagnostics"]["accepted_instance"]
                and obj["diagnostics"]["policy_return_count"] > 0
                and abs(prior["timestamp_seconds"] - row["camera_time"]) < 1e-9
                and f"track_id={obj['attributes']['track_id']}" in prior["evidence"]
            )
            configured = (
                prior["status"] == "prior"
                and prior["source"] == HUMAN_BODY_CONTACT_SOURCE
                and prior["confidence"] == contact["confidence"] == 0
                and contact["score"] == prior["value"] == expected_score
            )
            product = math.prod(1 - d["score"] for d in details.values())
            phi_ok = math.isclose(obj["phi"], 1 - product, abs_tol=1e-12)
            original_product = math.prod(
                1 - d["score"] for name, d in details.items() if name != "contact"
            )
            original_phi = 1 - original_product * (
                1 - contact["raw_contact_policy_score"]
            )
            raw_ok = math.isclose(
                original_phi, contact["raw_aggregate_phi"], abs_tol=1e-12
            )
            if not phi_ok or not raw_ok:
                mismatches.append(row["time"])
            body.append(
                {
                    "time": row["time"],
                    "phase": row["phase"],
                    "track_id": obj["attributes"]["track_id"],
                    "fresh_person_ownership": fresh,
                    "explicit_zero_measurement_confidence_prior": configured,
                    "raw_contact_score": contact["raw_contact_policy_score"],
                    "body_contact_score": contact["score"],
                    "original_phi": original_phi,
                    "phi": obj["phi"],
                }
            )
    checks = {
        "enough_fresh_body_prior_rows": len(body) >= 3,
        "fresh_same_person_source": bool(body)
        and all(r["fresh_person_ownership"] for r in body),
        "explicit_configured_prior": bool(body)
        and all(r["explicit_zero_measurement_confidence_prior"] for r in body),
        "no_nonhuman_prior_assignment": not leaks,
        "original_aggregation_and_raw_diagnostics": bool(body) and not mismatches,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "prior_score": expected_score,
        "body_prior_row_count": len(body),
        "moving_body_prior_row_count": sum(r["phase"] == "move" for r in body),
        "raw_saturated_rows": sum(r["raw_contact_score"] >= 1 - 1e-12 for r in body),
        "body_prior_rows": body,
        "nonhuman_leaks": leaks,
        "arithmetic_mismatches": mismatches,
        "scope": "declared blunt-body prior and unchanged phi arithmetic; not calibrated human harm, physical sharpness or new wheel acceptance",
    }

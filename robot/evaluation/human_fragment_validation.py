"""Provenance audit, not independent raw geometry or navigation acceptance."""

import math

import numpy as np

from robot.perception.human_fragment_coverage import bounds_corners


def summarize_short_fragment_coverage(records):
    sources, proofs, field_covered, errors = 0, 0, 0, []
    for row in records:
        diagnostic = row.get("diagnostics", row)
        audit = diagnostic.get("human_fragment_coverage_prior")
        if audit is None:
            errors.append("missing configured fragment audit")
            continue
        sources += 1
        if audit.get("short_fragment_coverage") is not True:
            errors.append("short fragment option not active")
        for proof in audit["covered"]:
            proofs += 1
            original, now = proof["original_source_time"], audit["source_time"]
            extent = np.linalg.norm(bounds_corners(proof["original_bounds_min_world_xy"],
                proof["original_bounds_max_world_xy"]) - proof["original_parent_center_world_xy"], axis=1).max()
            if not (proof["short_fragment_coverage"] is True and proof["status"] == "prior"
                and proof["original_parent_source_time"] == original
                and proof["current_source_time"] == now
                and 0 < now - original <= .35 + 1e-9
                and math.isclose(proof["missing_seconds"], now-original, abs_tol=1e-9)
                and math.isclose(proof["motion_expansion_m"],
                    (now-original)*proof["maximum_point_speed_prior_mps"], abs_tol=1e-9)
                and 0 < proof["parent_gap_seconds"] <= proof["maximum_parent_gap_seconds"] <= .35
                and extent <= proof["original_parent_radius_m"] + 1e-9
                and proof["maximum_covered_extent_m"] <= proof["current_body_radius_m"] + 1e-9
                and 0 <= proof["original_risk"] <= 1):
                errors.append("invalid whole-region provenance")
            check = proof.get("field_coverage")
            if check is None:
                errors.append("missing actual current-grid check")
            elif check["covered"]:
                field_covered += 1
                if check.get("checked_cell_count", 0) <= 0:
                    errors.append("empty field coverage")
    return dict(passed=bool(sources and not errors), sources=sources,
        geometric_proofs=proofs, field_covered_proofs=field_covered, errors=errors,
        scope="source/proof/current-grid diagnostic provenance; no independent raw replay or wheel acceptance")

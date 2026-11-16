"""Independent parked-robot carried-instance and field-policy component gate."""

import math
import numpy as np

from robot.perception.human_mass_prior import HUMAN_MASS_PRIOR_SOURCE
from robot.perception.visual_mass_runtime import VISUAL_MASS_RUNTIME_SOURCE


def summarize_carried_groups(records, carry_truth):
    separate, groups = [], []
    times = np.array([r["time"] for r in carry_truth])
    centers = (
        np.array([r["bowl_center_xyz"] for r in carry_truth])
        if carry_truth
        else np.empty((0, 3))
    )
    if len(times) and (
        not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
        or not np.isfinite(centers).all()
    ):
        raise ValueError("independent bowl truth must be finite and ordered")
    for row in records:
        d = row.get("carried_object_groups") or {}
        ids = d.get("members")
        if not ids:
            continue
        by_id = {o["attributes"]["track_id"]: o for o in row["objects"]}
        if ids["human"] not in by_id or ids["object"] not in by_id:
            continue
        human, obj = by_id[ids["human"]], by_id[ids["object"]]
        hm, om = (o["attributes"]["mass"]["mass_kg"] for o in (human, obj))
        current = all(
            o["diagnostics"]["accepted_instance"]
            and o["diagnostics"]["policy_return_count"] >= 8
            and abs(o["diagnostics"].get("source_time", math.inf) - row["camera_time"])
            < 1e-6
            for o in (human, obj)
        )
        mass = (
            hm["source"] == HUMAN_MASS_PRIOR_SOURCE
            and hm["status"] == "prior"
            and om["source"] == VISUAL_MASS_RUNTIME_SOURCE
            and om["status"] == "inferred"
            and om["value"] > 0
        )
        error = None
        if len(times) >= 2 and times[0] <= row["time"] <= times[-1]:
            oracle = np.array(
                [np.interp(row["time"], times, centers[:, i]) for i in range(3)]
            )
            error = float(
                np.linalg.norm(
                    np.array(obj["diagnostics"]["instance_centroid_world_xyz"]) - oracle
                )
            )
        numeric = all(
            len(o["diagnostics"]["factor_scores"]) == 4
            and all(
                0 <= v["score"] <= 1 for v in o["diagnostics"]["factor_scores"].values()
            )
            for o in (human, obj)
        )
        item = {
            "time": row["time"],
            "phase": row["phase"],
            "human_phi": human["phi"],
            "object_phi": obj["phi"],
            "fresh_separate_ownership": current,
            "independent_mass_provenance": mass,
            "four_numeric_factors": numeric,
            "bowl_center_error_m": error,
            "human_mass_kg_prior": hm["value"],
            "object_rough_mass_kg": om["value"],
            "completion_age_seconds": row["completion_age_sim_seconds"],
        }
        separate.append(item)
        for group in d.get("confirmed_groups", []):
            fields = {
                p["track_or_group_id"]: p
                for p in row.get("occupancy_policies", []) or []
            }
            expected = max(human["phi"], obj["phi"])
            policy_ok = (
                set(group["members"]) == {ids["human"], ids["object"]}
                and math.isclose(group["phi"], expected, abs_tol=1e-12)
                and all(
                    k in fields
                    and math.isclose(fields[k]["phi"], expected, abs_tol=1e-12)
                    for k in group["members"]
                )
            )
            groups.append(
                {
                    **item,
                    "group_phi": group["phi"],
                    "both_member_field_policies_are_maximum": policy_ok,
                    "relationship": d["relationship"],
                }
            )
    checks = {
        "separate_member_observations": len(separate) >= 3,
        "fresh_distinct_member_geometry": bool(separate)
        and all(r["fresh_separate_ownership"] for r in separate),
        "separate_mass_provenance": bool(separate)
        and all(r["independent_mass_provenance"] for r in separate),
        "four_numeric_factors_each": bool(separate)
        and all(r["four_numeric_factors"] for r in separate),
        "independent_bowl_geometry": bool(separate)
        and all(
            r["bowl_center_error_m"] is not None and r["bowl_center_error_m"] <= 0.25
            for r in separate
        ),
        "moving_confirmed_relationship": sum(r["phase"] == "move" for r in groups) >= 3,
        "maximum_shared_field_policy": bool(groups)
        and all(r["both_member_field_policies_are_maximum"] for r in groups),
        "fresh_group_completion": bool(groups)
        and all(
            math.isfinite(r["completion_age_seconds"])
            and 0 <= r["completion_age_seconds"] <= 0.35 + 1e-9
            for r in groups
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "separate_rows": separate,
        "confirmed_rows": groups,
        "separate_row_count": len(separate),
        "confirmed_row_count": len(groups),
        "scope": "parked-robot one-adult compact carried-object component; inferred relationship, rough object mass/adult prior and other factor priors; not grasp, drop/temperature, crowded or robot navigation acceptance",
    }

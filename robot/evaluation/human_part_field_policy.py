"""Opt-in current-human-part control-policy consistency, never factor copying."""

import copy
from dataclasses import replace
import math

from robot.perception.human_body_envelope import HUMAN_ENVELOPE_SOURCE


def apply_human_part_field_policy(
    guidance,
    audit,
    associations,
    identities,
    *,
    sensor_timestamp_seconds,
    mask_boundary_prior_pixels=0,
):
    """Raise only current part field risk and share its verified parent's side.

    This consumes the existing current positive part proof and verified human
    flux ablation. Physical CRA, observed class, occupancy and all source epochs
    stay separate. A child with higher applied risk keeps it. No association is
    retained or applied after its original current scan.
    """
    if (
        not math.isfinite(sensor_timestamp_seconds)
        or sensor_timestamp_seconds < 0
        or isinstance(mask_boundary_prior_pixels, bool)
        or mask_boundary_prior_pixels not in (0, 1)
    ):
        raise ValueError(
            "current human-part policy requires a valid source/boundary scope"
        )
    rows = {r["track_or_group_id"]: r for r in audit}
    parents = {r["track_id"]: r for r in identities}
    children = [p["child_track_id"] for p in associations]
    if (
        len(rows) != len(audit)
        or len(parents) != len(identities)
        or len(set(children)) != len(children)
        or any(p["parent_track_id"] in children for p in associations)
    ):
        raise ValueError("ambiguous or nested current human-part policy")
    changed, result = dict(guidance), copy.deepcopy(audit)
    output_rows = {r["track_or_group_id"]: r for r in result}
    for proof in associations:
        child, parent = proof["child_track_id"], proof["parent_track_id"]
        identity = parents.get(parent)
        strict = (
            proof["basis"] == "fresh_person_mask_whole_positive_small_fragment"
            and proof["status"] == "inferred"
        )
        boundary = (
            mask_boundary_prior_pixels == 1
            and proof["basis"]
            == "fresh_person_mask_boundary_prior_positive_small_fragment"
            and proof["status"] == "prior"
            and proof.get("mask_boundary_prior_pixels") == 1
            and 0.95 <= proof.get("child_original_mask_coverage", -1) < 1
            and 0
            < proof.get("child_boundary_return_count", 0)
            <= 0.05 * proof["child_positive_count"] + 1e-9
            and proof.get("all_child_returns_inside_bounded_mask") is True
        )
        if (
            child == parent
            or child not in guidance
            or parent not in guidance
            or child not in rows
            or parent not in rows
            or identity is None
            or not (strict or boundary)
            or proof["source_timestamp_seconds"] != sensor_timestamp_seconds
            or not 0
            <= proof["camera_timestamp_seconds"] - sensor_timestamp_seconds
            <= 0.15 + 1e-9
            or identity.get("lidar_timestamp_seconds") != sensor_timestamp_seconds
            or identity.get("source") != HUMAN_ENVELOPE_SOURCE
            or identity.get("status") != "prior"
            or identity.get("current_support_count") != proof["parent_positive_count"]
            or identity.get("original_person_camera_timestamp")
            != proof["original_person_camera_timestamp"]
            or identity.get("original_person_camera_frame_id")
            != proof["original_person_camera_frame_id"]
            or not math.isclose(
                identity["original_person_camera_timestamp"]
                + identity["identity_age_seconds"],
                proof["camera_timestamp_seconds"],
                abs_tol=1e-9,
            )
            or not 3
            <= proof["child_positive_count"]
            <= 0.1 * proof["parent_positive_count"]
            or not 0 <= proof["maximum_child_to_parent_3d_distance_m"] <= 0.3
            or not 0.9 <= proof["parent_mask_coverage"] <= 1
            or not 0.5 <= proof["person_confidence"] <= 1
            or rows[parent]
            .get("human_flux_ablation", {})
            .get("lidar_timestamp_seconds")
            != sensor_timestamp_seconds
        ):
            raise ValueError(
                "human-part field policy requires a current positive proof and verified parent flux"
            )
        a, b = guidance[child], guidance[parent]
        if (
            a.tangential_strength != 1.25
            or b.tangential_strength != 1.25
            or any(
                not math.isclose(p.boundary_flux, -(0.5 + 1.2 * p.risk), abs_tol=1e-12)
                for p in (a, b)
            )
        ):
            raise ValueError(
                "human-part field policy requires fixed lambda and the shared flux mapping"
            )
        risk = max(a.risk, b.risk)
        changed[child] = replace(
            a,
            safety_parameters=replace(
                a.safety_parameters, risk=risk, boundary_flux_magnitude=0.5 + 1.2 * risk
            ),
            passing_side=b.passing_side,
        )
        output_rows[child].update(
            applied_field_risk=risk,
            boundary_flux=changed[child].boundary_flux,
            applied_passing_side=b.passing_side.value,
            human_part_control_policy_prior={
                "status": "prior",
                "basis": "current_positive_visual_part_verified_human_control_policy",
                "parent_track_id": parent,
                "association": copy.deepcopy(proof),
                "parent_control_source": copy.deepcopy(
                    rows[parent]["human_flux_ablation"]
                ),
                "child_physical_cra_phi": rows[child]["cra_phi"],
                "scope": "current control policy only; no physical factor/class/SAM/geometry transfer or original-clock renewal",
            },
        )
    return changed, result

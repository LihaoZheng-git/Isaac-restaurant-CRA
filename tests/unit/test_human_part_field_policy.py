import copy
from dataclasses import replace

import pytest

from robot.evaluation.human_part_field_policy import apply_human_part_field_policy
from robot.perception.human_body_envelope import HUMAN_ENVELOPE_SOURCE
from robot.safety.class_safety_policy import ClassSafetyParameters
from robot.safety.safe_sage_guidance_policy import SafeSAGEGuidanceParameters


def parameters(risk, side):
    return SafeSAGEGuidanceParameters(
        ClassSafetyParameters(risk, 0.5 + 1.2 * risk), 1.25, side
    )


def inputs(boundary=False):
    guide = {
        7: parameters(1.0, "left"),
        8: parameters(0.68359375, "right"),
        9: parameters(0.2, "right"),
    }
    identity = dict(
        track_id=7,
        source=HUMAN_ENVELOPE_SOURCE,
        status="prior",
        lidar_timestamp_seconds=1.0,
        current_support_count=400,
        original_person_camera_timestamp=0.8,
        original_person_camera_frame_id=2,
        identity_age_seconds=0.25,
    )
    audit = [
        dict(
            track_or_group_id=k,
            cra_phi=0.6 if k == 7 else v.risk,
            applied_field_risk=v.risk,
            observed_class="unknown",
            class_basis="unavailable_class",
            boundary_flux=v.boundary_flux,
            **(
                {
                    "human_flux_ablation": dict(
                        lidar_timestamp_seconds=1.0,
                        original_person_camera_timestamp=0.8,
                        requested_boundary_flux=-1.7,
                    )
                }
                if k == 7
                else {}
            ),
        )
        for k, v in guide.items()
    ]
    proof = dict(
        child_track_id=8,
        parent_track_id=7,
        source_timestamp_seconds=1.0,
        camera_timestamp_seconds=1.05,
        original_person_camera_timestamp=0.8,
        original_person_camera_frame_id=2,
        child_positive_count=37,
        parent_positive_count=400,
        maximum_child_to_parent_3d_distance_m=0.277,
        parent_mask_coverage=1.0,
        person_confidence=0.83,
        status="prior" if boundary else "inferred",
        basis="fresh_person_mask_boundary_prior_positive_small_fragment"
        if boundary
        else "fresh_person_mask_whole_positive_small_fragment",
    )
    if boundary:
        proof.update(
            mask_boundary_prior_pixels=1,
            child_original_mask_coverage=36 / 37,
            child_boundary_return_count=1,
            all_child_returns_inside_bounded_mask=True,
        )
    return guide, audit, [proof], [identity]


@pytest.mark.parametrize("boundary", [False, True])
def test_current_part_shares_control_side_and_flux_without_physical_or_class_transfer(
    boundary,
):
    args = inputs(boundary)
    original = copy.deepcopy(args)
    result, audit = apply_human_part_field_policy(
        *args, sensor_timestamp_seconds=1.0, mask_boundary_prior_pixels=int(boundary)
    )
    assert result[8].passing_side.value == "left" and result[8].boundary_flux == -1.7
    assert result[8].tangential_strength == 1.25
    assert result[7] is args[0][7] and result[9] is args[0][9]
    child = next(r for r in audit if r["track_or_group_id"] == 8)
    assert child["cra_phi"] == 0.68359375 and child["observed_class"] == "unknown"
    assert child["applied_field_risk"] == 1.0
    assert child["human_part_control_policy_prior"]["status"] == "prior"
    assert (
        child["human_part_control_policy_prior"]["association"][
            "original_person_camera_timestamp"
        ]
        == 0.8
    )
    assert args == original


def test_higher_child_applied_risk_is_never_lowered():
    guide, audit, proofs, identities = inputs()
    guide[7] = parameters(0.3, "left")
    guide[8] = parameters(0.9, "right")
    for row in audit:
        value = guide[row["track_or_group_id"]]
        row.update(applied_field_risk=value.risk, boundary_flux=value.boundary_flux)
    audit[0]["cra_phi"] = 0.2
    audit[0]["human_flux_ablation"]["requested_boundary_flux"] = -0.86
    result, _ = apply_human_part_field_policy(
        guide, audit, proofs, identities, sensor_timestamp_seconds=1.0
    )
    assert result[8].risk == 0.9 and result[8].boundary_flux == pytest.approx(-1.58)


@pytest.mark.parametrize(
    "problem",
    [
        "stale",
        "unverified_parent",
        "source",
        "wrong_identity_epoch",
        "wrong_frame",
        "duplicate",
        "nested",
        "large_part",
        "missing_parent_flux",
        "unsupported_boundary",
        "lambda",
    ],
)
def test_invalid_or_unsupported_current_part_never_changes_field_policy(problem):
    guide, audit, proofs, identities = inputs(
        boundary=problem == "unsupported_boundary"
    )
    if problem == "stale":
        proofs[0]["source_timestamp_seconds"] = 0.9
    elif problem == "unverified_parent":
        identities = []
    elif problem == "source":
        identities[0]["source"] = "oracle_actor"
    elif problem == "wrong_identity_epoch":
        proofs[0]["original_person_camera_timestamp"] = 0.7
    elif problem == "wrong_frame":
        proofs[0]["original_person_camera_frame_id"] = 1
    elif problem == "duplicate":
        proofs += copy.deepcopy(proofs)
    elif problem == "nested":
        proofs.append({**proofs[0], "child_track_id": 7, "parent_track_id": 9})
    elif problem == "large_part":
        proofs[0]["child_positive_count"] = 41
    elif problem == "missing_parent_flux":
        audit[0].pop("human_flux_ablation")
    elif problem == "lambda":
        guide[8] = replace(guide[8], tangential_strength=0.5)
    original = copy.deepcopy((guide, audit, proofs, identities))
    with pytest.raises(ValueError):
        apply_human_part_field_policy(
            guide, audit, proofs, identities, sensor_timestamp_seconds=1.0
        )
    assert (guide, audit, proofs, identities) == original

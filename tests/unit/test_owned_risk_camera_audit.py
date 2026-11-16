"""Only documented zero-current-owner withholding changes legacy audit rules."""

from copy import deepcopy

import pytest

from scripts.evaluation.audit_owned_risk_camera import (
    verify_camera_metadata,
    verify_camera_source,
    verify_snapshot_clock,
)


def fixture(count=4):
    modes = {
        n: "evidence_supported" for n in ("thermal", "contact", "potential", "kinetic")
    }
    obj = dict(
        track_id=7,
        attributes=dict(timestamp_seconds=1.1),
        cra_control=dict(
            cra=dict(phi=0.4, aggregation="restaurant_evidence_weighted_noisy_or")
        ),
        diagnostics=dict(
            accepted_instance=True,
            policy_return_count=count,
            factor_scores={k: dict(mode=v) for k, v in modes.items()},
        ),
    )
    field = dict(sensor_timestamp=1.0, objects=[obj], policy_phi={7: 0.7})
    meta = dict(
        sensor_frame_id=10,
        sensor_timestamp_seconds=1.0,
        camera_frame_id=20,
        camera_timestamp_seconds=1.1,
        rejected=[],
        masks=[dict(track_id=7, phi=0.4, applied_phi=0.7, factor_modes=modes)],
    )
    return meta, field


def withheld(meta):
    meta["masks"] = []
    meta["rejected"] = [
        dict(
            track_id=7,
            reason="no_current_policy_return_ownership",
            policy_return_count=0,
        )
    ]


def test_original_and_calibrated_aggregations_are_both_read_without_recalculation():
    meta, field = fixture()
    for schema in ("noisy_or", "restaurant_evidence_weighted_noisy_or"):
        field["objects"][0]["cra_control"]["cra"]["aggregation"] = schema
        before = deepcopy((meta, field))
        assert (
            verify_camera_metadata(meta, field, current_ownership_only=True)[1] == set()
        )
        assert (meta, field) == before


def test_zero_current_owner_mask_is_withheld_with_exact_recorded_reason():
    meta, field = fixture(count=0)
    withheld(meta)
    assert verify_camera_metadata(meta, field, current_ownership_only=True)[1] == {7}


def test_legacy_camera_contract_still_allows_zero_owner_mask_and_forbids_rejections():
    meta, field = fixture(count=0)
    verify_camera_metadata(meta, field, current_ownership_only=False)
    withheld(meta)
    with pytest.raises(ValueError):
        verify_camera_metadata(meta, field, current_ownership_only=False)


@pytest.mark.parametrize(
    "fault",
    [
        "shown_zero",
        "missing_owned",
        "missing_rejection",
        "wrong_reason",
        "wrong_count",
        "unknown_id",
        "duplicate",
        "wrong_phi",
        "wrong_applied",
        "wrong_mode",
        "wrong_clock",
    ],
)
def test_inconsistent_camera_records_remain_failures(fault):
    meta, field = fixture()
    if fault in (
        "shown_zero",
        "missing_rejection",
        "wrong_reason",
        "wrong_count",
        "unknown_id",
        "duplicate",
    ):
        field["objects"][0]["diagnostics"]["policy_return_count"] = 0
        if fault != "shown_zero":
            withheld(meta)
        if fault == "missing_rejection":
            meta["rejected"] = []
        elif fault == "wrong_reason":
            meta["rejected"][0]["reason"] = "arbitrary_failure"
        elif fault == "wrong_count":
            meta["rejected"][0]["policy_return_count"] = 1
        elif fault == "unknown_id":
            meta["rejected"][0]["track_id"] = 99
        elif fault == "duplicate":
            meta["rejected"] *= 2
    elif fault == "missing_owned":
        meta["masks"] = []
    elif fault == "wrong_phi":
        meta["masks"][0]["phi"] += 0.01
    elif fault == "wrong_applied":
        meta["masks"][0]["applied_phi"] += 0.01
    elif fault == "wrong_mode":
        meta["masks"][0]["factor_modes"]["kinetic"] = "configured_prior"
    else:
        meta["camera_timestamp_seconds"] += 0.01
    with pytest.raises(ValueError):
        verify_camera_metadata(meta, field, current_ownership_only=True)


def test_no_mask_frame_requires_exact_original_sensor_and_camera_tuple():
    meta, _ = fixture()
    source = dict(lidar_id=10, lidar_time=1.0, camera_id=20, camera_time=1.1)
    verify_snapshot_clock(source, meta)
    for key in source:
        changed = dict(source)
        changed[key] += 1
        with pytest.raises(ValueError, match="source tuple"):
            verify_snapshot_clock(changed, meta)


@pytest.mark.parametrize("with_masks", [False, True])
def test_each_camera_assessment_is_bound_to_original_snapshot_identity(with_masks):
    meta, field = fixture(count=4 if with_masks else 0)
    if not with_masks:
        withheld(meta)
    source = dict(lidar_id=10, lidar_time=1.0, camera_id=20, camera_time=1.1)
    verify_camera_source(meta, field, source, current_ownership_only=True)
    # Changing a source ID consistently in the presentation metadata cannot
    # relabel the independently recorded snapshot, even with identical time.
    for key in ("sensor_frame_id", "camera_frame_id"):
        changed = deepcopy(meta)
        changed[key] += 1
        with pytest.raises(ValueError, match="source tuple"):
            verify_camera_source(changed, field, source, current_ownership_only=True)

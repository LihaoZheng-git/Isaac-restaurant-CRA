from dataclasses import replace

import numpy as np
import pytest

from robot.perception.human_body_envelope import (
    HumanBodyEnvelope,
    HumanBodyEnvelopeConfig,
    envelope_grid_points,
)
from robot.safety.obstacle_tracking import TrackedObstacle, ObstacleTrackingResult
from robot.safety.local_occupancy_grid import LocalOccupancyGridConfig


def tracking(time=1.0, center=(1.0, 0.0), count=1, *, matched=False, other=False):
    track = TrackedObstacle(
        7, center, center, (0.1, 0.0), (0.1, 0.0), 20, count, matched
    )
    tracks = (track,)
    ids = np.full(20, 7)
    if other:
        tracks += (
            replace(track, track_id=8, centroid_world_xy=(center[0] + 0.1, center[1])),
        )
        ids = np.r_[ids, np.full(20, 8)]
    return ObstacleTrackingResult(
        time,
        np.zeros((len(ids), 2)),
        ids,
        tracks,
        len(tracks),
        int(matched),
        int(not matched),
    )


def cue(time=1.0):
    return {
        7: dict(camera_timestamp_seconds=time, camera_frame_id=round(time * 10) + 1)
    }


def test_fresh_identity_then_current_lidar_retention_keeps_original_clock():
    m = HumanBodyEnvelope(HumanBodyEnvelopeConfig())
    a = m.update(tracking(), cue(), camera_timestamp_seconds=1.0)[0]
    b = m.update(
        tracking(1.1, (1.01, 0.0), 2, matched=True), {}, camera_timestamp_seconds=1.1
    )[0]
    assert a["status"] == b["status"] == "prior"
    assert b["radius_m"] == pytest.approx(0.6)
    assert b["original_person_camera_timestamp"] == 1.0
    assert b["original_person_camera_frame_id"] == 11
    assert b["lidar_timestamp_seconds"] == 1.1
    assert b["identity_age_seconds"] == pytest.approx(0.1)
    assert b["center_world_xy"] == pytest.approx((1.01, 0.0))


@pytest.mark.parametrize(
    "change",
    ["missing", "unmatched", "jump", "competitor", "old_observation", "gap", "expired"],
)
def test_unresolved_identity_blocks_field_instead_of_erasing_body(change):
    m = HumanBodyEnvelope(HumanBodyEnvelopeConfig(maximum_identity_age_seconds=0.15))
    m.update(tracking(), cue(), camera_timestamp_seconds=1.0)
    t = tracking(1.1, (1.01, 0.0), 2, matched=True)
    if change == "missing":
        t = replace(t, tracked_obstacles=())
    if change == "unmatched":
        t = tracking(1.1, count=2)
    if change == "jump":
        t = tracking(1.1, center=(2.0, 0.0), count=2, matched=True)
    if change == "competitor":
        t = tracking(1.1, count=2, matched=True, other=True)
    if change == "old_observation":
        t = tracking(1.1, count=1, matched=True)
    if change == "gap":
        t = tracking(1.4, count=2, matched=True)
    if change == "expired":
        t = tracking(1.2, count=2, matched=True)
    with pytest.raises(ValueError):
        m.update(t, {}, camera_timestamp_seconds=t.timestamp_seconds)
    assert 7 in m.identities
    assert m.identities[7]["camera_timestamp"] == 1.0


def test_fresh_reassessment_may_renew_visual_identity_not_by_borrowing():
    m = HumanBodyEnvelope(HumanBodyEnvelopeConfig())
    m.update(tracking(), cue(), camera_timestamp_seconds=1.0)
    b = m.update(
        tracking(2.0, count=2, matched=True), cue(2.0), camera_timestamp_seconds=2.0
    )[0]
    assert b["original_person_camera_timestamp"] == 2.0
    assert b["identity_age_seconds"] == 0


def test_no_person_cue_does_not_invent_a_body_and_bad_epochs_are_rejected():
    m = HumanBodyEnvelope(HumanBodyEnvelopeConfig())
    assert m.update(tracking(), {}, camera_timestamp_seconds=1.0) == []
    with pytest.raises(ValueError):
        m.update(tracking(), cue(), camera_timestamp_seconds=0.9)
    with pytest.raises(ValueError):
        m.update(tracking(), cue(0.9), camera_timestamp_seconds=1.0)


def test_duplicate_camera_frame_does_not_renew_identity_time():
    m = HumanBodyEnvelope(HumanBodyEnvelopeConfig())
    m.update(tracking(), cue(), camera_timestamp_seconds=1.0)
    duplicate = cue(1.1)
    duplicate[7]["camera_frame_id"] = 11
    row = m.update(
        tracking(1.1, count=2, matched=True), duplicate, camera_timestamp_seconds=1.1
    )[0]
    assert row["original_person_camera_timestamp"] == 1.0
    assert row["identity_age_seconds"] == pytest.approx(0.1)


def test_filled_prior_raster_contains_center_and_transforms_with_frame():
    m = HumanBodyEnvelope(HumanBodyEnvelopeConfig())
    rows = m.update(tracking(), cue(), camera_timestamp_seconds=1.0)
    grid = LocalOccupancyGridConfig(
        minimum_x=-2, maximum_x=2, minimum_y=-2, maximum_y=2, resolution=0.1
    )
    xy, ids = envelope_grid_points(
        rows, grid_config=grid, anchor_xy=np.array([1.0, 0.0]), anchor_yaw=np.pi / 2
    )
    assert np.min(np.linalg.norm(xy, axis=1)) < 0.08
    assert set(ids) == {3_000_000_007}
    assert np.max(np.linalg.norm(xy, axis=1)) <= 0.6 + 0.1 / np.sqrt(2)
    np.testing.assert_allclose(xy.min(axis=0), -xy.max(axis=0), atol=1e-12)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"body_radius_m": 0},
        {"minimum_points": True},
        {"maximum_lidar_gap_seconds": np.nan},
        {"minimum_competitor_point_fraction": 2},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        HumanBodyEnvelopeConfig(**kwargs)


def test_actual_field_adds_prior_cells_without_changing_observed_factor_scores(
    monkeypatch,
):
    from test_object_risk_map import mapped_worker
    from test_live_object_perception import packet
    import robot.evaluation.multi_object_perception as multi

    runtime = mapped_worker()
    runtime.human_body_envelope = HumanBodyEnvelope(
        HumanBodyEnvelopeConfig(minimum_points=3)
    )
    monkeypatch.setattr(
        multi,
        "fresh_person_selection_evidence",
        lambda objects, t, frame: {
            objects[0].attributes.track_id: dict(
                camera_timestamp_seconds=t, camera_frame_id=frame
            )
        },
    )
    result = runtime(packet())
    first = result.diagnostics["human_body_envelope_priors"][0]
    assert first["status"] == "prior" and first["current_support_count"] == 4
    assert result.field.all_obstacle_point_count == 4
    assert len(np.flatnonzero(result.source_occupancy_grid.occupied_mask)) > 1
    assert not result.readiness.ready
    assert all(row.policy.risk == pytest.approx(1 - 0.75**4) for row in result.objects)
    original_source = first["original_person_camera_timestamp"]
    monkeypatch.setattr(multi, "fresh_person_selection_evidence", lambda *args: {})
    next_result = runtime(packet(2, 1.1))
    assert (
        next_result.diagnostics["human_body_envelope_priors"][0][
            "original_person_camera_timestamp"
        ]
        == original_source
    )
    previous = runtime.previous
    with pytest.raises(ValueError, match="continuity unresolved"):
        runtime(packet(3, 1.5))
    assert runtime.previous is previous


def test_independent_envelope_validation_rejects_undercoverage_and_unaligned_truth():
    from robot.evaluation.human_body_envelope_validation import (
        validate_human_body_envelopes,
    )

    def records(radius):
        return [
            dict(
                sensor_timestamp=t,
                diagnostics=dict(
                    human_body_envelope_priors=[
                        dict(
                            track_id=7,
                            center_world_xy=(0.1, 0.0),
                            radius_m=radius,
                            status="prior",
                            lidar_timestamp_seconds=t,
                        )
                    ]
                ),
            )
            for t in (1.0, 1.1, 1.2)
        ]

    truth = [dict(time=t, actor_xyz=(0.0, 0.0, 0.0)) for t in (1.0, 1.1, 1.2)]
    assert validate_human_body_envelopes(records(0.6), truth, reference_radius_m=0.4)[
        "passed"
    ]
    assert not validate_human_body_envelopes(
        records(0.45), truth, reference_radius_m=0.4
    )["passed"]
    assert not validate_human_body_envelopes(records(0.6), [], reference_radius_m=0.4)[
        "passed"
    ]

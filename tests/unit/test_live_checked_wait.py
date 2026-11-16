from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from robot.evaluation.live_checked_wait import StagedLivePerception, packet_state_in_field
from robot.safety.safe_sage_guidance_policy import PassingSide
from test_live_object_perception import packet
from test_object_risk_map import mapped_worker


class Passing:
    config = SimpleNamespace(require_visual_person_selection=False,
                             maximum_identity_prior_selection_age_seconds=0.)
    encounter = (0, "right", 1.)

    def update(self, tracking, **kwargs):
        key = tracking.tracked_obstacles[0].track_id
        return {key: PassingSide.RIGHT}, dict(source_timestamp=tracking.timestamp_seconds,
            preferences={key: "right"}, encounter_selected_timestamp=1., decision="retained")


class Encounter:
    def update(self, tracking, **kwargs):
        key = tracking.tracked_obstacles[0].track_id
        return dict(source_timestamp_seconds=tracking.timestamp_seconds,
                    tracks={key: dict(track_id=key, source_timestamp_seconds=tracking.timestamp_seconds,
                                      classification="crossing")})


def configured():
    worker = mapped_worker()
    worker.motion_passing, worker.encounter_motion = Passing(), Encounter()
    return worker


def test_source_pose_transform_uses_rotated_translated_anchor():
    field = SimpleNamespace(anchor_yaw=np.pi/2, anchor_position_world_xy=np.array([10., 20.]))
    source = replace(packet(), robot_position_world_xy=(10., 22.), robot_yaw=np.pi)
    np.testing.assert_allclose(packet_state_in_field(source, field), [2., 0., np.pi/2], atol=1e-12)


def test_staging_rejected_freshness_or_pose_does_not_advance_actual_worker():
    worker = configured()
    worker(packet())
    prior = worker.previous
    tracker, memory = worker.tracker, worker.persistent_properties
    staged = StagedLivePerception(worker)
    result = staged(packet(2, 1.1))
    assert worker.previous is prior and worker.tracker is tracker
    assert worker.persistent_properties is memory
    with pytest.raises(RuntimeError, match="consumed"):
        staged(packet(3, 1.2))
    staged.discard()
    result = staged(packet(3, 1.2))
    with pytest.raises(ValueError, match="original proposal"):
        staged.accept(replace(result))
    staged.accept(result)
    assert worker.previous is result.field
    assert worker.previous.sensor_timestamp_seconds == 1.2
    assert worker.tracker is not tracker and worker.persistent_properties is not memory


def test_gap_retains_only_existing_unpublished_bootstrap():
    worker = configured()
    worker(packet())
    prior, tracker = worker.previous, worker.tracker
    staged = StagedLivePerception(worker)
    with pytest.raises(ValueError, match="unpublished recovery baseline"):
        staged(packet(2, 3.))
    assert worker.previous is prior and worker.tracker is tracker and staged.pending is None
    assert worker.temporal_recovery_baseline[1] == 3.
    result = staged(packet(3, 3.1))
    assert result.diagnostics["temporal_gap_recovery"]["baseline_timestamp"] == 3.
    staged.accept(result)
    assert worker.temporal_recovery_baseline is None


@pytest.mark.parametrize("accepted", [True, False])
def test_real_field_rebuild_uses_same_accepted_temporal_pair_and_preserves_geometry(monkeypatch, accepted):
    import robot.evaluation.live_checked_wait as live

    worker = configured()
    first = worker(packet())
    previous = worker.previous
    calls = []
    original = worker.temporal.estimate

    def record(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(worker.temporal, "estimate", record)
    monkeypatch.setattr(live, "check_wait", lambda control, state, field: dict(accepted=accepted))
    worker.checked_wait_control = object()
    result = worker(packet(2, 1.1))
    audit = result.diagnostics["checked_wait_transition"]
    assert len(calls) == (1 if accepted else 2)
    for call in calls:
        assert call["previous_h_full"] is previous.pipeline.h_full
        assert call["time_delta_seconds"] == pytest.approx(.1)
        assert call["previous_estimate"] is None
    assert result.field.sensor_timestamp_seconds == 1.1
    assert audit["previous_accepted_source_timestamp_seconds"] == 1.
    np.testing.assert_array_equal(first.source_occupancy_grid.occupied_mask,
                                  result.source_occupancy_grid.occupied_mask)
    assert result.objects[0].policy.risk == first.objects[0].policy.risk
    assert bool(result.field.crossing_yield_track_ids) == accepted
    assert result.diagnostics["crossing_yield_track_ids"] == result.field.crossing_yield_track_ids
    assert all(row["lambda"] in ({0., 1.25} if accepted else {1.25})
               for row in result.diagnostics["applied_field_policies"])
    assert worker.previous is result.field


def test_alternative_failure_rolls_back_entire_staged_transaction(monkeypatch):
    import robot.evaluation.live_checked_wait as live
    import robot.evaluation.multi_object_perception as multi

    worker = configured()
    worker(packet())
    before = worker.previous
    worker.checked_wait_control = object()
    monkeypatch.setattr(live, "check_wait", lambda *a: dict(accepted=False))
    original = multi.build_safe_sage_field_pipeline
    calls = []

    def fail_second(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise ValueError("alternative solve failed")
        return original(*args, **kwargs)

    monkeypatch.setattr(multi, "build_safe_sage_field_pipeline", fail_second)
    staged = StagedLivePerception(worker)
    with pytest.raises(ValueError, match="alternative solve failed"):
        staged(packet(2, 1.1))
    assert worker.previous is before and staged.pending is None

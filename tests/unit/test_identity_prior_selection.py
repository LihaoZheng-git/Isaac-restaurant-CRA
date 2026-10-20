"""Bounded identity selection is separate from physical evidence and fresh cues."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from test_human_body_envelope import tracking, cue
from test_motion_aware_passing import sample
from robot.perception.human_body_envelope import (
    HumanBodyEnvelope, HumanBodyEnvelopeConfig, HumanIdentitySelectionPrior,
)
from robot.safety.motion_aware_passing import MotionAwarePassing, MotionAwarePassingConfig


def config(**kwargs):
    return MotionAwarePassingConfig(require_visual_person_selection=True,
        maximum_identity_prior_selection_age_seconds=.5, **kwargs)


def step(policy, t, prior=None, fresh=()):
    current, _ = sample(t)
    return policy.update(current, robot_xy=(0, 0), robot_velocity_xy=(0, .2), goal_xy=(0, 4),
        visual_person_track_ids=fresh, identity_prior_selection_evidence=prior)


def evidence(t=.3, original=.1):
    return HumanIdentitySelectionPrior(7, original, 11, t + .05, t, 20)


def test_prior_can_select_without_faking_freshness_and_original_clocks_never_renew():
    p = MotionAwarePassing(config())
    for t in (.1, .2):
        assert not step(p, t, {})[0]
    sides, selected = step(p, .3, {7: evidence()})
    assert sides and selected["current_visual_person_track_ids"] == []
    assert selected["original_selection_visual_person_verified"] is False
    original = selected["original_selection_identity_prior"]
    assert original["original_person_camera_timestamp"] == .1
    assert original["status"] == "prior"
    for t in np.arange(.4, 8.3, .1):
        current, _ = sample(.4, speed=0.)
        current.timestamp_seconds = float(t)
        sides, held = p.update(current, robot_xy=(0, 0), robot_velocity_xy=(0, .2),
            goal_xy=(0, 4), visual_person_track_ids=(), identity_prior_selection_evidence={})
        assert held["original_selection_identity_prior"] == original
        assert held["encounter_selected_timestamp"] == .3
    sides, released = step(p, 8.4, {})
    assert not sides and released["decision"] == "released_missing_or_expired_target"


def test_fresh_cue_wins_and_ablation_still_requires_explicit_evidence():
    p = MotionAwarePassing(config())
    for t in (.1, .2):
        step(p, t, {})
    _, selected = step(p, .3, {7: evidence()}, fresh=(7,))
    assert selected["original_selection_visual_person_verified"] is True
    assert selected["original_selection_identity_prior"] is None
    with pytest.raises(ValueError, match="explicit current envelope"):
        step(MotionAwarePassing(config()), .1)
    strict = MotionAwarePassing(MotionAwarePassingConfig(require_visual_person_selection=True))
    with pytest.raises(ValueError, match="explicit ablation"):
        step(strict, .3, {7: evidence()})
    assert strict.last_timestamp is None


@pytest.mark.parametrize("bad", ["dict", "other_track", "stale_scan", "older_than_config"])
def test_invalid_prior_does_not_advance_selector(bad):
    p = MotionAwarePassing(config())
    value = evidence()
    if bad == "dict":
        value = value.diagnostic()
    elif bad == "other_track":
        value = replace(value, track_id=8)
    elif bad == "stale_scan":
        value = replace(value, lidar_timestamp_seconds=.2)
    elif bad == "older_than_config":
        p = MotionAwarePassing(replace(config(), maximum_identity_prior_selection_age_seconds=.1))
    with pytest.raises(ValueError):
        step(p, .3, {7: value})
    assert p.last_timestamp is None and p.history == {}


@pytest.mark.parametrize("change", [
    {"original_person_camera_timestamp": .4},
    {"original_person_camera_timestamp": -.1},
    {"selection_camera_timestamp": .7},
    {"selection_camera_timestamp": float("nan")},
    {"lidar_timestamp_seconds": .0},
    {"current_support_count": 2},
    {"original_person_camera_frame_id": True},
])
def test_future_expired_or_malformed_identity_evidence_rejected(change):
    with pytest.raises(ValueError):
        replace(evidence(), **change)


@pytest.mark.parametrize("age", [-.1, .5001, float("nan"), True])
def test_config_cannot_exceed_approved_bound(age):
    with pytest.raises(ValueError):
        replace(config(), maximum_identity_prior_selection_age_seconds=age)


def test_config_needs_visual_selection_and_worker_needs_body_adapter():
    from test_live_object_perception import worker
    from robot.perception.human_mass_prior import HumanMassPriorConfig
    from robot.perception.persistent_object_evidence import PersistentObjectEvidenceConfig
    with pytest.raises(ValueError):
        replace(config(), require_visual_person_selection=False)
    with pytest.raises(ValueError, match="body-envelope adapter"):
        worker(person_detector=object(), human_mass_prior_config=HumanMassPriorConfig("test"),
            persistent_property_config=PersistentObjectEvidenceConfig("test"),
            motion_passing_config=config())


def test_only_successful_current_envelope_produces_short_identity_prior():
    m = HumanBodyEnvelope(HumanBodyEnvelopeConfig())
    m.update(tracking(), cue(), camera_timestamp_seconds=1.)
    for i in range(1, 7):
        t = 1. + i * .1
        current = tracking(t, (1.+i*.01, 0.), i+1, matched=True)
        m.update(current, {}, camera_timestamp_seconds=t)
        before = deepcopy(m.identities)
        priors = m.passing_selection_priors(current, camera_timestamp_seconds=t, maximum_age_seconds=.5)
        assert bool(priors) == (i <= 5)
        if priors:
            assert priors[7].original_person_camera_timestamp == 1.
            assert priors[7].lidar_timestamp_seconds == t
        np.testing.assert_array_equal(m.identities[7]["center"], before[7]["center"])
        assert m.identities[7]["camera_timestamp"] == 1.
    with pytest.raises(ValueError, match="current envelope"):
        m.passing_selection_priors(tracking(1.7, count=8, matched=True),
            camera_timestamp_seconds=1.7, maximum_age_seconds=.5)


def test_actual_field_uses_identity_prior_without_renewing_factors(monkeypatch):
    from test_live_object_perception import packet
    from test_object_risk_map import mapped_worker
    import robot.evaluation.multi_object_perception as multi

    runtime, reference = mapped_worker(), mapped_worker()
    for w in (runtime, reference):
        w.human_body_envelope = HumanBodyEnvelope(HumanBodyEnvelopeConfig(minimum_points=3))
    runtime.motion_passing = MotionAwarePassing(config(minimum_point_count=3))
    reference.motion_passing = MotionAwarePassing(MotionAwarePassingConfig(
        require_visual_person_selection=True, minimum_point_count=3))
    monkeypatch.setattr(multi, "fresh_person_selection_evidence", lambda objects, t, frame: {
        objects[0].attributes.track_id: dict(camera_timestamp_seconds=t, camera_frame_id=frame)
    } if frame == 1 else {})
    for i in range(4):
        source = packet(i+1, 1.+i*.1)
        points = source.obstacle_xyz + [0, .6 - .05*i, 0]
        source = replace(source, obstacle_xyz=points, scene_xyz=points)
        a, b = runtime(source), reference(source)
    audit = a.diagnostics["motion_aware_passing"]
    assert audit["preferences"] and audit["original_selection_identity_prior"]["status"] == "prior"
    assert not b.diagnostics["motion_aware_passing"]["preferences"]
    assert audit["current_visual_person_track_ids"] == []
    assert a.field.sensor_timestamp_seconds == b.field.sensor_timestamp_seconds
    assert a.field.all_obstacle_point_count == b.field.all_obstacle_point_count == 4
    assert [o.attributes for o in a.objects] == [o.attributes for o in b.objects]
    assert [o.policy.risk for o in a.objects] == [o.policy.risk for o in b.objects]


def test_primary_comparison_rejects_identity_selection_variant():
    from robot.evaluation.restaurant_comparison import comparison_summary
    with pytest.raises(ValueError, match="separate comparison"):
        comparison_summary([{"configuration": {"motion_aware_passing_config": {
            "maximum_identity_prior_selection_age_seconds": .5}}}])


@pytest.mark.parametrize("failure", [None, "renewed", "fresh_claim", "missing_envelope", "expired", "missing_prior"])
def test_independent_validation_checks_original_identity_and_current_support(failure):
    from test_motion_passing_validation import example
    from robot.evaluation.motion_passing_validation import summarize_motion_passing
    from robot.perception.human_body_envelope import HUMAN_ENVELOPE_SOURCE
    records, truth = example()
    original = HumanIdentitySelectionPrior(7, .8, 42, 1.05, 1., 20).diagnostic()
    for row in records:
        audit = row["motion_aware_passing"]
        audit.update(configuration={"maximum_identity_prior_selection_age_seconds": .5},
            original_selection_visual_person_verified=False,
            original_selection_identity_prior=deepcopy(original))
        row["human_body_envelope_priors"] = [dict(track_id=7, source=HUMAN_ENVELOPE_SOURCE,
            status="prior", current_support_count=20, lidar_timestamp_seconds=row["time"])]
    if failure == "renewed":
        records[-1]["motion_aware_passing"]["original_selection_identity_prior"]["original_person_camera_timestamp"] = .9
    elif failure == "fresh_claim":
        records[0]["motion_aware_passing"]["original_selection_visual_person_verified"] = True
    elif failure == "missing_envelope":
        records[0]["human_body_envelope_priors"] = []
    elif failure == "expired":
        records[0]["motion_aware_passing"]["original_selection_identity_prior"]["original_person_camera_timestamp"] = .4
    elif failure == "missing_prior":
        records[0]["motion_aware_passing"]["original_selection_identity_prior"] = None
    result = summarize_motion_passing(records, truth, moving_robot=True, nominal_goal_world_xy=(0, 4))
    assert result["passed"] == (failure is None)
    assert result["checks"]["identity_prior_selection_provenance_bounded"] == (failure is None)

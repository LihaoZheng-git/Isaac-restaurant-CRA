import math

import pytest

from robot.perception.physical_attributes import EstimateStatus, ScalarEstimate
from robot.perception.temporal_object_evidence import (
    TEMPORAL_EVIDENCE_MARKER,
    TemporalObjectEvidence,
    TemporalObjectEvidenceConfig,
)
from robot.safety.obstacle_tracking import TrackedObstacle


def track(count=10, identity=7, xy=(1.0, 0.0), points=100, velocity=(0.0, 0.0)):
    return TrackedObstacle(
        identity, xy, xy, velocity, velocity, points, count, count > 1
    )


def estimate(value, unit, t=1.1, source="model", status=EstimateStatus.INFERRED):
    return ScalarEstimate(value, value * 0.8, value * 1.2, 0.7, unit, source, t, status)


def resolve(
    memory, item=None, *, t=1.1, mass=None, contact=None, others=(), ego=(0, 0)
):
    item = item or track()
    mass = mass or ScalarEstimate.unavailable(
        unit="kg", source="missing_instance", timestamp_seconds=t
    )
    return memory.resolve(
        track=item,
        tracked_obstacles=(item, *others),
        sensor_timestamp_seconds=t - 0.1,
        timestamp_seconds=t,
        camera_frame_id=round(t * 100),
        ego_position_world_xy=ego,
        fresh_mass=mass,
        fresh_contact=contact,
    )


def seeded(config=None, *, contact=0.2, velocity=(0, 0)):
    memory = TemporalObjectEvidence(config)
    resolve(
        memory,
        track(velocity=velocity),
        mass=estimate(4, "kg"),
        contact=estimate(contact, "1", source="visual_silhouette_contact_proxy"),
    )
    return memory


def test_short_gap_keeps_origin_and_degrades_uncertainty_without_refreshing_evidence():
    memory = seeded()
    mass, contact, diagnostics = resolve(memory, track(11), t=1.4)
    assert mass.value == 4 and contact.value == 0.2
    assert mass.timestamp_seconds == contact.timestamp_seconds == 1.1
    assert mass.status is contact.status is EstimateStatus.PRIOR
    assert mass.confidence == pytest.approx(0.7 * 2 ** (-0.3))
    assert mass.lower_bound < 3.2 and mass.upper_bound > 4.8
    assert contact.lower_bound < 0.16 and contact.upper_bound > 0.24
    assert TEMPORAL_EVIDENCE_MARKER in mass.evidence
    assert diagnostics["mass"]["source_camera_frame_id"] == 110
    mass2, _, _ = resolve(memory, track(12), t=1.5)
    assert mass2.timestamp_seconds == 1.1
    assert mass2.confidence < mass.confidence


def test_independent_property_expiry_and_repeated_borrowing_cannot_renew_source():
    memory = seeded()
    last_mass = None
    for count in range(11, 43):
        t = 1.1 + (count - 10) * 0.1
        mass, contact, _ = resolve(memory, track(count), t=t)
        if t > 2.1 + 1e-9:
            assert contact is None
        if t > 4.1 + 1e-9:
            assert not mass.is_available
        else:
            last_mass = mass
    assert last_mass.timestamp_seconds == 1.1
    assert not memory.has_unexpired(7, 4.3)


@pytest.mark.parametrize(
    "changes",
    [
        {"identity": 8},
        {"count": 1},
        {"count": 12},
        {"xy": (1.5, 0)},
        {"points": 20},
    ],
)
def test_id_reset_discontinuity_centroid_jump_or_cluster_split_invalidates_memory(
    changes,
):
    memory = seeded()
    item = track(11)
    arguments = dict(count=11, identity=7, xy=(1, 0), points=100)
    arguments.update(changes)
    item = track(**arguments)
    mass, contact, _ = resolve(memory, item, t=1.3)
    assert not mass.is_available and contact is None


def test_close_competitor_blocks_borrowing_and_later_clear_view_does_not_resurrect_it():
    memory = seeded()
    mass, contact, _ = resolve(
        memory, track(11), t=1.3, others=(track(identity=8, xy=(1.1, 0)),)
    )
    assert not mass.is_available and contact is None
    mass, contact, _ = resolve(memory, track(12), t=1.4)
    assert not mass.is_available and contact is None


def test_disappeared_track_does_not_reappear_with_borrowed_properties():
    memory = seeded()
    resolve(memory, track(identity=8, xy=(2, 0)), t=1.2)
    mass, contact, _ = resolve(memory, track(11), t=1.3)
    assert not mass.is_available and contact is None


def test_contact_viewpoint_gate_does_not_discard_valid_mass():
    memory = seeded()
    mass, contact, diagnostics = resolve(memory, track(11), t=1.3, ego=(0, 1))
    assert mass.is_available and contact is None
    assert "viewpoint" in diagnostics["contact"]["rejection_reason"]


def test_moving_identity_uses_observed_source_velocity_without_ego_motion_substitution():
    memory = seeded(velocity=(0.2, 0))
    mass, contact, _ = resolve(memory, track(11, xy=(1.1, 0), velocity=(0.2, 0)), t=1.6)
    assert mass.is_available and contact.is_available


def test_stronger_contact_cue_survives_a_weaker_current_silhouette_without_renewal():
    memory = seeded(contact=0.7)
    _, held, detail = resolve(
        memory,
        track(11),
        t=1.3,
        contact=estimate(0.1, "1", t=1.3, source="visual_silhouette_contact_proxy"),
    )
    assert held.value == 0.7 and held.timestamp_seconds == 1.1
    assert detail["contact"]["mode"] == "retained_stronger_contact"
    _, fresh, _ = resolve(
        memory,
        track(12),
        t=2.2,
        contact=estimate(0.1, "1", t=2.2, source="visual_silhouette_contact_proxy"),
    )
    assert fresh.value == 0.1 and fresh.timestamp_seconds == 2.2


@pytest.mark.parametrize(
    "status", [EstimateStatus.ORACLE, EstimateStatus.FALLBACK, EstimateStatus.PRIOR]
)
def test_supplied_prior_oracle_or_fallback_is_not_promoted_to_new_observation(status):
    memory = TemporalObjectEvidence()
    resolve(memory, mass=estimate(4, "kg", status=status))
    mass, _, _ = resolve(memory, track(11), t=1.3)
    assert not mass.is_available


def test_retention_config_rejects_invalid_limits():
    for change in (
        {"maximum_mass_age_seconds": 0},
        {"maximum_contact_age_seconds": float("nan")},
        {"minimum_point_count_ratio": 2},
        {"maximum_contact_view_change_radians": 2 * math.pi},
    ):
        with pytest.raises(ValueError):
            TemporalObjectEvidenceConfig(**change)

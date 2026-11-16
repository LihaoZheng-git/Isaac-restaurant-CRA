from copy import deepcopy
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest
from test_human_body_envelope import cue, tracking
from test_physical_factor_adapter import attributes

from robot.evaluation.human_mass_runtime import apply_body_identity_mass_retention
from robot.evaluation.multi_object_perception import ObjectAssessment
from robot.perception.human_body_envelope import HumanBodyEnvelope, HumanBodyEnvelopeConfig
from robot.perception.human_mass_identity_retention import (
    HUMAN_MASS_RETENTION_MARKER, HumanMassIdentityRetention, HumanMassIdentityRetentionConfig,
)
from robot.perception.human_mass_prior import HUMAN_MASS_PRIOR_SOURCE
from robot.perception.physical_attributes import EstimateStatus, ScalarEstimate
from robot.risk.object_risk_policy import ObjectRiskPriorConfig, evaluate_object_risk
from robot.risk.physical_factor_adapter import PhysicalFactorAdapterConfig, adapt_physical_factors


def mass(t=1.):
    return ScalarEstimate(70., 45., 100., .5, 'kg', HUMAN_MASS_PRIOR_SOURCE, t, EstimateStatus.PRIOR)


def start():
    body = HumanBodyEnvelope(HumanBodyEnvelopeConfig())
    tracks = tracking()
    body.update(tracks, cue(), camera_timestamp_seconds=1.)
    memory = HumanMassIdentityRetention()
    retained, audit = memory.update(tracks, body, {7: mass()}, timestamp_seconds=1., camera_frame_id=11)
    assert not retained and audit['rows'][0]['mode'] == 'fresh'
    return memory, body


def advance(memory, body, t, count):
    tracks = tracking(t, (1.+.1*(t-1), 0.), count, matched=True)
    body.update(tracks, {}, camera_timestamp_seconds=t)
    return memory.update(tracks, body, {}, timestamp_seconds=t, camera_frame_id=round(t*10)+1)


def test_current_identity_retention_keeps_original_expiry_and_widens_uncertainty():
    memory, body = start()
    for i, t in enumerate(np.arange(1.2, 4.01, .2), 2):
        values, audit = advance(memory, body, float(t), i)
        m = values[7]
        assert m.timestamp_seconds == 1. and m.source == HUMAN_MASS_PRIOR_SOURCE
        assert m.status is EstimateStatus.PRIOR and HUMAN_MASS_RETENTION_MARKER in m.evidence
        assert m.lower_bound < 45. and m.upper_bound > 100. and 0 < m.confidence < .5
        assert audit['rows'][0]['source_frame_id'] == 11
        assert audit['rows'][0]['source_timestamp'] == 1.
    assert not advance(memory, body, 4.1, 18)[0]
    assert not memory.sources  # repeated borrowing never renewed the source


def test_body_identity_alone_never_creates_mass_or_transfers_a_source():
    memory, body = start()
    other = HumanMassIdentityRetention()
    assert not other.update(tracking(), body, {}, timestamp_seconds=1., camera_frame_id=11)[0]
    current = tracking(1.1, (1.01, 0.), 2, matched=True)
    body.update(current, {}, camera_timestamp_seconds=1.1)
    with pytest.raises(ValueError, match='unverified body'):
        memory.update(current, body, {8: mass(1.1)},
            timestamp_seconds=1.1, camera_frame_id=12)
    assert memory.sources[7][0].timestamp_seconds == 1.


@pytest.mark.parametrize('change', ['wrong_source', 'future', 'borrowed', 'measured', 'wrong_frame'])
def test_only_an_original_same_frame_visual_adult_prior_can_seed(change):
    body = HumanBodyEnvelope(HumanBodyEnvelopeConfig())
    body.update(tracking(), cue(), camera_timestamp_seconds=1.)
    fresh = mass()
    frame = 11
    if change == 'wrong_source':
        fresh = replace(fresh, source='catalog_network')
    if change == 'future':
        fresh = replace(fresh, timestamp_seconds=1.1)
    if change == 'borrowed':
        fresh = replace(fresh, evidence=(HUMAN_MASS_RETENTION_MARKER,))
    if change == 'measured':
        fresh = replace(fresh, status=EstimateStatus.MEASURED)
    if change == 'wrong_frame':
        frame = 12
    memory = HumanMassIdentityRetention()
    with pytest.raises(ValueError, match='seed requires'):
        memory.update(tracking(), body, {7: fresh}, timestamp_seconds=1., camera_frame_id=frame)
    assert memory.last_time is None and not memory.sources


def test_stale_body_update_is_transactional_and_no_source_gap_is_filled():
    memory, body = start()
    before = deepcopy(memory.sources)
    with pytest.raises(ValueError, match='current positively supported'):
        memory.update(tracking(1.1, count=2, matched=True), body, {}, timestamp_seconds=1.1, camera_frame_id=12)
    assert memory.sources == before and memory.last_time == 1.
    with pytest.raises(ValueError, match='lidar_source_gap'):
        body.update(tracking(1.4, count=2, matched=True), {}, camera_timestamp_seconds=1.4)


@pytest.mark.parametrize('maximum', [3.01, float('nan'), True, 0.])
def test_retention_config_cannot_expand_three_second_policy(maximum):
    with pytest.raises(ValueError):
        HumanMassIdentityRetentionConfig(maximum_age_seconds=maximum)


def test_adapter_never_accepts_this_mass_beyond_three_seconds_even_with_long_session():
    memory, body = start()
    retained = advance(memory, body, 1.2, 2)[0][7]
    base = attributes()
    base = replace(base, mass=replace(base.mass, mass_kg=retained))
    settings = PhysicalFactorAdapterConfig(maximum_persistent_property_age_seconds=120.)
    assert not adapt_physical_factors(base, timestamp_seconds=2., config=settings).kinetic.rejection_reasons
    result = adapt_physical_factors(base, timestamp_seconds=4.01, config=settings)
    assert 'mass_kg: stale evidence' in result.kinetic.rejection_reasons


def test_runtime_rebuilds_mass_factors_but_preserves_current_contact_thermal_and_motion():
    memory, body = start()
    tracks = tracking(1.2, (1.02, 0.), 2, matched=True)
    body.update(tracks, {}, camera_timestamp_seconds=1.2)
    attrs = attributes()
    attrs = replace(attrs, mass=replace(attrs.mass, mass_kg=ScalarEstimate.unavailable(unit='kg',
        source='unavailable', timestamp_seconds=1.2)))
    # Give all current motion/route inputs the actual current epoch.
    attrs = replace(attrs, motion=replace(attrs.motion,
        relative_speed_mps=replace(attrs.motion.relative_speed_mps, timestamp_seconds=1.2),
        route_crossing_likelihood=replace(attrs.motion.route_crossing_likelihood, timestamp_seconds=1.2)))
    adapted = adapt_physical_factors(attrs, timestamp_seconds=1.2)
    prior = ObjectRiskPriorConfig()
    policy, details = evaluate_object_risk(adapted, prior)
    item = ObjectAssessment(attrs, adapted, policy, None,
        dict(property_evidence={'mass': {'mode': 'unavailable'}}, factor_scores=details))
    policies = {7: policy}
    output, audit = apply_body_identity_mass_retention(memory, body, tracks,
        SimpleNamespace(camera_timestamp_seconds=1.2, camera_frame_id=13), [item],
        [dict(accepted=False)], policies, prior)
    result = output[0]
    assert not result.adapted.kinetic.rejection_reasons
    assert result.adapted.contact is adapted.contact and result.adapted.thermal is adapted.thermal
    assert result.attributes.motion is attrs.motion
    assert result.attributes.mass.mass_kg.timestamp_seconds == 1.
    assert result.diagnostics['property_evidence']['mass']['mode'] == 'persistence_prior'
    assert audit['rows'][0]['current_lidar_timestamp'] == 1.2
    assert item.attributes.mass.mass_kg.value is None


def test_actual_fresh_specialist_result_can_replace_a_frozen_attribute_but_borrowing_cannot():
    memory, body = start()
    current = tracking(1.2, (1.02, 0.), 2, matched=True)
    body.update(current, cue(1.2), camera_timestamp_seconds=1.2)
    attrs = attributes()
    attrs = replace(attrs, mass=replace(attrs.mass, mass_kg=mass()))  # old generic-cache output
    adapted = adapt_physical_factors(attrs, timestamp_seconds=1.2)
    prior = ObjectRiskPriorConfig()
    policy, details = evaluate_object_risk(adapted, prior)
    raw = asdict(mass(1.2))
    item = ObjectAssessment(attrs, adapted, policy, None,
        dict(property_evidence={'mass': {'mode': 'persistence_prior'}}, factor_scores=details,
            geometry_evidence={'human_mass_prior': raw}))
    output, audit = apply_body_identity_mass_retention(memory, body, current,
        SimpleNamespace(camera_timestamp_seconds=1.2, camera_frame_id=13), [item],
        [dict(accepted=True, geometry_diagnostics={'human_mass_prior': raw})], {7: policy}, prior)
    assert output[0].attributes.mass.mass_kg.timestamp_seconds == 1.2
    assert memory.sources[7][0].timestamp_seconds == 1.2
    assert audit['rows'][0]['mode'] == 'fresh'
    assert item.attributes.mass.mass_kg.timestamp_seconds == 1.  # original untouched

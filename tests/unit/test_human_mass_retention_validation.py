from copy import deepcopy
from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import pytest
from test_human_body_envelope import cue, tracking
from test_human_mass_identity_retention import mass

from robot.evaluation.human_mass_retention_validation import validate_human_mass_identity_retention
from robot.evaluation.human_mass_runtime import apply_body_identity_mass_retention
from robot.evaluation.multi_object_perception import ObjectAssessment
from robot.perception.human_body_envelope import HumanBodyEnvelope, HumanBodyEnvelopeConfig
from robot.perception.human_mass_identity_retention import HumanMassIdentityRetention
from robot.perception.physical_attributes import EstimateStatus, PhysicalAttributes, ScalarEstimate
from robot.risk.object_risk_policy import ObjectRiskPriorConfig, evaluate_object_risk
from robot.risk.physical_factor_adapter import adapt_physical_factors


def records():
    body, memory, result = HumanBodyEnvelope(HumanBodyEnvelopeConfig()), HumanMassIdentityRetention(), []
    prior = ObjectRiskPriorConfig()
    for i in range(7):
        t = round(1.+i*.1, 8)
        tracks = tracking(t, (1.+.1*(t-1), 0.), i+1, matched=i > 0)
        envelopes = body.update(tracks, cue() if i == 0 else {}, camera_timestamp_seconds=t)
        attrs = PhysicalAttributes.unknown(track_id=7, timestamp_seconds=t)
        speed = ScalarEstimate(.5, .5, .5, .9, 'm/s', 'current_tracking', t, EstimateStatus.INFERRED)
        route = ScalarEstimate(1., 1., 1., .9, '1', 'current_route', t, EstimateStatus.INFERRED)
        attrs = replace(attrs, motion=replace(attrs.motion, relative_speed_mps=speed, route_crossing_likelihood=route))
        raw = asdict(mass()) if i == 0 else None
        if i == 0:
            attrs = replace(attrs, mass=replace(attrs.mass, mass_kg=mass()))
        geometry = {'human_mass_prior': raw} if raw is not None else {}
        adapted = adapt_physical_factors(attrs, timestamp_seconds=t)
        policy, factors = evaluate_object_risk(adapted, prior)
        obj = ObjectAssessment(attrs, adapted, policy, None, dict(accepted_instance=i == 0,
            geometry_evidence=geometry, property_evidence={'mass': {'mode': 'fresh' if i == 0 else 'unavailable'}}, factor_scores=factors))
        output, audit = apply_body_identity_mass_retention(memory, body, tracks,
            SimpleNamespace(camera_timestamp_seconds=t, camera_frame_id=round(t*10)+1), [obj],
            [dict(accepted=i == 0, geometry_diagnostics=geometry)], {7: policy}, prior)
        result.append(dict(time=t, camera_time=t, human_body_envelope_priors=envelopes,
            human_mass_body_identity_retention=audit,
            objects=[dict(attributes=asdict(output[0].attributes), diagnostics=output[0].diagnostics)]))
    return json.loads(json.dumps(result))


@pytest.mark.parametrize('live', [False, True])
def test_audit_checks_actual_original_seed_and_current_body_on_parked_and_live_layouts(live):
    data = records()
    if live:
        data = [dict(sensor_timestamp=r['time'], objects=r['objects'], diagnostics=r) for r in data]
    report = validate_human_mass_identity_retention(data)
    assert report['passed'] and report['retained_source_count'] == 6 and report['original_prior_count'] == 1
    assert json.loads(json.dumps(report, allow_nan=False))['passed']


@pytest.mark.parametrize('change', ['missing_seed', 'new_timestamp', 'confidence', 'bounds', 'body_epoch', 'support'])
def test_provenance_or_uncertainty_tampering_cannot_pass(change):
    data = deepcopy(records())
    row = data[-1]
    mass_input = row['objects'][0]['attributes']['mass']['mass_kg']
    if change == 'missing_seed':
        data = data[1:]
    if change == 'new_timestamp':
        mass_input['timestamp_seconds'] = row['camera_time']
    if change == 'confidence':
        mass_input['confidence'] = .5
    if change == 'bounds':
        mass_input['upper_bound'] = 100.
    if change == 'body_epoch':
        row['human_body_envelope_priors'][0]['lidar_timestamp_seconds'] -= .1
    if change == 'support':
        row['human_body_envelope_priors'][0]['current_support_count'] = 19
    assert not validate_human_mass_identity_retention(data)['passed']

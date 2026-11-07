"""Scoring admission, source clocks and actual multi-object field integration."""

from copy import deepcopy
from dataclasses import asdict, replace

import numpy as np
import pytest

from robot.evaluation.multi_object_perception import ObjectAssessment
from robot.evaluation.restaurant_calibration_runtime import (
    apply_restaurant_calibration, object_is_eligible, restaurant_calibration_preset,
)
from robot.perception.human_body_envelope import HUMAN_ENVELOPE_SOURCE
from robot.perception.physical_attributes import EstimateStatus, PhysicalAttributes, ScalarEstimate
from robot.perception.persistent_object_evidence import PERSISTENT_PROPERTY_MARKER
from robot.perception.temporal_object_evidence import TEMPORAL_EVIDENCE_MARKER
from robot.risk.object_risk_policy import ObjectRiskPriorConfig, evaluate_object_risk
from robot.risk.physical_factor_adapter import PhysicalFactorAdapterConfig, adapt_physical_factors
from robot.risk.restaurant_calibration import CalibratedObjectPolicy


def scalar(value, t=5., *, source="unit_route", marker=None):
    return ScalarEstimate(value, value, value, .5, "1", source, t,
        EstimateStatus.PRIOR, () if marker is None else (marker,))


def object_assessment(key=1, *, accepted=True, returns=4, cue=.4,
                      contact_time=3., adapted_time=5., marker=PERSISTENT_PROPERTY_MARKER,
                      config=None):
    attrs = PhysicalAttributes.unknown(track_id=key, timestamp_seconds=5.)
    attrs = replace(attrs, motion=replace(attrs.motion, route_crossing_likelihood=scalar(.9)))
    proxy = scalar(cue, contact_time, source="visual_silhouette_contact_proxy", marker=marker)
    cfg = config or PhysicalFactorAdapterConfig(allow_visual_contact_proxy=True,
        maximum_persistent_property_age_seconds=120.)
    adapted = adapt_physical_factors(attrs, timestamp_seconds=adapted_time,
        config=cfg, visual_contact_proxy=proxy, visual_contact_basis="visual_silhouette_geometry_proxy",
        contact_interaction_likelihood=scalar(.15, adapted_time))
    policy, details = evaluate_object_risk(adapted, ObjectRiskPriorConfig())
    diagnostics = dict(accepted_instance=accepted, policy_return_count=returns,
        factor_scores=details, contact_proxy=asdict(proxy), property_evidence={}, geometry_evidence={})
    return ObjectAssessment(attrs, adapted, policy, None, diagnostics)


def envelope(key=1):
    return dict(track_id=key, source=HUMAN_ENVELOPE_SOURCE, status="prior",
        original_person_camera_timestamp=4., identity_age_seconds=1.,
        lidar_timestamp_seconds=5., current_support_count=4)


def apply(objects, *, config=None, envelopes=(), carried=None, policies=None):
    return apply_restaurant_calibration(objects, policies or {o.attributes.track_id:o.policy for o in objects},
        config or restaurant_calibration_preset("kinetic"), envelopes, 5., carried_groups=carried)


def test_presets_are_explicit_and_unknown_ownership_is_not_invented():
    assert restaurant_calibration_preset("off") is None
    equal, kinetic = restaurant_calibration_preset("equal"), restaurant_calibration_preset("kinetic")
    assert equal.weights.kinetic == 1 and kinetic.weights.kinetic == 1.5
    assert equal.kinetic_energy_scale_j == kinetic.kinetic_energy_scale_j == 50
    with pytest.raises(ValueError):
        restaurant_calibration_preset("all-safe")
    item = object_assessment(accepted=False)
    objects, policies, audit = apply([item])
    assert objects[0] is item and policies[1] is item.policy
    assert audit["objects"] == []
    assert not object_is_eligible(dict(accepted_instance=True, policy_return_count=0))


def test_helper_uses_adapted_context_clock_and_original_property_age_contract():
    item = object_assessment()
    before = deepcopy(item.diagnostics)
    objects, policies, _ = apply([item])
    contact = policies[1].calibration.as_dict()["factors"]["contact"]
    assert contact["supported"] and contact["context_multiplier"] == pytest.approx(.7875)
    assert contact["contact_proxy"]["timestamp_seconds"] == 3.
    assert contact["contact_proxy"]["evidence"] == (PERSISTENT_PROPERTY_MARKER,)
    assert objects[0].attributes is item.attributes and objects[0].adapted is item.adapted
    assert item.diagnostics == before
    assert objects[0].diagnostics["legacy_factor_scores"] == before["factor_scores"]
    # Contact can be newer than attributes because it is an external adapter
    # input. The original adapter clock, not LiDAR or a fabricated max, applies.
    later = object_assessment(contact_time=5.1, adapted_time=5.1)
    _, policies, _ = apply([later])
    assert policies[1].calibration.factors[1].supported


def test_temporal_marker_gets_only_its_original_retained_lifetime():
    cfg = PhysicalFactorAdapterConfig(allow_visual_contact_proxy=True,
        maximum_retained_contact_age_seconds=2.5)
    item = object_assessment(marker=TEMPORAL_EVIDENCE_MARKER, config=cfg)
    _, policies, _ = apply([item])
    assert policies[1].calibration.factors[1].supported
    stale = object_assessment(marker=TEMPORAL_EVIDENCE_MARKER, config=cfg, contact_time=2.)
    _, policies, _ = apply([stale])
    assert not policies[1].calibration.factors[1].supported


def test_recovered_contact_display_mode_does_not_claim_only_priors():
    item = object_assessment()
    contact = replace(item.adapted.contact,
        inputs=tuple((name, ScalarEstimate.unavailable(unit="1", source="route_missing", timestamp_seconds=5.))
                     if name == "contact_interaction_likelihood" else (name, value)
                     for name, value in item.adapted.contact.inputs),
        estimate=replace(item.adapted.contact.estimate, confidence=0),
        rejection_reasons=("contact_interaction_likelihood: unavailable",))
    adapted = replace(item.adapted, contact=contact,
        profile=replace(item.adapted.profile, contact=contact.estimate))
    policy, details = evaluate_object_risk(adapted, ObjectRiskPriorConfig())
    item = replace(item, adapted=adapted, policy=policy,
        diagnostics=dict(item.diagnostics, factor_scores=details))
    assert item.diagnostics["factor_scores"]["contact"]["mode"] == "configured_prior"
    objects, _, _ = apply([item])
    d = objects[0].diagnostics
    assert d["factor_scores"]["contact"]["mode"] == "evidence_supported"
    assert d["factor_scores"]["contact"]["original_mode"] == "configured_prior"
    assert d["legacy_factor_scores"]["contact"]["mode"] == "configured_prior"
    assert d["factor_scores"]["contact"]["calibration_mode"] == "supported_visual_cue_with_explicit_context_prior"


def test_original_nominal_bound_adapter_requires_explicit_alternative_calibration():
    cfg = PhysicalFactorAdapterConfig(allow_visual_contact_proxy=True, use_upper_bounds=False)
    item = object_assessment(config=cfg, contact_time=5.)
    with pytest.raises(ValueError, match="upper-bound"):
        apply([item])
    unknown = replace(item, diagnostics=dict(item.diagnostics, accepted_instance=False))
    objects, _, _ = apply([unknown])
    assert objects[0] is unknown


def test_verified_human_floor_without_owned_returns_keeps_original_factors():
    item = object_assessment(accepted=False, returns=0)
    objects, policies, audit = apply([item], envelopes=[envelope()])
    assert policies[1].risk == .8
    assert policies[1].calibration.factors == ()
    assert objects[0].diagnostics["factor_scores"] == item.diagnostics["factor_scores"]
    assert policies[1].calibration.physical_phi == item.policy.risk
    assert audit["verified_human_ids"] == [1]


def test_verified_human_current_returns_admit_supported_physical_evidence():
    item = object_assessment(accepted=False)
    _, policies, _ = apply([item], envelopes=[envelope()])
    assert policies[1].calibration.factors[1].supported
    assert policies[1].calibration.physical_phi < item.policy.risk
    assert policies[1].risk == .8
    with pytest.raises(ValueError, match="current verified identity"):
        apply([item], envelopes=[dict(envelope(), lidar_timestamp_seconds=4.9)])


def test_carried_max_uses_recalibrated_individuals_and_preserves_legacy_audit():
    prop = object_assessment(1, cue=.8)
    unknown = object_assessment(2, accepted=False)
    # Original grouping assigned the more conservative prop to both members.
    policies = {1:prop.policy, 2:prop.policy}
    original_groups = dict(confirmed_groups=[dict(members=[1,2], phi=prop.policy.risk)])
    before = deepcopy(original_groups)
    objects, policies, audit = apply([prop, unknown], carried=original_groups, policies=policies)
    expected = max(item.policy.risk for item in objects)
    assert policies[1].risk == policies[2].risk == expected
    assert audit["confirmed_groups"][0]["phi"] == expected
    assert audit["legacy_confirmed_groups"] == before["confirmed_groups"]
    assert original_groups == before
    assert policies[2].risk != prop.policy.risk


def seeded_worker(config):
    from test_live_object_perception import packet
    from test_object_risk_map import mapped_worker, observed, resolve, scalar as observed_scalar
    runtime = mapped_worker()
    runtime.restaurant_calibration_config = config
    first = packet()
    result = runtime(first)
    track = result.field.tracking_result.tracked_obstacles[0]
    resolve(runtime.persistent_properties, track, observed(1.1), 1.1,
        contact=observed_scalar(.4, source="visual_silhouette_contact_proxy"))
    runtime.persistent_properties.freeze_properties()
    runtime.object_ownership_bounds = {track.track_id: (
        first.obstacle_xyz.min(0), first.obstacle_xyz.max(0), 1.1, track.centroid_world_xy)}
    return runtime


def test_real_mapped_worker_reaches_field_with_calibrated_retained_evidence():
    from test_live_object_perception import packet
    runtime = seeded_worker(restaurant_calibration_preset("kinetic"))
    baseline = seeded_worker(None)
    result, original = runtime(packet(2,1.1)), baseline(packet(2,1.1))
    item = result.objects[0]
    assert isinstance(item.policy, CalibratedObjectPolicy)
    assert item.policy.risk < original.objects[0].policy.risk
    assert item.diagnostics["policy_return_count"] == 4
    assert item.diagnostics["restaurant_calibration"]["factors"]["contact"]["supported"]
    assert item.attributes.mass.mass_kg.timestamp_seconds == 1.1
    actual = result.field.pipeline.configuration_space.occupancy_grids[0]
    old = original.field.pipeline.configuration_space.occupancy_grids[0]
    np.testing.assert_array_equal(actual.occupied_mask, old.occupied_mask)
    np.testing.assert_array_equal(actual.cell_track_ids, old.cell_track_ids)
    cells = actual.cell_track_ids == item.attributes.track_id
    assert cells.any() and np.allclose(actual.cell_risks[cells], item.policy.risk)
    audit = result.diagnostics["applied_field_policies"]
    assert audit[0]["cra_phi"] == audit[0]["applied_field_risk"] == item.policy.risk
    assert audit[0]["lambda"] == 1.25


def test_real_mapped_worker_leaves_unassociated_unknown_fields_identical():
    from test_object_risk_map import mapped_worker, multi_packet
    runtime, baseline = mapped_worker(), mapped_worker()
    runtime.restaurant_calibration_config = restaurant_calibration_preset("kinetic")
    result, original = runtime(multi_packet()), baseline(multi_packet())
    assert all(item.policy.risk == .68359375 for item in result.objects)
    np.testing.assert_array_equal(result.field.pipeline.h_full.values, original.field.pipeline.h_full.values)
    assert result.field.all_obstacle_point_count == original.field.all_obstacle_point_count == 9

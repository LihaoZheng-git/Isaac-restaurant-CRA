"""Independent policy and occupancy checks for the optional human experiment."""
import numpy as np
import pytest

from robot.evaluation.restaurant_baselines import applied_field_policies, validate_human_boundary_flux
from robot.perception.human_body_envelope import HumanBodyEnvelope, HumanBodyEnvelopeConfig
from test_human_body_envelope import tracking, cue
from test_restaurant_baselines import policy


def identity():
    return HumanBodyEnvelope(HumanBodyEnvelopeConfig()).update(
        tracking(), cue(), camera_timestamp_seconds=1.)


def test_only_verified_human_and_its_geometry_change_not_physical_cra_or_other_tracks():
    cra = policy()
    policies = {7: cra, 8: cra, 3_000_000_007: cra}
    original = {key: value.as_dict() for key, value in policies.items()}
    normal, _ = applied_field_policies(None, policies, [])
    changed, audit = applied_field_policies(None, policies, [],
        envelope_origins={3_000_000_007: 7}, human_boundary_flux=-1.7,
        verified_human_identities=identity(), sensor_timestamp_seconds=1.)
    for key in (7, 3_000_000_007):
        assert changed[key].boundary_flux == -1.7
        assert changed[key].tangential_strength == 1.25
        row = next(row for row in audit if row['track_or_group_id'] == key)
        assert row['cra_phi'] == cra.risk
        assert row['human_flux_ablation']['original_person_camera_timestamp'] == 1.
        assert row['observed_class'] == 'unknown'  # Retention is not fresh RGB.
    assert changed[8] is normal[8]
    assert original == {key: value.as_dict() for key, value in policies.items()}


def test_no_verified_identity_means_no_control_override_and_never_lower_risk():
    cra = policy()
    same, audit = applied_field_policies(None, {7: cra}, [], human_boundary_flux=-1.7)
    assert same[7] is cra.guidance_parameters
    assert 'human_flux_ablation' not in audit[0]
    low, _ = applied_field_policies(None, {7: cra}, [], human_boundary_flux=-.5,
        verified_human_identities=identity(), sensor_timestamp_seconds=1.)
    assert low[7].risk == cra.risk


@pytest.mark.parametrize('value', [True, False, np.nan, np.inf, -1.8, -.4, '1.7'])
def test_flux_rejects_invalid_or_out_of_mapping_values(value):
    with pytest.raises(ValueError):
        validate_human_boundary_flux(value)


@pytest.mark.parametrize('change', ['stale', 'source', 'duplicate', 'support', 'negative_age'])
def test_override_rejects_unverified_identity_inputs(change):
    rows = identity()
    if change == 'stale':
        rows[0]['lidar_timestamp_seconds'] = .9
    if change == 'source':
        rows[0]['source'] = 'oracle_actor_name'
    if change == 'duplicate':
        rows += rows
    if change == 'support':
        rows[0]['current_support_count'] = 0
    if change == 'negative_age':
        rows[0]['identity_age_seconds'] = -.1
    with pytest.raises(ValueError, match='verified current'):
        applied_field_policies(None, {7: policy()}, [], human_boundary_flux=-1.7,
            verified_human_identities=rows, sensor_timestamp_seconds=1.)


def test_identity_cannot_launder_a_future_camera_source():
    rows = identity()
    rows[0]['original_person_camera_timestamp'] = 2.
    with pytest.raises(ValueError, match='clock differs'):
        applied_field_policies(None, {7: policy()}, [], human_boundary_flux=-1.7,
            verified_human_identities=rows, sensor_timestamp_seconds=1.)


def test_override_cannot_silently_change_a_primary_baseline():
    for baseline in ('fixed_class_safe_sage', 'geometric_velocity_cbf'):
        with pytest.raises(ValueError, match='continuous CRA'):
            applied_field_policies(baseline, {7: policy()}, [], human_boundary_flux=-1.7)


def test_actual_observed_only_field_keeps_raw_geometry_and_identity_without_disk(monkeypatch):
    from test_object_risk_map import mapped_worker
    from test_live_object_perception import packet
    import robot.evaluation.multi_object_perception as multi

    strict, with_disk, observed = mapped_worker(), mapped_worker(), mapped_worker()
    for runtime in (with_disk, observed):
        runtime.human_body_envelope = HumanBodyEnvelope(HumanBodyEnvelopeConfig(minimum_points=3))
        runtime.human_boundary_flux = -1.7
    observed.human_observed_occupancy_only = True
    monkeypatch.setattr(multi, 'fresh_person_selection_evidence', lambda objects, t, frame: {
        objects[0].attributes.track_id: dict(camera_timestamp_seconds=t, camera_frame_id=frame)})
    a, b, c = (runtime(packet()) for runtime in (strict, with_disk, observed))
    np.testing.assert_array_equal(a.source_occupancy_grid.occupied_mask, c.source_occupancy_grid.occupied_mask)
    assert b.source_occupancy_grid.occupied_mask.sum() > c.source_occupancy_grid.occupied_mask.sum()
    assert not np.any(c.source_occupancy_grid.cell_track_ids >= 3_000_000_000)
    assert c.field.all_obstacle_point_count == a.field.all_obstacle_point_count == 4
    assert c.objects[0].policy.as_dict() == a.objects[0].policy.as_dict()
    assert c.diagnostics['human_body_envelope_priors'][0]['applied_to_occupancy'] is False
    policies = c.diagnostics['applied_field_policies']
    assert next(p for p in policies if p['track_or_group_id'] == c.objects[0].attributes.track_id)['boundary_flux'] == -1.7
    assert all(p['boundary_flux'] == pytest.approx(-1.3203125)
               for p in policies if p['track_or_group_id'] != c.objects[0].attributes.track_id)
    before = observed.human_body_envelope.identities.copy()
    monkeypatch.setattr(multi, 'fresh_person_selection_evidence', lambda *args: {})
    later = observed(packet(2, 1.1))
    row = next(p for p in later.diagnostics['applied_field_policies'] if 'human_flux_ablation' in p)
    assert row['human_flux_ablation']['original_person_camera_timestamp'] == 1.1
    assert observed.human_body_envelope.identities[0]['camera_timestamp'] == before[0]['camera_timestamp']
    assert observed.human_body_envelope.identities[0]['lidar_timestamp'] == 1.1


def test_api_rejects_observed_only_without_identity_and_unsafe_option_combinations():
    from test_live_object_perception import worker
    from robot.perception.human_mass_prior import HumanMassPriorConfig
    from robot.perception.persistent_object_evidence import PersistentObjectEvidenceConfig
    from robot.perception.human_fragment_coverage import HumanFragmentCoverageConfig
    from robot.safety.motion_aware_passing import MotionAwarePassingConfig
    from robot.risk.object_risk_policy import ObjectRiskPriorConfig
    with pytest.raises(ValueError, match='verified human identity'):
        worker(human_observed_occupancy_only=True)
    base = dict(person_detector=object(), human_mass_prior_config=HumanMassPriorConfig('unit separated adult'),
        persistent_property_config=PersistentObjectEvidenceConfig('unit separated adult'),
        object_factor_prior_config=ObjectRiskPriorConfig(),
        human_body_envelope_config=HumanBodyEnvelopeConfig(), human_observed_occupancy_only=True)
    with pytest.raises(ValueError, match='unoccupied prior disk'):
        worker(**base, human_fragment_coverage_config=HumanFragmentCoverageConfig('unit separated actor'))
    with pytest.raises(ValueError, match='fresh-only'):
        worker(**base, motion_passing_config=MotionAwarePassingConfig(
            require_visual_person_selection=True, maximum_identity_prior_selection_age_seconds=.5))
    runtime = worker(**base, human_boundary_flux=-1.7)
    assert runtime.human_observed_occupancy_only
    assert runtime.human_boundary_flux == -1.7

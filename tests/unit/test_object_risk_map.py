"""Policy, identity, provenance and actual multi-track field regression gates."""

from dataclasses import replace

import numpy as np
import pytest

from test_live_object_perception import packet, worker
from robot.perception.persistent_object_evidence import (
    PERSISTENT_PROPERTY_MARKER,
    PersistentObjectEvidence,
    PersistentObjectEvidenceConfig,
)
from robot.perception.physical_attributes import (
    EstimateStatus,
    PhysicalAttributes,
    ScalarEstimate,
)
from robot.risk.cra import PhysicalRiskProfile, PhysicalRiskVector, RiskFactorEstimate
from robot.risk.object_risk_policy import ObjectRiskPriorConfig, evaluate_object_risk
from robot.risk.physical_factor_adapter import (
    PhysicalFactorAdapterConfig,
    adapt_physical_factors,
)
from robot.safety.cra_safe_sage_policy import CRASafeSAGEPolicy


def mapped_worker():
    result = worker()
    result.field_config = replace(
        result.field_config,
        guidance_field_config=replace(
            result.field_config.guidance_field_config,
            allow_dense_scene_arbitration=True,
        ),
    )
    result.persistent_properties = PersistentObjectEvidence(
        PersistentObjectEvidenceConfig("synthetic separated static unit fixtures")
    )
    result.object_factor_prior_config = ObjectRiskPriorConfig()
    return result


def scalar(
    value, unit="1", t=1.1, status=EstimateStatus.INFERRED, source="unit_observation"
):
    return ScalarEstimate(value, value, value, 0.75, unit, source, t, status)


def unknown(t=1.1):
    return PhysicalAttributes.unknown(track_id=0, timestamp_seconds=t)


def observed(t=1.1, status=EstimateStatus.INFERRED):
    base = unknown(t)
    return replace(
        base,
        mass=replace(base.mass, mass_kg=scalar(2, "kg", t, status)),
        geometry=replace(base.geometry, fall_height_m=scalar(0, "m", t, status)),
        thermal=replace(
            base.thermal,
            heat_source_likelihood=scalar(
                0.02, t=t, status=status, source="registered_rgb_object_plume_proxy"
            ),
        ),
    )


def test_articulating_fresh_only_mode_never_stores_or_borrows_old_properties():
    track = worker()(packet()).field.tracking_result.tracked_obstacles[0]
    memory = PersistentObjectEvidence(
        PersistentObjectEvidenceConfig(
            "articulating unit actor; fresh observations only",
            retain_observed_properties=False,
        )
    )
    attrs, _, _, _ = resolve(memory, track, observed(), 1.1)
    assert attrs.mass.mass_kg.value == 2
    memory.finish_epoch((track,), 1.1)
    memory.freeze_properties()
    later = replace(track, observation_count=2, matched_previous_track=True)
    attrs, _, _, details = resolve(memory, later, unknown(1.2), 1.2)
    assert not memory.properties and not memory.has_thermal(track.track_id, 1.2)
    assert not attrs.mass.mass_kg.is_available
    assert not attrs.geometry.fall_height_m.is_available
    assert all(d["mode"] != "persistence_prior" for d in details.values())


def resolve(memory, track, attrs, t, *, tracks=None, contact=None, potential=None):
    return memory.resolve(
        track=track,
        tracks=tracks or (track,),
        attributes=attrs,
        potential=potential
        or ScalarEstimate.unavailable(
            unit="1", source="unit_missing", timestamp_seconds=t
        ),
        contact=contact,
        timestamp=t,
        camera_frame_id=round(t * 10),
    )


def test_articulating_mass_only_retention_never_borrows_contact_support_or_heat():
    track = worker()(packet()).field.tracking_result.tracked_obstacles[0]
    memory = PersistentObjectEvidence(PersistentObjectEvidenceConfig(
        "one articulating adult; mass only, 3 s original expiry",
        maximum_property_age_seconds=3, confidence_half_life_seconds=1.5,
        retained_properties=("mass",)))
    resolve(memory, track, observed(), 1.1, contact=scalar(.8))
    assert set(k[1] for k in memory.properties) == {"mass"}
    memory.finish_epoch((track,), 1.1)
    memory.freeze_properties()
    later = replace(track, observation_count=2, matched_previous_track=True)
    attrs, _, contact, details = resolve(memory, later, unknown(1.3), 1.3)
    assert attrs.mass.mass_kg.status is EstimateStatus.PRIOR
    assert attrs.mass.mass_kg.timestamp_seconds == 1.1
    assert attrs.mass.mass_kg.confidence < .75
    assert not attrs.geometry.fall_height_m.is_available
    assert not attrs.thermal.heat_source_likelihood.is_available
    assert contact is None
    assert details["mass"]["mode"] == "persistence_prior"
    assert all(v["mode"] != "persistence_prior" for k, v in details.items() if k != "mass")
    memory.finish_epoch((later,), 1.3)
    attrs, _, _, _ = resolve(memory, later, unknown(4.2), 4.2)
    assert not attrs.mass.mass_kg.is_available  # Borrowing never renews original expiry.


@pytest.mark.parametrize("names", [("mass", "mass"), ("motion",), ["mass"], (3,)])
def test_property_retention_subset_rejects_invalid_or_motion_names(names):
    with pytest.raises(ValueError, match="retained properties"):
        PersistentObjectEvidenceConfig("unit", retained_properties=names)


def test_numeric_priors_do_not_claim_measured_low_factors_or_saturate_unknown_phi():
    adapted = adapt_physical_factors(unknown(), timestamp_seconds=1.1)
    policy, details = evaluate_object_risk(adapted, ObjectRiskPriorConfig())
    assert policy.risk == pytest.approx(1 - 0.75**4)
    assert all(
        item["score"] == 0.25 and item["mode"] == "configured_prior"
        for item in details.values()
    )
    assert all(
        diagnostic.rejection_reasons
        for diagnostic in (
            adapted.contact,
            adapted.thermal,
            adapted.potential,
            adapted.kinetic,
        )
    )
    assert all(
        item.estimate.source == "configured_object_factor_prior"
        for item in policy.assessment.factors
    )
    assert CRASafeSAGEPolicy().evaluate(adapted.profile).risk == 1
    assert policy.tangential_strength == 1.25


def test_thermal_acquisition_is_not_frozen_by_the_first_low_observation():
    track = worker()(packet()).field.tracking_result.tracked_obstacles[0]
    memory = PersistentObjectEvidence(
        PersistentObjectEvidenceConfig("unchanged unit scene")
    )
    with pytest.raises(RuntimeError, match="observe"):
        memory.freeze_properties()
    resolve(memory, track, observed(), 1.1)
    memory.finish_epoch((track,), 1.1)
    assert not memory.has_thermal(track.track_id, 1.1)
    later = replace(track, observation_count=2, matched_previous_track=True)
    hot = replace(
        observed(2),
        thermal=replace(
            observed(2).thermal,
            heat_source_likelihood=scalar(
                0.8, t=2, source="registered_rgb_object_plume_proxy"
            ),
        ),
    )
    resolve(memory, later, hot, 2)
    memory.finish_epoch((later,), 2)
    memory.freeze_properties()
    assert memory.has_thermal(track.track_id, 3)
    attrs, _, _, _ = resolve(memory, replace(later, observation_count=3), unknown(3), 3)
    assert attrs.thermal.heat_source_likelihood.value == 0.8
    assert attrs.thermal.heat_source_likelihood.timestamp_seconds == 2


def test_frozen_intrinsic_estimate_is_not_renewed_by_new_view_dependent_outputs():
    track = worker()(packet()).field.tracking_result.tracked_obstacles[0]
    memory = PersistentObjectEvidence(
        PersistentObjectEvidenceConfig("explicitly unchanged unit object")
    )
    resolve(
        memory,
        track,
        observed(),
        1.1,
        contact=scalar(0.02, source="visual_silhouette_contact_proxy"),
    )
    memory.finish_epoch((track,), 1.1)
    memory.freeze_properties()
    later = replace(track, observation_count=2, matched_previous_track=True)
    current = observed(2)
    current = replace(current, mass=replace(current.mass, mass_kg=scalar(3, "kg", 2)))
    attrs, _, contact, _ = resolve(
        memory,
        later,
        current,
        2,
        contact=scalar(0.04, t=2, source="visual_silhouette_contact_proxy"),
    )
    assert attrs.mass.mass_kg.value == 2 and attrs.mass.mass_kg.timestamp_seconds == 1.1
    assert contact.value == 0.02 and contact.timestamp_seconds == 1.1
    assert attrs.mass.mass_kg.status is EstimateStatus.PRIOR


def test_observed_high_risk_can_still_reach_one_and_prior_sensitivity_is_explicit():
    adapted = adapt_physical_factors(unknown(), timestamp_seconds=1.1)
    profile = replace(
        adapted.profile, contact=RiskFactorEstimate(1, 1, 1, "synthetic_high_contact")
    )
    adapted = replace(
        adapted,
        profile=profile,
        contact=replace(
            adapted.contact, estimate=profile.contact, rejection_reasons=()
        ),
    )
    policy, _ = evaluate_object_risk(adapted, ObjectRiskPriorConfig())
    assert policy.risk == 1
    low, _ = evaluate_object_risk(
        adapt_physical_factors(unknown(), timestamp_seconds=1.1),
        ObjectRiskPriorConfig(PhysicalRiskVector(0.1, 0.1, 0.1, 0.1)),
    )
    assert low.risk < 1 - 0.75**4
    with pytest.raises(ValueError, match="below one"):
        ObjectRiskPriorConfig(PhysicalRiskVector.ones())


def test_properties_keep_original_epochs_and_motion_remains_live():
    runtime = worker()
    track = runtime(packet()).field.tracking_result.tracked_obstacles[0]
    memory = PersistentObjectEvidence(
        PersistentObjectEvidenceConfig("unchanged unit scene")
    )
    resolve(
        memory,
        track,
        observed(),
        1.1,
        contact=scalar(0.02, source="visual_silhouette_contact_proxy"),
        potential=scalar(1),
    )
    later = replace(track, observation_count=2, matched_previous_track=True)
    attrs = replace(
        unknown(3),
        motion=replace(
            unknown(3).motion,
            relative_speed_mps=scalar(0.2, "m/s", 3),
            route_crossing_likelihood=scalar(1, t=3),
        ),
    )
    merged, potential, contact, details = resolve(memory, later, attrs, 3)
    assert merged.motion == attrs.motion
    for item in (
        merged.mass.mass_kg,
        merged.geometry.fall_height_m,
        merged.thermal.heat_source_likelihood,
        contact,
        potential,
    ):
        assert item.status is EstimateStatus.PRIOR
        assert item.timestamp_seconds == 1.1
        assert item.confidence < 0.75
        assert PERSISTENT_PROPERTY_MARKER in item.evidence
    adapted = adapt_physical_factors(
        merged,
        timestamp_seconds=3,
        potential_interaction_likelihood=potential,
        visual_contact_proxy=contact,
        visual_contact_basis="visual_silhouette_geometry_proxy",
        contact_interaction_likelihood=attrs.motion.route_crossing_likelihood,
        thermal_basis="visual_object_plume_proxy",
        thermal_interaction_likelihood=attrs.motion.route_crossing_likelihood,
        config=PhysicalFactorAdapterConfig(
            allow_visual_contact_proxy=True,
            allow_visual_thermal_proxy=True,
            maximum_persistent_property_age_seconds=120,
        ),
    )
    assert not any(
        getattr(adapted, name).rejection_reasons
        for name in ("potential", "kinetic", "contact", "thermal")
    )
    assert all(item["mode"] == "persistence_prior" for item in details.values())
    # Later borrowing is still anchored to the first source, never the last use.
    latest = replace(later, observation_count=3)
    result, _, _, _ = resolve(memory, latest, unknown(122), 122)
    assert not result.mass.mass_kg.is_available
    assert not result.thermal.heat_source_likelihood.is_available


@pytest.mark.parametrize("status", [EstimateStatus.ORACLE, EstimateStatus.FALLBACK])
def test_non_observed_properties_are_never_promoted_to_persistence(status):
    track = worker()(packet()).field.tracking_result.tracked_obstacles[0]
    memory = PersistentObjectEvidence(PersistentObjectEvidenceConfig("unit scope"))
    resolve(memory, track, observed(status=status), 1.1)
    later = replace(track, observation_count=2, matched_previous_track=True)
    attrs, _, _, _ = resolve(memory, later, unknown(2), 2)
    assert not attrs.mass.mass_kg.is_available


def test_visible_vertical_change_invalidates_old_support_without_becoming_fall_height():
    track = worker()(packet()).field.tracking_result.tracked_obstacles[0]
    memory = PersistentObjectEvidence(
        PersistentObjectEvidenceConfig("unit support scope")
    )
    attrs = observed()
    attrs = replace(
        attrs, geometry=replace(attrs.geometry, center_height_m=scalar(0.3, "m"))
    )
    resolve(memory, track, attrs, 1.1, potential=scalar(1))
    raised = unknown(2)
    raised = replace(
        raised, geometry=replace(raised.geometry, center_height_m=scalar(0.7, "m", 2))
    )
    result, potential, _, details = resolve(
        memory,
        replace(track, observation_count=2, matched_previous_track=True),
        raised,
        2,
    )
    assert not result.geometry.fall_height_m.is_available and not potential.is_available
    assert "not fall height" in details["fall_height"]["reason"]


def test_persistence_age_opt_in_never_accepts_stale_live_speed_or_route():
    attrs = observed()
    speed = scalar(0.2, "m/s")
    speed = replace(
        speed, status=EstimateStatus.PRIOR, evidence=(PERSISTENT_PROPERTY_MARKER,)
    )
    attrs = replace(
        attrs,
        timestamp_seconds=3,
        motion=replace(
            attrs.motion,
            relative_speed_mps=speed,
            route_crossing_likelihood=scalar(1, t=3),
        ),
    )
    adapted = adapt_physical_factors(
        attrs,
        timestamp_seconds=3,
        config=PhysicalFactorAdapterConfig(maximum_persistent_property_age_seconds=120),
    )
    assert "relative_speed_mps: stale evidence" in adapted.kinetic.rejection_reasons


def test_support_move_and_competing_identity_invalidate_retained_inputs():
    track = worker()(packet()).field.tracking_result.tracked_obstacles[0]
    memory = PersistentObjectEvidence(PersistentObjectEvidenceConfig("unit scope"))
    resolve(memory, track, observed(), 1.1, potential=scalar(1))
    moved = replace(
        track,
        centroid_world_xy=(track.centroid_world_xy[0] + 0.2, 0),
        observation_count=2,
        matched_previous_track=True,
    )
    attrs, potential, _, details = resolve(memory, moved, unknown(2), 2)
    assert attrs.mass.mass_kg.is_available
    assert not attrs.geometry.fall_height_m.is_available and not potential.is_available
    assert "support location" in details["fall_height"]["reason"]
    competitor = replace(moved, track_id=1)
    latest = replace(moved, observation_count=3)
    attrs, _, _, _ = resolve(memory, latest, unknown(3), 3, tracks=(latest, competitor))
    assert not attrs.mass.mass_kg.is_available


def multi_packet(frame=1, t=1, x=0):
    value = packet(frame, t, x)
    points = np.concatenate(
        (value.obstacle_xyz, value.obstacle_xyz + [0, 1.2, 0], [[2.5, -1, 0.2]])
    )
    return replace(value, obstacle_xyz=points, scene_xyz=points)


def test_moving_returns_expire_without_static_memory_or_source_time_renewal():
    runtime = mapped_worker()
    runtime.occupancy_retention_seconds = 0.15
    for index in range(5):
        value = packet(index + 1, 1 + 0.1 * index)
        points = value.obstacle_xyz + [0, 0.04 * index, 0]
        latest = runtime(replace(value, obstacle_xyz=points, scene_xyz=points))
    retained = latest.diagnostics["retained_occupancy_epochs"]
    assert len(retained) == 1
    assert retained[0]["timestamp"] == pytest.approx(1.3)
    assert retained[0]["sensor_frame_id"] == 4
    assert not runtime.static_points
    assert latest.objects[0].attributes.motion.velocity_y_mps.value == pytest.approx(
        0.4
    )
    assert latest.objects[
        0
    ].attributes.motion.relative_speed_mps.timestamp_seconds == pytest.approx(1.4)


@pytest.mark.parametrize("translated", [False, True])
def test_bounded_translated_ownership_uses_observed_track_displacement_and_original_age(
    translated,
):
    runtime = mapped_worker()
    runtime.persistent_properties.config = replace(
        runtime.persistent_properties.config, translate_instance_bounds_prior=translated
    )
    first = packet()
    result = runtime(first)
    track = result.field.tracking_result.tracked_obstacles[0]
    resolve(runtime.persistent_properties, track, observed(1.1), 1.1)
    runtime.persistent_properties.freeze_properties()
    runtime.object_ownership_bounds = {
        track.track_id: (
            first.obstacle_xyz.min(0),
            first.obstacle_xyz.max(0),
            1.1,
            track.centroid_world_xy,
        )
    }
    second = packet(2, 1.1)
    points = second.obstacle_xyz + [0, 0.16, 0]
    result = runtime(replace(second, obstacle_xyz=points, scene_xyz=points))
    diagnostic = result.objects[0].diagnostics
    assert diagnostic["policy_return_count"] == (4 if translated else 0)
    assert diagnostic["property_evidence"]["mass"]["source_timestamp"] == 1.1
    if translated:
        assert (
            diagnostic["instance_ownership_basis"] == "tracked_translation_bounds_prior"
        )
        assert diagnostic["instance_ownership_translation_xy"] == pytest.approx(
            [0, 0.16]
        )


def test_all_tracks_and_unclustered_occupancy_receive_numeric_field_policies():
    runtime = mapped_worker()
    result = runtime(multi_packet())
    assert len(result.objects) == 2
    assert result.field.all_obstacle_point_count == 9
    assert not result.diagnostics["sensor_perception_ready"]
    assert result.diagnostics["controller_numeric_policy_ready"]
    for item in result.objects:
        assert len(item.diagnostics["factor_scores"]) == 4
        assert item.policy.risk == pytest.approx(1 - 0.75**4)
    grid = result.field.pipeline.configuration_space.occupancy_grids[0]
    assert 2_000_000_000 in grid.cell_track_ids[grid.occupied_mask]
    assert np.all(grid.cell_risks[grid.occupied_mask] < 1)
    second = runtime(multi_packet(2, 1.1, 0.02))
    assert len(second.objects) == 2
    assert all(
        item.attributes.motion.relative_speed_mps.value == pytest.approx(0.2)
        for item in second.objects
    )


def test_retained_mass_never_authorizes_old_body_bounds_for_articulating_actor():
    runtime = mapped_worker()
    runtime.persistent_properties.config = replace(runtime.persistent_properties.config,
        retained_properties=("mass",), allow_instance_bounds_prior=False)
    first = packet()
    result = runtime(first)
    track = result.field.tracking_result.tracked_obstacles[0]
    resolve(runtime.persistent_properties, track, observed(1.1), 1.1)
    runtime.persistent_properties.freeze_properties()
    runtime.object_ownership_bounds = {track.track_id: (
        first.obstacle_xyz.min(0), first.obstacle_xyz.max(0), 1.1, track.centroid_world_xy)}
    result = runtime(packet(2, 1.1))
    assert result.objects[0].diagnostics["property_evidence"]["mass"]["mode"] == "persistence_prior"
    assert result.objects[0].diagnostics["policy_return_count"] == 0
    assert result.field.all_obstacle_point_count == 4


def test_distinct_computed_object_policies_reach_same_field_without_dropping_occupancy():
    runtime = mapped_worker()
    original = runtime._assess_track

    def assess(*args, **kwargs):
        value = original(*args, **kwargs)
        track = args[3]
        profile = PhysicalRiskProfile(
            **{
                factor: RiskFactorEstimate(
                    0.01 if track.track_id == 0 else 0.7,
                    1,
                    1,
                    "synthetic_unit_factor_evidence",
                )
                for factor in ("thermal", "contact", "potential", "kinetic")
            }
        )
        value["adapted"] = replace(
            value["adapted"],
            profile=profile,
            **{
                name: replace(
                    getattr(value["adapted"], name),
                    estimate=getattr(profile, name),
                    rejection_reasons=(),
                )
                for name in ("thermal", "contact", "potential", "kinetic")
            },
        )
        return value

    runtime._assess_track = assess
    result = runtime(multi_packet())
    assert result.objects[0].policy.risk < result.objects[1].policy.risk
    grid = result.field.pipeline.configuration_space.occupancy_grids[0]
    for item in result.objects:
        cells = grid.cell_track_ids == item.attributes.track_id
        assert cells.any()
        assert np.allclose(grid.cell_risks[cells], item.policy.risk)


def test_failed_multi_object_field_does_not_commit_tracker_or_property_memory():
    runtime = mapped_worker()
    first = runtime(multi_packet())
    memory = runtime.persistent_properties
    with pytest.raises(ValueError):
        runtime(multi_packet(2, 1.5))
    assert runtime.previous is first.field
    assert runtime.persistent_properties is memory
    recovered = runtime(multi_packet(3, 1.1))
    assert all(
        track.observation_count == 2
        for track in recovered.field.tracking_result.tracked_obstacles
    )


def test_temporal_gap_requires_unpublished_baseline_then_actual_fresh_pair():
    runtime = mapped_worker()
    first = runtime(multi_packet())
    with pytest.raises(ValueError, match="unpublished recovery baseline"):
        runtime(multi_packet(2, 1.5))
    assert runtime.previous is first.field
    assert runtime.temporal_recovery_baseline[:2] == (2, 1.5)
    recovered = runtime(multi_packet(3, 1.6))
    assert recovered.field.sensor_timestamp_seconds == 1.6
    assert (
        recovered.field.temporal_safety_derivative.time_delta_seconds
        == pytest.approx(0.1)
    )
    assert recovered.diagnostics["temporal_gap_recovery"]["used"]
    assert recovered.diagnostics["temporal_gap_recovery"]["baseline_timestamp"] == 1.5
    assert runtime.temporal_recovery_baseline is None


def test_frozen_visual_cadence_keeps_all_track_motion_current_and_shares_image_encoding():
    from robot.perception.sam_instance_segmentation import SamInstanceResult

    class Segmenter:
        images = 0
        proposals = 0

        def set_image(self, rgb):
            self.images += 1

        def propose(self, *args, **kwargs):
            self.proposals += 1
            return SamInstanceResult(None, ("controlled unit mask miss",))

    runtime = mapped_worker()
    segmenter = runtime.segmenter = Segmenter()
    rotation = np.eye(4)
    rotation[:3, :3] = [[0, -1, 0], [0, 0, 1], [-1, 0, 0]]

    def value(frame, t):
        base = packet(frame, t)
        points = np.concatenate(
            (
                np.repeat(base.obstacle_xyz, 2, axis=0),
                np.repeat(base.obstacle_xyz + [0, 0.45, 0], 2, axis=0),
                np.repeat(base.obstacle_xyz + [0.5, 0, 0], 2, axis=0),
            )
        )
        return replace(
            base, obstacle_xyz=points, scene_xyz=points, camera_from_scan_robot=rotation
        )

    result = runtime(value(1, 1))
    assert len(result.objects) == 3
    assert segmenter.images == 1 and segmenter.proposals == 3
    runtime.persistent_properties.freeze_properties()
    result = runtime(value(2, 1.1))
    assert segmenter.images == 1 and segmenter.proposals == 3
    assert all(not item.diagnostics["visual_attempted"] for item in result.objects)
    assert all(
        item.attributes.motion.relative_speed_mps.timestamp_seconds == 1.1
        for item in result.objects
    )
    for index in range(2, 11):
        result = runtime(value(index + 1, 1 + index * 0.1))
    assert segmenter.images == 2 and segmenter.proposals == 5
    assert sum(item.diagnostics["visual_attempted"] for item in result.objects) == 2
    result = runtime(value(12, 2.1))
    assert segmenter.images == 3 and segmenter.proposals == 6
    assert sum(item.diagnostics["visual_attempted"] for item in result.objects) == 1
    assert all(len(item.diagnostics["factor_scores"]) == 4 for item in result.objects)

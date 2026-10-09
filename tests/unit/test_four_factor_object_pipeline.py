"""Combined synthetic sensor controls; scores are computed, never injected.

Object masks and bindings are fixture priors here, not real detector accuracy.
Fresh Isaac captures separately exercise the actual detector/LiDAR association.
"""

from dataclasses import replace

import cv2
import numpy as np
import pytest

from robot.evaluation.cra_experiments import EvidenceMode, RestaurantScenario
from robot.evaluation.perception_readiness import evaluate_perception_readiness
from robot.perception.camera_lidar_fusion import TrackSemanticAssociation
from robot.perception.ground_support import estimate_spherical_ground_support
from robot.perception.motion_attributes import RouteMotionContext, estimate_track_motion
from robot.perception.object_plume_evidence import (
    OBJECT_PLUME_BASIS,
    ObjectPlumeObservation,
    estimate_object_plume_evidence,
)
from robot.perception.physical_attributes import (
    EstimateStatus,
    PhysicalAttributes,
    ScalarEstimate,
)
from robot.perception.semantic_detection import (
    PixelBoundingBox,
    SemanticDetection,
    SemanticDetectionFrame,
)
from robot.perception.specialist_track_evidence import SpecialistAssociationFrame
from robot.perception.spherical_mass import (
    estimate_spherical_geometry,
    estimate_spherical_mass,
)
from robot.perception.thermal_interaction import estimate_thermal_route_interaction
from robot.perception.visual_shape_contact import (
    SHAPE_CONTACT_BASIS,
    estimate_silhouette_contact_proxy,
)
from robot.risk.physical_factor_adapter import (
    PhysicalFactorAdapterConfig,
    adapt_physical_factors,
)
from robot.safety.cra_safe_sage_policy import CRASafeSAGEPolicy
from robot.safety.obstacle_tracking import TrackedObstacle


def prior(value, lower, upper, unit):
    return ScalarEstimate(
        value,
        lower,
        upper,
        0.5,
        unit,
        "synthetic_fixture_prior",
        2,
        EstimateStatus.PRIOR,
    )


def pipeline(
    *,
    radius=0.15,
    velocity=0.0,
    pointed=False,
    plume=False,
    plume_fraction=1.0,
    continuity=True,
    density=True,
):
    angles = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    zs = np.linspace(-0.8, 0.8, 15)
    cloud = np.asarray(
        [
            (
                radius * np.sqrt(1 - z * z) * np.cos(a),
                radius * np.sqrt(1 - z * z) * np.sin(a),
                radius * (1 + z),
            )
            for z in zs
            for a in angles
        ]
    )
    spherical = estimate_spherical_geometry(cloud, timestamp_seconds=2)
    floor = np.asarray(
        [
            (x, y, 0)
            for x in np.linspace(0.8, 1.7, 25)
            for y in np.linspace(-0.8, 0.8, 31)
        ]
    )
    ground = estimate_spherical_ground_support(
        floor,
        spherical,
        timestamp_seconds=2,
        scene_timestamp_seconds=2,
        ground_continuity_prior=prior(1, 1, 1, "1") if continuity else None,
        prior_scope="synthetic flat-floor control",
    )
    mass = estimate_spherical_mass(
        spherical,
        effective_density_kg_m3=prior(25, 5, 100, "kg/m^3") if density else None,
        timestamp_seconds=2,
    )
    track = TrackedObstacle(
        1, (0, 0), (0, 0), (velocity, 0), (velocity, 0), len(cloud), 8, True
    )
    motion = estimate_track_motion(
        track,
        timestamp_seconds=2,
        route_context=RouteMotionContext((-1, 0), (2, 0), 0.2),
    )
    mask = np.zeros((300, 300), np.uint8)
    if pointed:
        cv2.fillPoly(mask, [np.asarray([[150, 205], [175, 255], [125, 255]])], 1)
    else:
        cv2.circle(mask, (150, 230), 25, 1, -1)
    detected = SemanticDetection(
        0, "arbitrary", 0.95, PixelBoundingBox(125, 205, 176, 256), mask.astype(bool)
    )
    reference = np.full((300, 300, 3), 20, np.uint8)
    observations = []
    link = TrackSemanticAssociation(1, "matched", 100, 0, 100, 1, 1, True, (150, 230))
    for i, timestamp in enumerate((1.8, 1.9, 2.0)):
        image = reference.copy()
        if plume:
            image[195 - round(55 * plume_fraction):195, 142:158] = 220
        frame = SemanticDetectionFrame(
            i + 10, timestamp, 300, 300, (detected,), "fixture", 0
        )
        links = SpecialistAssociationFrame(i + 10, timestamp, 300, 300, (link,))
        observations.append(
            ObjectPlumeObservation(
                image, frame, links, timestamp, True, EstimateStatus.PRIOR
            )
        )
    thermal = estimate_object_plume_evidence(
        reference,
        reference_timestamp_seconds=1.7,
        observations=tuple(observations),
        timestamp_seconds=2,
        clean_reference_validated=True,
        negative_validation_scope="synthetic positive/negative RGB control",
    )[1].thermal
    contact = estimate_silhouette_contact_proxy(
        detected, timestamp_seconds=2, evidence_status=EstimateStatus.PRIOR
    )
    attributes = PhysicalAttributes.unknown(track_id=1, timestamp_seconds=2)
    attributes = replace(
        attributes,
        mass=mass,
        motion=motion,
        thermal=thermal,
        geometry=replace(attributes.geometry, fall_height_m=ground.fall_height_m),
    )
    adapted = adapt_physical_factors(
        attributes,
        timestamp_seconds=2,
        potential_interaction_likelihood=ground.potential_interaction_likelihood,
        thermal_interaction_likelihood=estimate_thermal_route_interaction(
            thermal, motion.route_crossing_likelihood, timestamp_seconds=2
        ),
        thermal_basis=OBJECT_PLUME_BASIS,
        visual_contact_proxy=contact,
        visual_contact_basis=SHAPE_CONTACT_BASIS,
        contact_interaction_likelihood=motion.route_crossing_likelihood,
        config=PhysicalFactorAdapterConfig(
            allow_visual_contact_proxy=True, allow_visual_thermal_proxy=True
        ),
    )
    return attributes, adapted, CRASafeSAGEPolicy().evaluate(adapted.profile)


def test_four_computed_factors_reach_policy_with_prior_readiness_only():
    attrs, adapted, policy = pipeline()
    assert all(
        not getattr(adapted, name).rejection_reasons
        for name in ("potential", "kinetic", "contact", "thermal")
    )
    for mode, expected in ((EvidenceMode.PRIOR, True), (EvidenceMode.SENSOR, False)):
        ready = evaluate_perception_readiness(
            RestaurantScenario("fixture", "potential", "control", mode, 0), adapted
        )
        assert ready.ready is expected
        assert ready.aggregate_phi_reportable is expected
    assert 0 < policy.risk < 1
    assert policy.risk == policy.guidance_parameters.risk
    assert policy.tangential_strength == 1.25
    assert attrs.contact.sharpness.status is EstimateStatus.UNAVAILABLE
    assert attrs.thermal.steam_likelihood.status is EstimateStatus.UNAVAILABLE


@pytest.mark.parametrize(
    "change,factor",
    [
        ({"radius": 0.3}, "kinetic"),
        ({"velocity": -0.6}, "kinetic"),
        ({"pointed": True}, "contact"),
        ({"plume": True}, "thermal"),
    ],
)
def test_independent_observed_changes_raise_corresponding_risk_and_phi(change, factor):
    _, base, base_policy = pipeline()
    _, changed, changed_policy = pipeline(**change)
    assert (
        getattr(changed.profile, factor).nominal_risk
        > getattr(base.profile, factor).nominal_risk
    )
    assert changed_policy.risk > base_policy.risk
    assert changed_policy.boundary_flux < base_policy.boundary_flux
    assert changed_policy.tangential_strength == base_policy.tangential_strength == 1.25


@pytest.mark.parametrize("change", [{"continuity": False}, {"density": False}])
def test_missing_inputs_still_saturate_in_combined_pipeline(change):
    _, result, policy = pipeline(**change)
    assert result.potential.rejection_reasons
    assert policy.risk == 1


def test_continuous_plume_coverage_reaches_thermal_factor_and_policy():
    results = [pipeline(plume=fraction > 0, plume_fraction=fraction)
               for fraction in (0, .25, 1)]
    cues = [attrs.thermal.heat_source_likelihood.value for attrs, _, _ in results]
    thermal = [adapted.profile.thermal.nominal_risk for _, adapted, _ in results]
    phi = [policy.risk for _, _, policy in results]
    assert cues[0] < cues[1] < cues[2]
    assert thermal[0] < thermal[1] < thermal[2]
    assert phi[0] < phi[1] < phi[2]
    assert all(policy.tangential_strength == 1.25 for _, _, policy in results)
    # The paired intervention must not silently change other factor evidence.
    for name in ("potential", "kinetic", "contact"):
        assert len({getattr(result[1].profile, name).nominal_risk for result in results}) == 1

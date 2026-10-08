"""Independent geometry fixtures and provenance/CRA regression checks."""

from dataclasses import replace

import cv2
import numpy as np
import pytest

from robot.evaluation.cra_experiments import EvidenceMode, RestaurantScenario
from robot.evaluation.perception_readiness import (
    EvaluationScope,
    evaluate_perception_readiness,
)
from robot.perception.camera_lidar_fusion import TrackSemanticAssociation
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
from robot.perception.specialist_track_evidence import (
    SpecialistAssociationFrame,
    build_track_shape_contact_proxies,
)
from robot.perception.spherical_mass import (
    SphereFitConfig,
    estimate_spherical_geometry,
    estimate_spherical_mass,
)
from robot.perception.visual_shape_contact import (
    SHAPE_CONTACT_BASIS,
    SHAPE_CONTACT_SOURCE,
    SilhouetteContactConfig,
    combine_shape_and_part_contact,
    estimate_silhouette_contact_proxy,
)
from robot.risk.physical_factor_adapter import (
    PhysicalFactorAdapterConfig,
    adapt_physical_factors,
)
from robot.risk.physical_profile_provider import PhysicalAttributeCRARiskProfileProvider
from robot.safety.cra_safe_sage_policy import CRASafeSAGEPolicy


def detection(kind="circle", confidence=0.95, class_name="arbitrary"):
    mask = np.zeros((160, 160), np.uint8)
    if kind == "circle":
        cv2.circle(mask, (80, 80), 45, 1, -1)
    elif kind == "triangle":
        cv2.fillPoly(mask, [np.array([[80, 25], [130, 125], [30, 125]])], 1)
    elif kind == "thin":
        mask[74:86, 25:135] = 1
    elif kind == "jagged":
        angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        radii = np.where(np.arange(24) % 2, 30, 50)
        vertices = np.column_stack(
            (80 + radii * np.cos(angles), 80 + radii * np.sin(angles))
        ).astype(np.int32)
        cv2.fillPoly(mask, [vertices], 1)
    return SemanticDetection(
        0, class_name, confidence, PixelBoundingBox(0, 0, 160, 160), mask.astype(bool)
    )


def points(radius=0.2, center=(1.0, 0.0, 0.3)):
    # Dense visible hemisphere; no authored shape enters runtime except as
    # synthetic test input with independently known expected results.
    a, b = np.meshgrid(np.linspace(0.25, 1.35, 16), np.linspace(0, 2 * np.pi, 25))
    unit = np.column_stack(
        (
            -np.cos(a.ravel()),
            np.sin(a.ravel()) * np.cos(b.ravel()),
            np.sin(a.ravel()) * np.sin(b.ravel()),
        )
    )
    return radius * unit + center


def scalar(
    value, unit="1", status=EstimateStatus.INFERRED, source="fixture", timestamp=1.0
):
    return ScalarEstimate(value, value, value, 0.9, unit, source, timestamp, status)


def test_round_outline_is_lower_than_independent_pointed_elongated_jagged_shapes():
    values = {
        name: estimate_silhouette_contact_proxy(detection(name), timestamp_seconds=1.0)
        for name in ("circle", "triangle", "thin", "jagged")
    }
    assert 0 <= values["circle"].value < 0.15
    assert all(
        values[name].value > values["circle"].value + 0.1
        for name in ("triangle", "thin", "jagged")
    )
    assert values["circle"].upper_bound > values["circle"].value
    assert values["circle"].status is EstimateStatus.INFERRED


def test_class_names_do_not_change_shape_score_and_oracle_stays_oracle():
    first = estimate_silhouette_contact_proxy(
        detection(class_name="knife"), timestamp_seconds=1.0
    )
    second = estimate_silhouette_contact_proxy(
        detection(class_name="ball"), timestamp_seconds=1.0
    )
    assert first == second
    oracle = estimate_silhouette_contact_proxy(
        detection(), timestamp_seconds=1.0, evidence_status=EstimateStatus.ORACLE
    )
    assert oracle.status is EstimateStatus.ORACLE


@pytest.mark.parametrize(
    "case", ["tiny", "border", "fragmented", "hole", "low_confidence"]
)
def test_insufficient_or_incomplete_masks_are_unknown(case):
    item = detection()
    mask = item.mask.copy()
    if case == "tiny":
        mask[:] = False
        mask[75:80, 75:80] = True
    elif case == "border":
        mask[:80, 70:90] = True
    elif case == "fragmented":
        mask[70:90, 20:140] = False
    elif case == "hole":
        mask[75:85, 75:85] = False
    elif case == "low_confidence":
        item = replace(item, confidence=0.1)
    result = estimate_silhouette_contact_proxy(
        replace(item, mask=mask), timestamp_seconds=1.0
    )
    assert result.status is EstimateStatus.UNAVAILABLE
    assert result.value is None


def test_edge_evidence_cannot_be_suppressed_by_smooth_outline():
    shape = estimate_silhouette_contact_proxy(detection(), timestamp_seconds=1.0)
    part = scalar(0.8, timestamp=0.9)
    combined = combine_shape_and_part_contact(shape, part)
    assert combined.value == 0.8
    assert combined.upper_bound >= part.upper_bound
    assert combined.timestamp_seconds == 0.9
    assert combine_shape_and_part_contact(shape, None) == shape


def association(track_id=1, index=0):
    return TrackSemanticAssociation(
        track_id, "matched", 100, index, 100, 1.0, 1.0, True, (80.0, 80.0)
    )


def frames(associations=None):
    frame = SemanticDetectionFrame(3, 1.0, 160, 160, (detection(),), "fixture", 0.0)
    links = SpecialistAssociationFrame(
        3, 1.0, 160, 160, associations or (association(),)
    )
    return frame, links


def test_track_binding_rejects_ambiguity_and_frame_mismatch():
    frame, links = frames()
    assert set(build_track_shape_contact_proxies(frame, links)) == {1}
    assert not build_track_shape_contact_proxies(
        frame, replace(links, associations=(association(1), association(2)))
    )
    assert not build_track_shape_contact_proxies(
        frame,
        replace(links, associations=(association(1), association(2))),
        part_proxies={1: scalar(0.8)},
    )
    with pytest.raises(ValueError, match="same image timestamp"):
        build_track_shape_contact_proxies(
            frame, links, part_proxies={1: scalar(0.8, timestamp=0.9)}
        )
    with pytest.raises(ValueError, match="align"):
        build_track_shape_contact_proxies(
            frame, replace(links, source_timestamp_seconds=0.9)
        )
    with pytest.raises(ValueError, match="index"):
        build_track_shape_contact_proxies(
            frame, replace(links, associations=(association(index=2),))
        )


@pytest.mark.parametrize("part", [scalar(1.2), scalar(0.5, "m")])
def test_invalid_part_proxies_rejected_even_with_missing_shape(part):
    missing = ScalarEstimate.unavailable(
        unit="1", source="missing", timestamp_seconds=1.0
    )
    with pytest.raises(ValueError):
        combine_shape_and_part_contact(missing, part)


def test_lidar_fit_recovers_translated_visible_sphere_and_density_prior_keeps_status():
    geometry = estimate_spherical_geometry(points(), timestamp_seconds=1.0)
    assert geometry.radius_m.value == pytest.approx(0.2, abs=1e-8)
    assert geometry.center_xyz == pytest.approx((1.0, 0.0, 0.3))
    assert geometry.radius_m.lower_bound < 0.2 < geometry.radius_m.upper_bound
    assert geometry.volume_m3.status is EstimateStatus.INFERRED
    density = scalar(25.0, "kg/m^3", EstimateStatus.PRIOR)
    mass = estimate_spherical_mass(
        geometry, effective_density_kg_m3=density, timestamp_seconds=1.0
    )
    assert mass.mass_kg.status is EstimateStatus.PRIOR
    assert mass.mass_kg.value == pytest.approx(4 * np.pi * 0.2**3 / 3 * 25)
    assert not estimate_spherical_mass(
        geometry, effective_density_kg_m3=None, timestamp_seconds=1.0
    ).mass_kg.is_available


def test_double_radius_gives_eightfold_heaviness_only_under_same_density():
    density = scalar(25.0, "kg/m^3", EstimateStatus.PRIOR)
    masses = [
        estimate_spherical_mass(
            estimate_spherical_geometry(points(r), timestamp_seconds=1.0),
            effective_density_kg_m3=density,
            timestamp_seconds=1.0,
        ).mass_kg
        for r in (0.15, 0.3)
    ]
    assert masses[1].value / masses[0].value == pytest.approx(8.0)


@pytest.mark.parametrize("case", ["flat", "tiny_patch", "non_sphere", "too_few"])
def test_unsupported_lidar_shapes_are_unavailable(case):
    cloud = points()
    if case == "flat":
        cloud[:, 0] = 1.0
    elif case == "tiny_patch":
        cloud = cloud[np.linalg.norm(cloud[:, 1:3] - [0.0, 0.3], axis=1) < 0.08]
    elif case == "non_sphere":
        cloud[:, 2] *= 2.0
    elif case == "too_few":
        cloud = cloud[:3]
    result = estimate_spherical_geometry(cloud, timestamp_seconds=1.0)
    assert not result.volume_m3.is_available
    assert result.rejection_reasons


@pytest.mark.parametrize("bad", [np.zeros((5, 2)), np.array([[np.nan, 0, 0]])])
def test_bad_lidar_data_rejected(bad):
    with pytest.raises(ValueError, match="finite Nx3"):
        estimate_spherical_geometry(bad, timestamp_seconds=1.0)


def test_future_density_rejected_and_stale_contact_remains_unknown():
    geometry = estimate_spherical_geometry(points(), timestamp_seconds=1.0)
    with pytest.raises(ValueError, match="newer"):
        estimate_spherical_mass(
            geometry,
            effective_density_kg_m3=scalar(25.0, "kg/m^3", timestamp=2.0),
            timestamp_seconds=1.0,
        )
    contact = estimate_silhouette_contact_proxy(detection(), timestamp_seconds=1.0)
    result = adapt_physical_factors(
        PhysicalAttributes.unknown(track_id=1, timestamp_seconds=3.0),
        timestamp_seconds=3.0,
        visual_contact_proxy=contact,
        visual_contact_basis=SHAPE_CONTACT_BASIS,
        contact_interaction_likelihood=scalar(0.8, timestamp=3.0),
        config=PhysicalFactorAdapterConfig(allow_visual_contact_proxy=True),
    )
    assert result.contact.rejection_reasons


def test_computed_shape_and_heaviness_reach_cra_with_unknowns_preserved():
    results = []
    for radius, kind in ((0.15, "circle"), (0.3, "thin")):
        mass = estimate_spherical_mass(
            estimate_spherical_geometry(points(radius), timestamp_seconds=1.0),
            effective_density_kg_m3=scalar(25.0, "kg/m^3", EstimateStatus.PRIOR),
            timestamp_seconds=1.0,
        )
        base = PhysicalAttributes.unknown(track_id=1, timestamp_seconds=1.0)
        attributes = replace(
            base,
            mass=mass,
            motion=replace(
                base.motion,
                relative_speed_mps=scalar(0.2, "m/s"),
                route_crossing_likelihood=scalar(0.8),
            ),
        )
        shape = estimate_silhouette_contact_proxy(
            detection(kind), timestamp_seconds=1.0
        )
        provider = PhysicalAttributeCRARiskProfileProvider(
            {1: attributes},
            timestamp_seconds=1.0,
            visual_contact_proxies={1: shape},
            visual_contact_bases={1: SHAPE_CONTACT_BASIS},
            contact_interactions={1: scalar(0.8)},
            config=PhysicalFactorAdapterConfig(allow_visual_contact_proxy=True),
        )
        result = provider.diagnostics_by_track_id[1]
        assert result.contact.basis == SHAPE_CONTACT_BASIS
        assert result.kinetic.energy_j.status is EstimateStatus.PRIOR
        assert result.potential.rejection_reasons and result.thermal.rejection_reasons
        assert CRASafeSAGEPolicy().evaluate(result.profile).assessment.phi == 1.0
        assert attributes.contact.sharpness.status is EstimateStatus.UNAVAILABLE
        scenario = RestaurantScenario(
            "shape", "contact", "bounded proxy", EvidenceMode.SENSOR, 0
        )
        assert evaluate_perception_readiness(
            scenario, result, scope=EvaluationScope.FACTOR_SPECIFIC
        ).ready
        assert not evaluate_perception_readiness(
            scenario, result
        ).aggregate_phi_reportable
        results.append(result)
    assert results[0].kinetic.estimate.severity < results[1].kinetic.estimate.severity
    assert results[0].contact.estimate.severity < results[1].contact.estimate.severity


def test_new_basis_requires_opt_in_and_correct_source():
    base = PhysicalAttributes.unknown(track_id=1, timestamp_seconds=1.0)
    shape = estimate_silhouette_contact_proxy(detection(), timestamp_seconds=1.0)
    with pytest.raises(ValueError, match="opt-in"):
        adapt_physical_factors(
            base,
            timestamp_seconds=1.0,
            visual_contact_proxy=shape,
            visual_contact_basis=SHAPE_CONTACT_BASIS,
        )
    with pytest.raises(ValueError, match="source"):
        adapt_physical_factors(
            base,
            timestamp_seconds=1.0,
            visual_contact_proxy=scalar(0.1),
            visual_contact_basis=SHAPE_CONTACT_BASIS,
            config=PhysicalFactorAdapterConfig(allow_visual_contact_proxy=True),
        )
    assert shape.source == SHAPE_CONTACT_SOURCE


@pytest.mark.parametrize(
    "factory,kwargs",
    [
        (SphereFitConfig, {"minimum_points": 3}),
        (SphereFitConfig, {"confidence_cap": 1.1}),
        (SphereFitConfig, {"minimum_normal_span": 3.0}),
        (SilhouetteContactConfig, {"minimum_pixels": 0}),
        (SilhouetteContactConfig, {"polygon_tolerance_fraction": 0.5}),
    ],
)
def test_invalid_configs_rejected(factory, kwargs):
    with pytest.raises(ValueError):
        factory(**kwargs)

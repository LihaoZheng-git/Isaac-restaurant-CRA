from dataclasses import FrozenInstanceError

import pytest

from robot.perception.mass_estimation import (
    MASS_FUSION_SOURCE,
    MASS_PRIOR_SCHEMA_VERSION,
    VOLUME_DENSITY_MASS_SOURCE,
    BoxVolumeConfig,
    ClassMassPrior,
    ClassMassPriorTable,
    estimate_box_volume,
    estimate_mass_from_volume_density,
    estimate_track_mass,
    fuse_mass_estimates,
)
from robot.perception.physical_attributes import (
    EstimateStatus,
    GeometryAttributes,
    ScalarEstimate,
)


def scalar(
    value: float,
    *,
    lower: float | None = None,
    upper: float | None = None,
    confidence: float = 0.8,
    unit: str,
    source: str,
    timestamp: float = 1.0,
    status: EstimateStatus = EstimateStatus.INFERRED,
) -> ScalarEstimate:
    return ScalarEstimate(
        value=value,
        lower_bound=value if lower is None else lower,
        upper_bound=value if upper is None else upper,
        confidence=confidence,
        unit=unit,
        source=source,
        timestamp_seconds=timestamp,
        status=status,
    )


def geometry(
    *,
    source: str = "rgbd_complete_object_geometry",
    confidence: float = 0.8,
) -> GeometryAttributes:
    return GeometryAttributes(
        width_m=scalar(
            0.5,
            lower=0.4,
            upper=0.6,
            confidence=confidence,
            unit="m",
            source=source,
        ),
        depth_m=scalar(
            0.4,
            lower=0.3,
            upper=0.5,
            confidence=confidence,
            unit="m",
            source=source,
        ),
        height_m=scalar(
            1.0,
            lower=0.9,
            upper=1.1,
            confidence=confidence,
            unit="m",
            source=source,
        ),
        center_height_m=scalar(
            0.5,
            lower=0.4,
            upper=0.6,
            confidence=confidence,
            unit="m",
            source=source,
        ),
        fall_height_m=ScalarEstimate.unavailable(
            unit="m",
            source="stability_required",
            timestamp_seconds=1.0,
        ),
    )


def prior_table() -> ClassMassPriorTable:
    return ClassMassPriorTable(
        {
            "dining chair": ClassMassPrior(
                nominal_kg=6.0,
                lower_kg=3.0,
                upper_kg=12.0,
                confidence=0.35,
                source="controlled_benchmark_measurement",
                evidence=("illustrative unit-test prior",),
            )
        }
    )


def test_mass_prior_schema_version_is_explicit() -> None:
    assert MASS_PRIOR_SCHEMA_VERSION == 1


def test_class_prior_lookup_normalizes_semantic_name() -> None:
    mass = prior_table().estimate(
        "  Dining-Chair ",
        timestamp_seconds=2.0,
    ).mass_kg

    assert mass.value == pytest.approx(6.0)
    assert mass.lower_bound == pytest.approx(3.0)
    assert mass.upper_bound == pytest.approx(12.0)
    assert mass.status is EstimateStatus.PRIOR
    assert mass.timestamp_seconds == pytest.approx(2.0)


def test_unconfigured_class_is_explicitly_unavailable() -> None:
    mass = prior_table().estimate(
        "person",
        timestamp_seconds=2.0,
    ).mass_kg

    assert not mass.is_available
    assert "no configured mass prior" in mass.evidence[0]


def test_prior_table_copies_input_and_is_read_only() -> None:
    source = {
        "chair": ClassMassPrior(
            nominal_kg=5.0,
            lower_kg=2.0,
            upper_kg=10.0,
            confidence=0.3,
            source="unit_test",
        )
    }
    table = ClassMassPriorTable(source)
    source.clear()

    assert "chair" in table.priors
    with pytest.raises(TypeError):
        table.priors["person"] = source  # type: ignore[index,assignment]
    with pytest.raises(FrozenInstanceError):
        table.source = "changed"  # type: ignore[misc]


def test_duplicate_normalized_prior_names_are_rejected() -> None:
    item = ClassMassPrior(
        nominal_kg=1.0,
        lower_kg=0.5,
        upper_kg=2.0,
        confidence=0.4,
        source="unit_test",
    )

    with pytest.raises(ValueError, match="Duplicate"):
        ClassMassPriorTable({"hot-soup": item, "hot soup": item})


@pytest.mark.parametrize(
    "changes",
    [
        {"nominal_kg": -1.0},
        {"lower_kg": -0.1},
        {"lower_kg": 8.0, "upper_kg": 2.0},
        {"confidence": 1.1},
        {"source": ""},
    ],
)
def test_invalid_class_prior_is_rejected(changes: dict[str, object]) -> None:
    values: dict[str, object] = {
        "nominal_kg": 5.0,
        "lower_kg": 2.0,
        "upper_kg": 8.0,
        "confidence": 0.4,
        "source": "unit_test",
    }
    values.update(changes)

    with pytest.raises((TypeError, ValueError)):
        ClassMassPrior(**values)  # type: ignore[arg-type]


def test_complete_geometry_produces_box_volume_with_interval_products() -> None:
    volume = estimate_box_volume(
        geometry(),
        timestamp_seconds=2.0,
    )

    assert volume.value == pytest.approx(0.2)
    assert volume.lower_bound == pytest.approx(0.108)
    assert volume.upper_bound == pytest.approx(0.33)
    assert volume.unit == "m^3"
    assert volume.confidence == pytest.approx(0.8)


def test_visible_lidar_envelope_is_not_assumed_to_be_full_volume() -> None:
    volume = estimate_box_volume(
        geometry(source="lidar_track_visible_envelope"),
        timestamp_seconds=2.0,
    )

    assert not volume.is_available
    assert "not complete geometry" in volume.evidence[0]


def test_low_confidence_geometry_is_not_used_for_volume() -> None:
    volume = estimate_box_volume(
        geometry(confidence=0.3),
        timestamp_seconds=2.0,
    )

    assert not volume.is_available
    assert "below the configured threshold" in volume.evidence[0]


def test_explicit_policy_can_accept_a_named_geometry_source() -> None:
    volume = estimate_box_volume(
        geometry(source="calibrated_multiview_geometry"),
        timestamp_seconds=2.0,
        config=BoxVolumeConfig(
            accepted_geometry_sources=("calibrated_multiview_geometry",),
        ),
    )

    assert volume.is_available


def test_volume_corruption_guard_returns_unavailable() -> None:
    volume = estimate_box_volume(
        geometry(),
        timestamp_seconds=2.0,
        config=BoxVolumeConfig(maximum_volume_m3=0.1),
    )

    assert not volume.is_available
    assert "corruption guard" in volume.evidence[0]


def test_volume_density_mass_uses_nonnegative_interval_arithmetic() -> None:
    volume = scalar(
        0.2,
        lower=0.1,
        upper=0.3,
        confidence=0.8,
        unit="m^3",
        source="rgbd_volume",
    )
    density = scalar(
        30.0,
        lower=20.0,
        upper=50.0,
        confidence=0.6,
        unit="kg/m^3",
        source="material_density_prior",
        status=EstimateStatus.PRIOR,
    )

    mass = estimate_mass_from_volume_density(
        volume,
        density,
        timestamp_seconds=2.0,
    ).mass_kg

    assert mass.value == pytest.approx(6.0)
    assert mass.lower_bound == pytest.approx(2.0)
    assert mass.upper_bound == pytest.approx(15.0)
    assert mass.confidence == pytest.approx(0.6)
    assert mass.source == VOLUME_DENSITY_MASS_SOURCE


def test_unavailable_volume_does_not_become_zero_mass() -> None:
    volume = ScalarEstimate.unavailable(
        unit="m^3",
        source="no_geometry",
        timestamp_seconds=1.0,
    )
    density = scalar(
        30.0,
        unit="kg/m^3",
        source="material_density_prior",
    )

    mass = estimate_mass_from_volume_density(
        volume,
        density,
        timestamp_seconds=2.0,
    ).mass_kg

    assert not mass.is_available


@pytest.mark.parametrize(
    ("volume_unit", "density_unit"),
    [("m", "kg/m^3"), ("m^3", "kg")],
)
def test_volume_density_rejects_wrong_units(
    volume_unit: str,
    density_unit: str,
) -> None:
    with pytest.raises(ValueError, match="must use unit"):
        estimate_mass_from_volume_density(
            scalar(0.2, unit=volume_unit, source="volume"),
            scalar(30.0, unit=density_unit, source="density"),
            timestamp_seconds=2.0,
        )


def test_single_mass_candidate_preserves_its_provenance() -> None:
    candidate = scalar(
        5.0,
        lower=3.0,
        upper=8.0,
        confidence=0.7,
        unit="kg",
        source="image_mass_model",
    )

    result = fuse_mass_estimates(
        [candidate],
        timestamp_seconds=2.0,
    ).mass_kg

    assert result is candidate


def test_mass_fusion_keeps_interval_hull_and_penalizes_disagreement() -> None:
    first = scalar(
        4.0,
        lower=3.0,
        upper=5.0,
        unit="kg",
        source="image_mass_model",
    )
    second = scalar(
        10.0,
        lower=8.0,
        upper=12.0,
        unit="kg",
        source="volume_density",
    )

    result = fuse_mass_estimates(
        [first, second],
        timestamp_seconds=2.0,
    ).mass_kg

    assert result.value == pytest.approx(7.0)
    assert result.lower_bound == pytest.approx(3.0)
    assert result.upper_bound == pytest.approx(12.0)
    assert result.confidence == pytest.approx(0.32)
    assert result.source == MASS_FUSION_SOURCE


def test_no_mass_candidates_returns_unavailable_not_zero() -> None:
    result = fuse_mass_estimates([], timestamp_seconds=2.0).mass_kg

    assert not result.is_available
    assert result.value is None


def test_track_pipeline_falls_back_to_class_prior_for_visible_envelope() -> None:
    density = scalar(
        30.0,
        unit="kg/m^3",
        source="material_density_prior",
        status=EstimateStatus.PRIOR,
    )

    result = estimate_track_mass(
        "dining chair",
        timestamp_seconds=2.0,
        prior_table=prior_table(),
        geometry=geometry(source="lidar_track_visible_envelope"),
        density_kg_per_m3=density,
    ).mass_kg

    assert result.value == pytest.approx(6.0)
    assert result.status is EstimateStatus.PRIOR


def test_track_pipeline_fuses_learned_geometry_and_prior_evidence() -> None:
    learned = scalar(
        7.0,
        lower=4.0,
        upper=11.0,
        confidence=0.7,
        unit="kg",
        source="rgbd_mass_model",
    )
    density = scalar(
        30.0,
        lower=20.0,
        upper=40.0,
        confidence=0.6,
        unit="kg/m^3",
        source="material_density_prior",
        status=EstimateStatus.PRIOR,
    )

    result = estimate_track_mass(
        "dining chair",
        timestamp_seconds=2.0,
        prior_table=prior_table(),
        learned_mass=learned,
        geometry=geometry(),
        density_kg_per_m3=density,
    ).mass_kg

    assert result.is_available
    assert result.source == MASS_FUSION_SOURCE
    assert result.lower_bound == pytest.approx(2.16)
    assert result.upper_bound == pytest.approx(13.2)


def test_geometry_and_density_must_be_supplied_together() -> None:
    with pytest.raises(ValueError, match="supplied together"):
        estimate_track_mass(
            "chair",
            timestamp_seconds=2.0,
            geometry=geometry(),
        )


def test_future_mass_candidate_is_rejected() -> None:
    future = scalar(
        5.0,
        unit="kg",
        source="future_model",
        timestamp=3.0,
    )

    with pytest.raises(ValueError, match="newer"):
        fuse_mass_estimates([future], timestamp_seconds=2.0)


@pytest.mark.parametrize(
    "changes",
    [
        {"accepted_geometry_sources": ()},
        {"minimum_dimension_confidence": -0.1},
        {"minimum_dimension_confidence": 1.1},
        {"maximum_volume_m3": 0.0},
    ],
)
def test_invalid_volume_configuration_is_rejected(
    changes: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        BoxVolumeConfig(**changes)  # type: ignore[arg-type]

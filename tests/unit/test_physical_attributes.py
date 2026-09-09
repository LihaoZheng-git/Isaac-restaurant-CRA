from dataclasses import FrozenInstanceError, replace

import pytest

from robot.perception.physical_attributes import (
    PHYSICAL_ATTRIBUTE_SCHEMA_VERSION,
    CategoricalEstimate,
    ContactAttributes,
    EstimateStatus,
    GeometryAttributes,
    MassAttributes,
    MotionAttributes,
    PhysicalAttributes,
    ScalarEstimate,
    StabilityAttributes,
    ThermalAttributes,
)


def scalar(
    value: float,
    *,
    unit: str = "1",
    lower: float | None = None,
    upper: float | None = None,
    confidence: float = 0.8,
    timestamp: float = 1.0,
    status: EstimateStatus = EstimateStatus.INFERRED,
) -> ScalarEstimate:
    return ScalarEstimate(
        value=value,
        lower_bound=value if lower is None else lower,
        upper_bound=value if upper is None else upper,
        confidence=confidence,
        unit=unit,
        source="unit_test",
        timestamp_seconds=timestamp,
        status=status,
    )


def category(
    value: str,
    *,
    confidence: float = 0.8,
    timestamp: float = 1.0,
) -> CategoricalEstimate:
    return CategoricalEstimate(
        value=value,
        confidence=confidence,
        source="unit_test",
        timestamp_seconds=timestamp,
    )


def unknown_record(*, timestamp: float = 1.0) -> PhysicalAttributes:
    return PhysicalAttributes.unknown(
        track_id=7,
        semantic_class="unknown",
        timestamp_seconds=timestamp,
        evidence=("estimator has not run",),
    )


def test_schema_version_is_explicit() -> None:
    assert PHYSICAL_ATTRIBUTE_SCHEMA_VERSION == 1


def test_scalar_estimate_preserves_interval_and_copies_evidence() -> None:
    evidence = ["LiDAR size and class prior"]
    estimate = ScalarEstimate(
        value=6.0,
        lower_bound=3.5,
        upper_bound=10.0,
        confidence=0.65,
        unit="kg",
        source="hybrid_mass_estimator",
        timestamp_seconds=2.0,
        status=EstimateStatus.INFERRED,
        evidence=evidence,
    )
    evidence.append("later mutation")

    assert estimate.value == pytest.approx(6.0)
    assert estimate.interval_width == pytest.approx(6.5)
    assert estimate.evidence == ("LiDAR size and class prior",)
    assert estimate.is_available
    assert not estimate.is_fallback
    with pytest.raises(FrozenInstanceError):
        estimate.confidence = 0.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("value", float("nan")),
        ("lower_bound", float("-inf")),
        ("upper_bound", True),
        ("confidence", -0.1),
        ("confidence", 1.1),
        ("timestamp_seconds", -0.1),
    ],
)
def test_invalid_scalar_metadata_is_rejected(
    field_name: str,
    value: object,
) -> None:
    values: dict[str, object] = {
        "value": 0.5,
        "lower_bound": 0.2,
        "upper_bound": 0.8,
        "confidence": 0.9,
        "unit": "1",
        "source": "unit_test",
        "timestamp_seconds": 1.0,
    }
    values[field_name] = value

    with pytest.raises((TypeError, ValueError)):
        ScalarEstimate(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("value", "lower", "upper"),
    [
        (0.5, 0.8, 0.2),
        (0.1, 0.2, 0.8),
        (0.9, 0.2, 0.8),
    ],
)
def test_invalid_scalar_intervals_are_rejected(
    value: float,
    lower: float,
    upper: float,
) -> None:
    with pytest.raises(ValueError):
        ScalarEstimate(
            value=value,
            lower_bound=lower,
            upper_bound=upper,
            confidence=0.9,
            unit="1",
            source="unit_test",
            timestamp_seconds=1.0,
        )


def test_available_scalar_requires_value_and_both_bounds() -> None:
    with pytest.raises(ValueError, match="require value and both bounds"):
        ScalarEstimate(
            value=0.5,
            lower_bound=None,
            upper_bound=0.8,
            confidence=0.9,
            unit="1",
            source="unit_test",
            timestamp_seconds=1.0,
        )


def test_unavailable_scalar_is_not_a_measured_zero() -> None:
    estimate = ScalarEstimate.unavailable(
        unit="kg",
        source="mass_model_unavailable",
        timestamp_seconds=1.0,
        evidence=("object crop was occluded",),
    )

    assert estimate.value is None
    assert estimate.lower_bound is None
    assert estimate.upper_bound is None
    assert estimate.interval_width is None
    assert estimate.confidence == pytest.approx(0.0)
    assert not estimate.is_available


def test_unavailable_scalar_rejects_numeric_content() -> None:
    with pytest.raises(ValueError, match="cannot contain numeric values"):
        ScalarEstimate(
            value=0.0,
            lower_bound=None,
            upper_bound=None,
            confidence=0.0,
            unit="kg",
            source="unavailable",
            timestamp_seconds=1.0,
            status=EstimateStatus.UNAVAILABLE,
        )


def test_unavailable_scalar_requires_zero_confidence() -> None:
    with pytest.raises(ValueError, match="zero confidence"):
        ScalarEstimate(
            value=None,
            lower_bound=None,
            upper_bound=None,
            confidence=0.1,
            unit="kg",
            source="unavailable",
            timestamp_seconds=1.0,
            status=EstimateStatus.UNAVAILABLE,
        )


def test_fallback_scalar_is_marked_explicitly() -> None:
    estimate = ScalarEstimate.fallback(
        value=10.0,
        lower_bound=5.0,
        upper_bound=15.0,
        confidence=0.0,
        unit="kg",
        source="conservative_class_fallback",
        timestamp_seconds=1.0,
    )

    assert estimate.status is EstimateStatus.FALLBACK
    assert estimate.is_available
    assert estimate.is_fallback


def test_categorical_estimate_normalizes_identifier() -> None:
    estimate = CategoricalEstimate(
        value="  Hot-Liquid  ",
        confidence=0.7,
        source="thermal_classifier",
        timestamp_seconds=1.0,
        evidence=("open bowl and visible steam",),
    )

    assert estimate.value == "hot_liquid"
    assert estimate.is_available


def test_categorical_unavailable_is_explicit() -> None:
    estimate = CategoricalEstimate.unavailable(
        source="dangerous_part_model_unavailable",
        timestamp_seconds=1.0,
    )

    assert estimate.value is None
    assert estimate.confidence == pytest.approx(0.0)
    assert not estimate.is_available


@pytest.mark.parametrize("mass", [-0.1, -10.0])
def test_mass_attributes_reject_negative_interval(mass: float) -> None:
    with pytest.raises(ValueError, match="negative"):
        MassAttributes(mass_kg=scalar(mass, unit="kg"))


def test_mass_attributes_require_kilograms() -> None:
    with pytest.raises(ValueError, match="'kg'"):
        MassAttributes(mass_kg=scalar(5.0, unit="g"))


def test_geometry_requires_nonnegative_metric_dimensions() -> None:
    with pytest.raises(ValueError, match="height_m cannot contain negative"):
        GeometryAttributes(
            width_m=scalar(0.5, unit="m"),
            depth_m=scalar(0.5, unit="m"),
            height_m=scalar(-0.2, unit="m"),
            center_height_m=scalar(0.5, unit="m"),
            fall_height_m=scalar(0.5, unit="m"),
        )


def test_motion_allows_signed_velocity_but_not_negative_speed() -> None:
    motion = MotionAttributes(
        velocity_x_mps=scalar(-1.0, unit="m/s", lower=-1.1, upper=-0.9),
        velocity_y_mps=scalar(0.2, unit="m/s", lower=0.1, upper=0.3),
        relative_speed_mps=scalar(1.02, unit="m/s"),
        route_crossing_likelihood=scalar(0.8),
    )

    assert motion.velocity_x_mps.value == pytest.approx(-1.0)
    with pytest.raises(ValueError, match="negative"):
        replace(motion, relative_speed_mps=scalar(-0.1, unit="m/s"))


def test_probability_attributes_require_dimensionless_unit_interval() -> None:
    with pytest.raises(ValueError, match=r"bounds must lie in \[0, 1\]"):
        StabilityAttributes(
            instability_likelihood=scalar(1.2, lower=1.1, upper=1.3),
            support_margin_m=scalar(-0.02, unit="m"),
            support_state=category("overhanging"),
        )


def test_negative_support_margin_is_valid_instability_evidence() -> None:
    stability = StabilityAttributes(
        instability_likelihood=scalar(0.9),
        support_margin_m=scalar(-0.02, unit="m", lower=-0.03, upper=-0.01),
        support_state=category("overhanging"),
    )

    assert stability.support_margin_m.value == pytest.approx(-0.02)


def test_contact_and_thermal_attributes_validate_categories() -> None:
    contact = ContactAttributes(
        sharpness=scalar(0.95),
        hardness=scalar(0.8),
        dangerous_part_exposure=scalar(0.9),
        orientation_toward_route=scalar(0.7),
        dangerous_part=category("Knife Blade"),
    )
    thermal = ThermalAttributes(
        heat_source_likelihood=scalar(0.9),
        steam_likelihood=scalar(0.8),
        open_container_likelihood=scalar(0.9),
        carried_likelihood=scalar(0.8),
        spill_likelihood=scalar(0.7),
        source_type=category("Hot Soup"),
    )

    assert contact.dangerous_part.value == "knife_blade"
    assert thermal.source_type.value == "hot_soup"


def test_unknown_record_has_complete_but_unavailable_evidence() -> None:
    record = unknown_record(timestamp=2.0)

    assert record.track_id == 7
    assert record.semantic_class == "unknown"
    assert record.available_estimate_count == 0
    assert len(record.estimates()) == 24
    assert all(not estimate.is_available for estimate in record.estimates())
    assert all(
        estimate.evidence == ("estimator has not run",)
        for estimate in record.estimates()
    )


def test_record_normalizes_semantic_class_and_counts_available_data() -> None:
    base = unknown_record()
    record = replace(
        base,
        semantic_class="  Dining-Chair ",
        semantic_confidence=0.91,
        semantic_source="temporal_yolo_fusion",
        mass=MassAttributes(
            mass_kg=scalar(
                6.0,
                unit="kg",
                lower=3.5,
                upper=10.0,
                status=EstimateStatus.PRIOR,
            )
        ),
    )

    assert record.semantic_class == "dining_chair"
    assert record.available_estimate_count == 1
    with pytest.raises(FrozenInstanceError):
        record.track_id = 8  # type: ignore[misc]


def test_record_rejects_evidence_newer_than_record_timestamp() -> None:
    base = unknown_record(timestamp=1.0)
    future_mass = MassAttributes(
        mass_kg=scalar(6.0, unit="kg", timestamp=1.1)
    )

    with pytest.raises(ValueError, match="newer than the record"):
        replace(base, mass=future_mass)


def test_record_rejects_wrong_attribute_group_type() -> None:
    base = unknown_record()

    with pytest.raises(TypeError, match="mass must be a MassAttributes"):
        replace(base, mass=base.motion)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("track_id", -1),
        ("track_id", True),
        ("semantic_class", ""),
        ("semantic_confidence", 1.1),
        ("timestamp_seconds", float("inf")),
    ],
)
def test_record_rejects_invalid_identity_metadata(
    field_name: str,
    value: object,
) -> None:
    base = unknown_record()

    with pytest.raises((TypeError, ValueError)):
        replace(base, **{field_name: value})

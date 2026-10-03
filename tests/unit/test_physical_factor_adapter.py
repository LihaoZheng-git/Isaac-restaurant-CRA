from dataclasses import replace
import math

import pytest

from robot.perception.physical_attributes import (
    EstimateStatus,
    MassAttributes,
    PhysicalAttributes,
    ScalarEstimate,
)
from robot.risk.cra import CRARiskAssessor
from robot.risk.physical_factor_adapter import (
    PhysicalFactorAdapterConfig,
    adapt_physical_factors,
)


def scalar(
    value,
    unit="1",
    *,
    lower=None,
    upper=None,
    status=EstimateStatus.MEASURED,
    confidence=1.0,
    timestamp=2.0,
):
    return ScalarEstimate(
        value,
        value if lower is None else lower,
        value if upper is None else upper,
        confidence,
        unit,
        "controlled_test",
        timestamp,
        status,
    )


def attributes(mass=10.0, height=2.0, speed=2.0, likelihood=0.5):
    base = PhysicalAttributes.unknown(track_id=7, timestamp_seconds=2.0)
    return replace(
        base,
        mass=MassAttributes(scalar(mass, "kg")),
        geometry=replace(base.geometry, fall_height_m=scalar(height, "m")),
        motion=replace(
            base.motion,
            relative_speed_mps=scalar(speed, "m/s"),
            route_crossing_likelihood=scalar(likelihood),
        ),
    )


def adapt(item=None, **kwargs):
    return adapt_physical_factors(
        attributes() if item is None else item,
        timestamp_seconds=2.0,
        potential_interaction_likelihood=scalar(0.5),
        **kwargs,
    )


def test_energy_equations_and_unknown_specialists():
    result = adapt()
    assert result.potential.energy_j.value == pytest.approx(196.2)
    assert result.kinetic.energy_j.value == 20.0
    assert result.profile.potential.severity == 1.0
    assert result.profile.kinetic.nominal_risk == pytest.approx(0.1)
    assert result.track_id == 7
    for factor in (result.profile.contact, result.profile.thermal):
        assert factor.confidence == 0.0
        assert factor.source == "physical_evidence_cra_adapter"
    # Honest integration: unknown specialists saturate default aggregation.
    assert CRARiskAssessor().assess(result.profile).phi == 1.0


def test_explicit_visual_contact_path_does_not_invent_physical_inputs():
    item = attributes()
    with pytest.raises(ValueError, match="opt-in"):
        adapt(item, visual_contact_proxy=scalar(0.4))
    result = adapt(
        item, visual_contact_proxy=scalar(0.4, confidence=0.8),
        contact_interaction_likelihood=scalar(0.5),
        config=PhysicalFactorAdapterConfig(allow_visual_contact_proxy=True),
    )
    assert result.contact.basis == "visual_exposed_edge_proxy"
    assert result.contact.as_dict()["basis"] == "visual_exposed_edge_proxy"
    assert result.contact.estimate.nominal_risk == pytest.approx(0.2)
    assert not item.contact.sharpness.is_available
    assert not item.contact.hardness.is_available


@pytest.mark.parametrize("timestamp,reason", [(0, "stale"), (3, "future")])
def test_visual_contact_proxy_age_and_missing_interaction_remain_unknown(timestamp, reason):
    result = adapt(
        visual_contact_proxy=scalar(0.4, timestamp=timestamp),
        config=PhysicalFactorAdapterConfig(allow_visual_contact_proxy=True),
    )
    assert result.contact.estimate.confidence == 0
    assert f"visual_exposed_edge_proxy: {reason} evidence" in result.contact.rejection_reasons
    assert "contact_interaction_likelihood: unavailable" in result.contact.rejection_reasons


def test_upper_bounds_and_nominal_option():
    item = attributes()
    item = replace(
        item,
        mass=MassAttributes(scalar(10, "kg", lower=5, upper=20)),
        geometry=replace(item.geometry, fall_height_m=scalar(2, "m", lower=1, upper=3)),
        motion=replace(
            item.motion, relative_speed_mps=scalar(2, "m/s", lower=1, upper=4)
        ),
    )
    result = adapt(
        item,
        config=PhysicalFactorAdapterConfig(
            potential_energy_scale_j=1000, kinetic_energy_scale_j=1000
        ),
    )
    assert result.potential.energy_j.lower_bound == pytest.approx(49.05)
    assert result.potential.energy_j.upper_bound == pytest.approx(588.6)
    assert result.kinetic.energy_j.lower_bound == 2.5
    assert result.kinetic.energy_j.upper_bound == 160
    assert result.profile.kinetic.severity == 0.16
    nominal = adapt(item, config=PhysicalFactorAdapterConfig(use_upper_bounds=False))
    assert nominal.profile.kinetic.severity == 0.2


def test_interaction_upper_bound_selected():
    item = attributes()
    item = replace(
        item,
        motion=replace(
            item.motion, route_crossing_likelihood=scalar(0.5, lower=0.2, upper=0.9)
        ),
    )
    assert adapt(item).profile.kinetic.interaction_likelihood == 0.9


@pytest.mark.parametrize("mass,height,speed", [(0, 2, 2), (10, 0, 0)])
def test_measured_zero_is_not_unknown(mass, height, speed):
    result = adapt(attributes(mass=mass, height=height, speed=speed))
    assert result.profile.potential.severity == 0
    assert result.profile.kinetic.severity == 0
    assert result.profile.potential.confidence == 1


def test_object_and_centroid_height_never_replace_fall_height():
    base = PhysicalAttributes.unknown(track_id=1, timestamp_seconds=2)
    item = replace(
        attributes(),
        geometry=replace(
            base.geometry, height_m=scalar(3, "m"), center_height_m=scalar(2, "m")
        ),
    )
    result = adapt(item)
    assert result.profile.potential.confidence == 0
    assert "fall_height_m: unavailable" in result.potential.rejection_reasons
    assert result.profile.kinetic.confidence == 1


def test_missing_interaction_is_not_inferred_from_route_or_instability():
    item = attributes()
    item = replace(
        item, stability=replace(item.stability, instability_likelihood=scalar(1))
    )
    result = adapt_physical_factors(item, timestamp_seconds=2)
    assert result.profile.potential.confidence == 0
    assert "harmful_fall_interaction: unavailable" in result.potential.rejection_reasons


@pytest.mark.parametrize(
    "status,cap,energy_status",
    [
        (EstimateStatus.MEASURED, 1.0, EstimateStatus.INFERRED),
        (EstimateStatus.PRIOR, 0.5, EstimateStatus.PRIOR),
        (EstimateStatus.FALLBACK, 0.0, EstimateStatus.FALLBACK),
        (EstimateStatus.INFERRED, 1.0, EstimateStatus.INFERRED),
    ],
)
def test_status_caps_and_provenance(status, cap, energy_status):
    item = replace(attributes(), mass=MassAttributes(scalar(10, "kg", status=status)))
    result = adapt(item)
    assert result.profile.kinetic.confidence == cap
    assert result.kinetic.as_dict()["inputs"]["mass_kg"]["status"] == status.value
    assert result.kinetic.energy_j.status is energy_status


def test_priors_can_be_disabled():
    item = replace(
        attributes(), mass=MassAttributes(scalar(10, "kg", status=EstimateStatus.PRIOR))
    )
    result = adapt(item, config=PhysicalFactorAdapterConfig(allow_priors=False))
    assert "mass_kg: priors disabled" in result.kinetic.rejection_reasons


@pytest.mark.parametrize("timestamp,reason", [(0.5, "stale"), (3, "future")])
def test_freshness_is_per_contributing_input(timestamp, reason):
    item = attributes()
    item = replace(
        item,
        timestamp_seconds=max(2, timestamp),
        geometry=replace(
            item.geometry, fall_height_m=scalar(2, "m", timestamp=timestamp)
        ),
    )
    result = adapt(item)
    assert f"fall_height_m: {reason} evidence" in result.potential.rejection_reasons
    assert result.profile.kinetic.confidence == 1


def test_weakest_input_confidence_not_semantic_confidence():
    item = attributes()
    item = replace(
        item,
        semantic_confidence=0.01,
        motion=replace(
            item.motion, relative_speed_mps=scalar(2, "m/s", confidence=0.3)
        ),
    )
    result = adapt(item)
    assert result.profile.kinetic.confidence == 0.3
    assessment = CRARiskAssessor().assess(result.profile)
    kinetic = next(
        item for item in assessment.factors if item.factor.value == "kinetic"
    )
    assert kinetic.conservative_risk >= kinetic.nominal_risk


def test_overflow_fails_unknown():
    result = adapt(attributes(mass=1e308, speed=1e308))
    assert "energy: non-finite arithmetic" in result.kinetic.rejection_reasons
    assert result.profile.kinetic.confidence == 0


@pytest.mark.parametrize(
    "name",
    [
        "potential_energy_scale_j",
        "kinetic_energy_scale_j",
        "gravity_mps2",
        "maximum_evidence_age_seconds",
    ],
)
@pytest.mark.parametrize("value", [0, -1, math.inf, math.nan, True])
def test_invalid_scales(name, value):
    with pytest.raises((TypeError, ValueError)):
        PhysicalFactorAdapterConfig(**{name: value})


@pytest.mark.parametrize("name", ["prior_confidence_cap", "fallback_confidence_cap"])
@pytest.mark.parametrize("value", [-0.1, 1.1, math.nan, True])
def test_invalid_caps(name, value):
    with pytest.raises((TypeError, ValueError)):
        PhysicalFactorAdapterConfig(**{name: value})


def test_invalid_interaction_units_and_bounds():
    for interaction in (scalar(1, "m"), scalar(2)):
        with pytest.raises(ValueError):
            adapt_physical_factors(
                attributes(),
                timestamp_seconds=2,
                potential_interaction_likelihood=interaction,
            )


@pytest.mark.parametrize("mass,speed", [(10, 2), (20, 2), (20, 3)])
def test_monotonic_energy_and_clipped_risk(mass, speed):
    baseline = adapt(attributes(mass=5, speed=1))
    result = adapt(attributes(mass=mass, speed=speed))
    assert result.profile.kinetic.nominal_risk >= baseline.profile.kinetic.nominal_risk
    assert 0 <= result.profile.kinetic.severity <= 1


def test_all_unavailable_remains_unknown():
    result = adapt(PhysicalAttributes.unknown(track_id=1, timestamp_seconds=2))
    assert result.profile.potential.confidence == result.profile.kinetic.confidence == 0
    assert not result.kinetic.energy_j.is_available


def test_complete_contact_and_thermal_specialists_produce_all_four_factors():
    item = attributes()
    item = replace(
        item,
        contact=replace(
            item.contact,
            sharpness=scalar(0.9),
            hardness=scalar(0.8),
            dangerous_part_exposure=scalar(0.7),
            orientation_toward_route=scalar(0.5),
        ),
        thermal=replace(item.thermal, heat_source_likelihood=scalar(0.8)),
    )
    result = adapt_physical_factors(
        item,
        timestamp_seconds=2,
        potential_interaction_likelihood=scalar(0.5),
        thermal_interaction_likelihood=scalar(0.4),
    )
    assert result.profile.contact.severity == pytest.approx(0.87)
    assert result.profile.contact.interaction_likelihood == pytest.approx(0.35)
    assert result.profile.thermal.severity == pytest.approx(0.8)
    assert result.profile.thermal.interaction_likelihood == pytest.approx(0.4)
    assert CRARiskAssessor().assess(result.profile).phi < 1


@pytest.mark.parametrize(
    "status,expected_confidence",
    [
        (EstimateStatus.INFERRED, 1.0),
        (EstimateStatus.PRIOR, 0.5),
        (EstimateStatus.FALLBACK, 0.0),
    ],
)
def test_contact_and_thermal_apply_the_same_status_confidence_caps(
    status, expected_confidence
):
    item = attributes()
    item = replace(
        item,
        contact=replace(
            item.contact,
            sharpness=scalar(0.9, status=status),
            hardness=scalar(0.8),
            dangerous_part_exposure=scalar(0.7),
            orientation_toward_route=scalar(0.5),
        ),
        thermal=replace(
            item.thermal,
            heat_source_likelihood=scalar(0.8, status=status),
        ),
    )

    result = adapt_physical_factors(
        item,
        timestamp_seconds=2,
        potential_interaction_likelihood=scalar(0.5),
        thermal_interaction_likelihood=scalar(0.4),
    )

    assert result.profile.contact.confidence == expected_confidence
    assert result.profile.thermal.confidence == expected_confidence


def test_direct_factor_fallback_cap_is_configurable_but_remains_explicit():
    item = attributes()
    item = replace(
        item,
        thermal=replace(
            item.thermal,
            heat_source_likelihood=scalar(
                0.8, status=EstimateStatus.FALLBACK, confidence=0.9
            ),
        ),
    )

    result = adapt_physical_factors(
        item,
        timestamp_seconds=2,
        potential_interaction_likelihood=scalar(0.5),
        thermal_interaction_likelihood=scalar(0.4),
        config=PhysicalFactorAdapterConfig(fallback_confidence_cap=0.2),
    )

    assert result.profile.thermal.confidence == 0.2
    assert result.thermal.inputs[0][1].status is EstimateStatus.FALLBACK


def test_steam_without_heat_source_does_not_create_thermal_risk():
    item = attributes()
    item = replace(item, thermal=replace(item.thermal, steam_likelihood=scalar(1)))
    result = adapt_physical_factors(
        item, timestamp_seconds=2, thermal_interaction_likelihood=scalar(1)
    )
    assert result.profile.thermal.confidence == 0
    assert "heat_source_likelihood: unavailable" in result.thermal.rejection_reasons


def test_contact_missing_orientation_remains_unknown():
    item = attributes()
    item = replace(
        item,
        contact=replace(
            item.contact,
            sharpness=scalar(1),
            hardness=scalar(1),
            dangerous_part_exposure=scalar(1),
        ),
    )
    assert adapt(item).profile.contact.confidence == 0


def test_mass_contract_output_connects_without_claiming_measurement():
    from robot.perception.mass_estimation import (
        ClassMassPrior,
        ClassMassPriorTable,
        estimate_track_mass,
    )

    table = ClassMassPriorTable(
        {"chair": ClassMassPrior(10, 5, 20, 0.8, "illustrative_chair_prior")}
    )
    mass = estimate_track_mass("chair", timestamp_seconds=2, prior_table=table)
    result = adapt(replace(attributes(), mass=mass))
    assert result.kinetic.energy_j.upper_bound == 40
    assert result.profile.kinetic.confidence == 0.5
    assert result.kinetic.inputs[0][1].status is EstimateStatus.PRIOR


def test_missing_route_score_does_not_become_interaction_one():
    item = attributes()
    unknown = PhysicalAttributes.unknown(track_id=7, timestamp_seconds=2)
    result = adapt(
        replace(
            item,
            motion=replace(
                item.motion,
                route_crossing_likelihood=unknown.motion.route_crossing_likelihood,
            ),
        )
    )
    assert result.profile.kinetic.confidence == 0
    assert "route_crossing_likelihood: unavailable" in result.kinetic.rejection_reasons


def test_configurable_gravity_caps_and_age():
    item = replace(
        attributes(),
        mass=MassAttributes(scalar(10, "kg", status=EstimateStatus.PRIOR, timestamp=0)),
    )
    result = adapt(
        item,
        config=PhysicalFactorAdapterConfig(
            gravity_mps2=10, maximum_evidence_age_seconds=3, prior_confidence_cap=0.2
        ),
    )
    assert result.potential.energy_j.value == 200
    assert result.profile.potential.confidence == 0.2

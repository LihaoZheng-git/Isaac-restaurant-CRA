from dataclasses import FrozenInstanceError
import itertools
import json

import pytest

from robot.risk.cra import (
    CRA_SCHEMA_VERSION,
    CRAAggregation,
    CRAAssessmentConfig,
    CRARiskAssessor,
    PhysicalRiskFactor,
    PhysicalRiskProfile,
    PhysicalRiskVector,
    RiskFactorEstimate,
)


def known_estimate(
    risk: float,
    *,
    confidence: float = 1.0,
    source: str = "oracle",
) -> RiskFactorEstimate:
    return RiskFactorEstimate(
        severity=risk,
        interaction_likelihood=1.0,
        confidence=confidence,
        source=source,
    )


def make_profile(
    **overrides: RiskFactorEstimate,
) -> PhysicalRiskProfile:
    values = {
        factor.value: RiskFactorEstimate.known_absent(source="oracle")
        for factor in PhysicalRiskFactor
    }
    values.update(overrides)
    return PhysicalRiskProfile(**values)


def test_factor_risk_separates_severity_and_interaction_likelihood() -> None:
    estimate = RiskFactorEstimate(
        severity=0.8,
        interaction_likelihood=0.25,
        confidence=0.9,
        source="controlled annotation",
        evidence=("sharp edge is exposed",),
    )

    assert estimate.nominal_risk == pytest.approx(0.2)
    assert estimate.source == "controlled annotation"
    assert estimate.evidence == ("sharp edge is exposed",)


def test_evidence_is_copied_to_an_immutable_tuple() -> None:
    evidence = ["open container"]
    estimate = RiskFactorEstimate(
        severity=0.5,
        interaction_likelihood=0.5,
        confidence=0.5,
        evidence=evidence,
    )
    evidence.append("later mutation")

    assert estimate.evidence == ("open container",)
    with pytest.raises(FrozenInstanceError):
        estimate.confidence = 0.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("severity", -0.1),
        ("severity", 1.1),
        ("severity", float("nan")),
        ("interaction_likelihood", float("inf")),
        ("confidence", True),
    ],
)
def test_invalid_factor_values_are_rejected(
    field_name: str,
    value: object,
) -> None:
    values: dict[str, object] = {
        "severity": 0.5,
        "interaction_likelihood": 0.5,
        "confidence": 0.5,
    }
    values[field_name] = value

    with pytest.raises((TypeError, ValueError)):
        RiskFactorEstimate(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "evidence",
    ["one string is not an evidence sequence", ("",), (3,)],
)
def test_invalid_evidence_is_rejected(evidence: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        RiskFactorEstimate(
            severity=0.5,
            interaction_likelihood=0.5,
            confidence=0.5,
            evidence=evidence,  # type: ignore[arg-type]
        )


def test_known_safe_profile_has_zero_risk_and_no_dominant_factor() -> None:
    assessment = CRARiskAssessor().assess(PhysicalRiskProfile.known_safe())

    assert assessment.nominal_phi == pytest.approx(0.0)
    assert assessment.phi == pytest.approx(0.0)
    assert assessment.uncertainty_uplift == pytest.approx(0.0)
    assert assessment.overall_confidence == pytest.approx(1.0)
    assert assessment.dominant_factor is None


def test_unknown_profile_uses_fail_closed_default_fallback() -> None:
    assessment = CRARiskAssessor().assess(PhysicalRiskProfile.unknown())

    assert assessment.nominal_phi == pytest.approx(0.0)
    assert assessment.phi == pytest.approx(1.0)
    assert assessment.uncertainty_uplift == pytest.approx(1.0)
    assert assessment.overall_confidence == pytest.approx(0.0)
    for factor in PhysicalRiskFactor:
        diagnostic = assessment.factor(factor)
        assert diagnostic.conservative_risk == pytest.approx(1.0)


def test_confidence_blends_upward_toward_factor_fallback() -> None:
    config = CRAAssessmentConfig(
        fallback_risks=PhysicalRiskVector(
            thermal=0.8,
            contact=0.0,
            potential=0.0,
            kinetic=0.0,
        )
    )
    profile = make_profile(
        thermal=known_estimate(0.2, confidence=0.25),
    )

    result = CRARiskAssessor(config).assess(profile)
    thermal = result.factor(PhysicalRiskFactor.THERMAL)

    assert thermal.nominal_risk == pytest.approx(0.2)
    assert thermal.conservative_risk == pytest.approx(0.65)
    assert thermal.uncertainty_uplift == pytest.approx(0.45)
    assert result.phi == pytest.approx(0.65)


def test_low_fallback_never_reduces_a_high_nominal_estimate() -> None:
    config = CRAAssessmentConfig(
        fallback_risks=PhysicalRiskVector(
            thermal=0.3,
            contact=0.0,
            potential=0.0,
            kinetic=0.0,
        )
    )
    result = CRARiskAssessor(config).assess(
        make_profile(thermal=known_estimate(0.9, confidence=0.0))
    )

    assert result.factor(PhysicalRiskFactor.THERMAL).conservative_risk == pytest.approx(
        0.9
    )
    assert result.phi == pytest.approx(0.9)


def test_noisy_or_accumulates_independent_factor_hazards() -> None:
    profile = make_profile(
        thermal=known_estimate(0.5),
        contact=known_estimate(0.5),
    )

    result = CRARiskAssessor().assess(profile)

    assert result.aggregation is CRAAggregation.NOISY_OR
    assert result.phi == pytest.approx(0.75)


@pytest.mark.parametrize(
    ("strategy", "expected_phi"),
    [
        (CRAAggregation.NOISY_OR, 0.75),
        (CRAAggregation.WEIGHTED_SUM, 0.50),
        (CRAAggregation.MAXIMUM, 0.50),
    ],
)
def test_replaceable_aggregation_strategies(
    strategy: CRAAggregation,
    expected_phi: float,
) -> None:
    config = CRAAssessmentConfig(
        aggregation=strategy,
        weights=PhysicalRiskVector(
            thermal=1.0,
            contact=1.0,
            potential=0.0,
            kinetic=0.0,
        ),
        fallback_risks=PhysicalRiskVector.zeros(),
    )
    profile = make_profile(
        thermal=known_estimate(0.5),
        contact=known_estimate(0.5),
    )

    assert CRARiskAssessor(config).assess(profile).phi == pytest.approx(expected_phi)


def test_factor_weight_scales_noisy_or_contribution() -> None:
    config = CRAAssessmentConfig(
        weights=PhysicalRiskVector(
            thermal=0.5,
            contact=0.0,
            potential=0.0,
            kinetic=0.0,
        ),
        fallback_risks=PhysicalRiskVector.zeros(),
    )
    result = CRARiskAssessor(config).assess(make_profile(thermal=known_estimate(0.8)))

    assert result.phi == pytest.approx(0.4)
    assert result.factor(
        PhysicalRiskFactor.THERMAL
    ).weighted_conservative_risk == pytest.approx(0.4)


@pytest.mark.parametrize("factor", tuple(PhysicalRiskFactor))
def test_phi_is_monotone_in_each_factor_severity(
    factor: PhysicalRiskFactor,
) -> None:
    assessor = CRARiskAssessor(
        CRAAssessmentConfig(fallback_risks=PhysicalRiskVector.zeros())
    )
    values = []
    for severity in (0.0, 0.2, 0.5, 0.8, 1.0):
        values.append(
            assessor.assess(
                make_profile(
                    **{
                        factor.value: RiskFactorEstimate(
                            severity=severity,
                            interaction_likelihood=0.7,
                            confidence=1.0,
                        )
                    }
                )
            ).phi
        )

    assert values == sorted(values)


@pytest.mark.parametrize("factor", tuple(PhysicalRiskFactor))
def test_phi_is_monotone_in_interaction_likelihood(
    factor: PhysicalRiskFactor,
) -> None:
    assessor = CRARiskAssessor(
        CRAAssessmentConfig(fallback_risks=PhysicalRiskVector.zeros())
    )
    values = [
        assessor.assess(
            make_profile(
                **{
                    factor.value: RiskFactorEstimate(
                        severity=0.7,
                        interaction_likelihood=likelihood,
                        confidence=1.0,
                    )
                }
            )
        ).phi
        for likelihood in (0.0, 0.2, 0.5, 0.8, 1.0)
    ]

    assert values == sorted(values)


def test_lower_confidence_cannot_reduce_conservative_phi() -> None:
    config = CRAAssessmentConfig(
        fallback_risks=PhysicalRiskVector(
            thermal=0.9,
            contact=0.0,
            potential=0.0,
            kinetic=0.0,
        )
    )
    assessor = CRARiskAssessor(config)
    values = [
        assessor.assess(
            make_profile(thermal=known_estimate(0.2, confidence=confidence))
        ).phi
        for confidence in (1.0, 0.75, 0.5, 0.25, 0.0)
    ]

    assert values == sorted(values)


def test_all_supported_aggregators_remain_bounded() -> None:
    for strategy, values in itertools.product(
        CRAAggregation,
        itertools.product((0.0, 0.25, 0.75, 1.0), repeat=4),
    ):
        profile = PhysicalRiskProfile(
            **{
                factor.value: known_estimate(value)
                for factor, value in zip(
                    PhysicalRiskFactor,
                    values,
                    strict=True,
                )
            }
        )
        result = CRARiskAssessor(
            CRAAssessmentConfig(
                aggregation=strategy,
                fallback_risks=PhysicalRiskVector.zeros(),
            )
        ).assess(profile)
        assert 0.0 <= result.phi <= 1.0


def test_dominant_factor_uses_weighted_conservative_risk() -> None:
    config = CRAAssessmentConfig(
        weights=PhysicalRiskVector(
            thermal=0.2,
            contact=1.0,
            potential=1.0,
            kinetic=1.0,
        ),
        fallback_risks=PhysicalRiskVector.zeros(),
    )
    profile = make_profile(
        thermal=known_estimate(0.9),
        contact=known_estimate(0.4),
    )

    result = CRARiskAssessor(config).assess(profile)

    assert result.dominant_factor is PhysicalRiskFactor.CONTACT


def test_assessment_serialization_is_versioned_and_json_compatible() -> None:
    profile = make_profile(
        contact=RiskFactorEstimate(
            severity=0.8,
            interaction_likelihood=0.5,
            confidence=0.9,
            source="oracle",
            evidence=("blade is exposed",),
        )
    )

    payload = CRARiskAssessor().assess(profile).as_dict()
    encoded = json.dumps(payload, sort_keys=True)

    assert payload["schema_version"] == CRA_SCHEMA_VERSION
    assert payload["dominant_factor"] == "contact"
    assert payload["factors"]["contact"]["evidence"] == ["blade is exposed"]
    assert '"aggregation": "noisy_or"' in encoded


def test_profile_and_factor_lookup_accept_enum_or_stable_string() -> None:
    profile = make_profile(kinetic=known_estimate(0.6))
    result = CRARiskAssessor().assess(profile)

    assert profile.estimate_for("kinetic").nominal_risk == pytest.approx(0.6)
    assert result.factor("kinetic").factor is PhysicalRiskFactor.KINETIC


@pytest.mark.parametrize(
    "values",
    [
        {"thermal": -0.1, "contact": 0.0, "potential": 0.0, "kinetic": 0.0},
        {"thermal": 0.0, "contact": 0.0, "potential": 0.0, "kinetic": 1.1},
    ],
)
def test_invalid_physical_risk_vectors_are_rejected(
    values: dict[str, float],
) -> None:
    with pytest.raises(ValueError):
        PhysicalRiskVector(**values)


def test_zero_weight_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="At least one"):
        CRAAssessmentConfig(weights=PhysicalRiskVector.zeros())


def test_invalid_configuration_and_assessor_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="aggregation"):
        CRAAssessmentConfig(aggregation="unsupported")
    with pytest.raises(TypeError, match="weights"):
        CRAAssessmentConfig(weights={})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="config"):
        CRARiskAssessor(config={})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="profile"):
        CRARiskAssessor().assess({})  # type: ignore[arg-type]


def test_profile_requires_all_four_factor_estimates() -> None:
    with pytest.raises(TypeError, match="thermal"):
        PhysicalRiskProfile(
            thermal=0.0,  # type: ignore[arg-type]
            contact=RiskFactorEstimate.known_absent(),
            potential=RiskFactorEstimate.known_absent(),
            kinetic=RiskFactorEstimate.known_absent(),
        )

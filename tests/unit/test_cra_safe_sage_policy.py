import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from robot.risk.cra import (
    CRAAssessmentConfig,
    CRARiskAssessor,
    PhysicalRiskFactor,
    PhysicalRiskProfile,
    PhysicalRiskVector,
    RiskFactorEstimate,
)
from robot.safety.class_safety_policy import ClassSafetyParameters
from robot.safety.cra_safe_sage_policy import (
    CRASafeSAGEPolicy,
    CRASafeSAGEPolicyConfig,
    CRASafeSAGEPolicyResult,
)
from robot.safety.safe_sage_guidance_policy import (
    PassingSide,
    SafeSAGEGuidanceParameters,
    SafeSAGEGuidancePolicy,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def absent() -> RiskFactorEstimate:
    """Return one known-absent factor."""

    return RiskFactorEstimate.known_absent(source="test")


def factor_estimate(
    risk: float,
    *,
    confidence: float = 1.0,
) -> RiskFactorEstimate:
    """Return an estimate whose nominal risk equals ``risk``."""

    return RiskFactorEstimate(
        severity=risk,
        interaction_likelihood=1.0,
        confidence=confidence,
        source="test",
    )


def single_factor_profile(
    risk: float,
    *,
    factor: PhysicalRiskFactor = PhysicalRiskFactor.THERMAL,
    confidence: float = 1.0,
) -> PhysicalRiskProfile:
    """Return a profile with one active physical factor."""

    estimates = {item.value: absent() for item in PhysicalRiskFactor}
    estimates[factor.value] = factor_estimate(risk, confidence=confidence)
    return PhysicalRiskProfile(**estimates)


def safe_sage_parameters(risk: float) -> SafeSAGEGuidanceParameters:
    """Return deterministic guidance data for result-validation tests."""

    return SafeSAGEGuidanceParameters(
        safety_parameters=ClassSafetyParameters(
            risk=risk,
            boundary_flux_magnitude=0.5 + 1.2 * risk,
        ),
        tangential_strength=1.25,
        passing_side=PassingSide.RIGHT,
    )


def test_default_endpoints_match_repository_paper_baseline() -> None:
    baseline = SafeSAGEGuidancePolicy.from_json(
        PROJECT_ROOT / "configs" / "safe_sage_paper_baseline.json"
    )
    config = CRASafeSAGEPolicyConfig()

    assert config.normal_flux_magnitude_for_risk(0.0) == pytest.approx(
        baseline.parameters_for("object").boundary_flux_magnitude
    )
    assert config.normal_flux_magnitude_for_risk(1.0) == pytest.approx(
        baseline.parameters_for("person").boundary_flux_magnitude
    )
    assert config.tangential_strength_for_risk(0.0) == pytest.approx(
        baseline.parameters_for("object").tangential_strength
    )
    assert config.tangential_strength_for_risk(1.0) == pytest.approx(
        baseline.parameters_for("person").tangential_strength
    )


def test_default_mapping_interpolates_normal_flux_linearly() -> None:
    config = CRASafeSAGEPolicyConfig()

    assert config.normal_flux_magnitude_for_risk(0.0) == pytest.approx(0.5)
    assert config.normal_flux_magnitude_for_risk(0.5) == pytest.approx(1.1)
    assert config.normal_flux_magnitude_for_risk(1.0) == pytest.approx(1.7)


def test_default_mapping_holds_tangential_strength_constant() -> None:
    config = CRASafeSAGEPolicyConfig()

    assert tuple(
        config.tangential_strength_for_risk(risk)
        for risk in (0.0, 0.25, 0.5, 0.75, 1.0)
    ) == pytest.approx((1.25, 1.25, 1.25, 1.25, 1.25))


def test_configurable_tangential_mapping_is_monotone() -> None:
    config = CRASafeSAGEPolicyConfig(
        tangential_strength_at_zero_risk=0.25,
        tangential_strength_at_unit_risk=1.25,
    )

    strengths = tuple(
        config.tangential_strength_for_risk(risk)
        for risk in (0.0, 0.25, 0.5, 0.75, 1.0)
    )

    assert strengths == pytest.approx((0.25, 0.5, 0.75, 1.0, 1.25))
    assert strengths == tuple(sorted(strengths))


def test_passing_side_is_normalized() -> None:
    config = CRASafeSAGEPolicyConfig(passing_side=" LEFT ")

    assert config.passing_side is PassingSide.LEFT


def test_policy_returns_existing_safe_sage_parameter_type() -> None:
    result = CRASafeSAGEPolicy().evaluate(single_factor_profile(0.5))

    assert isinstance(result, CRASafeSAGEPolicyResult)
    assert isinstance(result.guidance_parameters, SafeSAGEGuidanceParameters)
    assert result.risk == pytest.approx(0.5)
    assert result.boundary_flux == pytest.approx(-1.1)
    assert result.tangential_strength == pytest.approx(1.25)
    assert result.passing_side is PassingSide.RIGHT


def test_known_safe_profile_maps_to_zero_risk_endpoint() -> None:
    result = CRASafeSAGEPolicy().evaluate(PhysicalRiskProfile.known_safe())

    assert result.risk == pytest.approx(0.0)
    assert result.boundary_flux == pytest.approx(-0.5)
    assert result.assessment.dominant_factor is None


def test_unknown_profile_maps_to_conservative_unit_endpoint() -> None:
    result = CRASafeSAGEPolicy().evaluate(PhysicalRiskProfile.unknown())

    assert result.risk == pytest.approx(1.0)
    assert result.boundary_flux == pytest.approx(-1.7)
    assert result.assessment.uncertainty_uplift == pytest.approx(1.0)


def test_higher_cra_risk_never_weakens_normal_flux() -> None:
    policy = CRASafeSAGEPolicy()
    flux_magnitudes = tuple(
        policy.evaluate(
            single_factor_profile(risk)
        ).guidance_parameters.boundary_flux_magnitude
        for risk in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    )

    assert flux_magnitudes == tuple(sorted(flux_magnitudes))


def test_lower_confidence_never_weakens_normal_flux() -> None:
    policy = CRASafeSAGEPolicy()
    confident = policy.evaluate(single_factor_profile(0.2, confidence=1.0))
    uncertain = policy.evaluate(single_factor_profile(0.2, confidence=0.25))

    assert uncertain.risk > confident.risk
    assert abs(uncertain.boundary_flux) > abs(confident.boundary_flux)


def test_policy_uses_conservative_phi_instead_of_nominal_phi() -> None:
    assessor = CRARiskAssessor(
        CRAAssessmentConfig(
            fallback_risks=PhysicalRiskVector(
                thermal=1.0,
                contact=0.0,
                potential=0.0,
                kinetic=0.0,
            )
        )
    )
    result = CRASafeSAGEPolicy(risk_assessor=assessor).evaluate(
        single_factor_profile(0.2, confidence=0.5)
    )

    assert result.assessment.nominal_phi == pytest.approx(0.2)
    assert result.risk == pytest.approx(0.6)
    assert result.boundary_flux == pytest.approx(-(0.5 + 1.2 * 0.6))


def test_factor_diagnostics_are_preserved_in_result() -> None:
    result = CRASafeSAGEPolicy().evaluate(
        single_factor_profile(0.8, factor=PhysicalRiskFactor.CONTACT)
    )

    assert result.assessment.dominant_factor is PhysicalRiskFactor.CONTACT
    assert result.assessment.factor(
        PhysicalRiskFactor.CONTACT
    ).conservative_risk == pytest.approx(0.8)


def test_existing_assessment_is_not_recomputed() -> None:
    assessor = CRARiskAssessor()
    assessment = assessor.assess(single_factor_profile(0.4))
    result = CRASafeSAGEPolicy().parameters_for_assessment(assessment)

    assert result.assessment is assessment
    assert result.risk == pytest.approx(0.4)


def test_equivalent_phi_produces_equivalent_safe_sage_parameters() -> None:
    policy = CRASafeSAGEPolicy()
    thermal = policy.evaluate(
        single_factor_profile(0.4, factor=PhysicalRiskFactor.THERMAL)
    )
    contact = policy.evaluate(
        single_factor_profile(0.4, factor=PhysicalRiskFactor.CONTACT)
    )

    assert thermal.risk == pytest.approx(contact.risk)
    assert thermal.boundary_flux == pytest.approx(contact.boundary_flux)
    assert thermal.tangential_strength == pytest.approx(contact.tangential_strength)


def test_result_is_json_serializable_and_explainable() -> None:
    result = CRASafeSAGEPolicy().evaluate(
        single_factor_profile(0.7, factor=PhysicalRiskFactor.POTENTIAL)
    )
    payload = result.as_dict()
    encoded = json.dumps(payload, sort_keys=True)

    assert payload["cra"]["dominant_factor"] == "potential"
    assert payload["safe_sage"]["risk"] == pytest.approx(0.7)
    assert payload["safe_sage"]["boundary_flux"] == pytest.approx(-1.34)
    assert payload["safe_sage"]["passing_side"] == "right"
    assert '"tangential_strength": 1.25' in encoded


def test_policy_configuration_and_result_are_frozen() -> None:
    config = CRASafeSAGEPolicyConfig()
    result = CRASafeSAGEPolicy(config=config).evaluate(single_factor_profile(0.3))

    with pytest.raises(FrozenInstanceError):
        config.passing_side = PassingSide.LEFT
    with pytest.raises(FrozenInstanceError):
        result.assessment = result.assessment


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"normal_flux_magnitude_at_zero_risk": 0.0},
            "must be positive",
        ),
        (
            {
                "normal_flux_magnitude_at_zero_risk": 1.0,
                "normal_flux_magnitude_at_unit_risk": 0.9,
            },
            "cannot decrease",
        ),
        (
            {"tangential_strength_at_zero_risk": -0.1},
            "cannot be negative",
        ),
        (
            {
                "tangential_strength_at_zero_risk": 1.0,
                "tangential_strength_at_unit_risk": 0.9,
            },
            "cannot decrease",
        ),
        (
            {"normal_flux_magnitude_at_unit_risk": float("inf")},
            "must be finite",
        ),
        (
            {"tangential_strength_at_unit_risk": float("nan")},
            "must be finite",
        ),
    ],
)
def test_invalid_configured_endpoints_are_rejected(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        CRASafeSAGEPolicyConfig(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"normal_flux_magnitude_at_zero_risk": True},
        {"normal_flux_magnitude_at_unit_risk": "large"},
        {"tangential_strength_at_zero_risk": None},
        {"tangential_strength_at_unit_risk": False},
    ],
)
def test_non_numeric_configured_endpoints_are_rejected(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(TypeError, match="real number"):
        CRASafeSAGEPolicyConfig(**overrides)


@pytest.mark.parametrize("side", ["clockwise", "", 4, None])
def test_invalid_passing_side_is_rejected(side: object) -> None:
    with pytest.raises((TypeError, ValueError), match="passing_side"):
        CRASafeSAGEPolicyConfig(passing_side=side)


@pytest.mark.parametrize("risk", [-0.1, 1.1, float("inf"), float("nan")])
def test_invalid_interpolation_risk_is_rejected(risk: float) -> None:
    config = CRASafeSAGEPolicyConfig()

    with pytest.raises(ValueError):
        config.normal_flux_magnitude_for_risk(risk)
    with pytest.raises(ValueError):
        config.tangential_strength_for_risk(risk)


@pytest.mark.parametrize("risk", [True, "high", None])
def test_non_numeric_interpolation_risk_is_rejected(risk: object) -> None:
    config = CRASafeSAGEPolicyConfig()

    with pytest.raises(TypeError, match="real number"):
        config.normal_flux_magnitude_for_risk(risk)
    with pytest.raises(TypeError, match="real number"):
        config.tangential_strength_for_risk(risk)


def test_invalid_policy_dependencies_are_rejected() -> None:
    with pytest.raises(TypeError, match="risk_assessor"):
        CRASafeSAGEPolicy(risk_assessor=0)
    with pytest.raises(TypeError, match="config"):
        CRASafeSAGEPolicy(config={})


def test_invalid_policy_inputs_are_rejected() -> None:
    policy = CRASafeSAGEPolicy()

    with pytest.raises(TypeError, match="profile"):
        policy.evaluate("hot soup")
    with pytest.raises(TypeError, match="assessment"):
        policy.parameters_for_assessment(0.8)


def test_result_rejects_invalid_members_and_inconsistent_risk() -> None:
    assessment = CRARiskAssessor().assess(single_factor_profile(0.4))

    with pytest.raises(TypeError, match="assessment"):
        CRASafeSAGEPolicyResult(
            assessment="risk",
            guidance_parameters=safe_sage_parameters(0.4),
        )
    with pytest.raises(TypeError, match="guidance_parameters"):
        CRASafeSAGEPolicyResult(
            assessment=assessment,
            guidance_parameters="parameters",
        )
    with pytest.raises(ValueError, match="must equal"):
        CRASafeSAGEPolicyResult(
            assessment=assessment,
            guidance_parameters=safe_sage_parameters(0.5),
        )

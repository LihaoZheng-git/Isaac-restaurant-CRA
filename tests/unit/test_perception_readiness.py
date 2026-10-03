from dataclasses import replace

import pytest

from robot.evaluation.cra_experiments import EvidenceMode, RestaurantScenario
from robot.evaluation.perception_readiness import (
    EvaluationScope,
    evaluate_perception_readiness,
)
from robot.perception.physical_attributes import EstimateStatus, ScalarEstimate
from robot.risk.cra import PhysicalRiskProfile, RiskFactorEstimate
from robot.risk.physical_factor_adapter import (
    PhysicalFactorAdapterConfig,
    PhysicalFactorAdapterResult,
    PhysicalFactorDiagnostic,
)


def scalar(status=EstimateStatus.INFERRED):
    if status is EstimateStatus.UNAVAILABLE:
        return ScalarEstimate.unavailable(
            unit="1", source="fixture", timestamp_seconds=1.0
        )
    return ScalarEstimate(
        0.5,
        0.4,
        0.6,
        0.8,
        "1",
        "fixture",
        1.0,
        status,
    )


def diagnostic(name, status=EstimateStatus.INFERRED):
    item = scalar(status)
    accepted = status is not EstimateStatus.UNAVAILABLE
    estimate = (
        RiskFactorEstimate(0.5, 0.5, 0.8, source="fixture")
        if accepted
        else RiskFactorEstimate.unknown(source="fixture")
    )
    energy = ScalarEstimate.unavailable(
        unit="J", source="fixture", timestamp_seconds=1.0
    )
    reasons = () if accepted else ("cue: unavailable",)
    return PhysicalFactorDiagnostic(
        name, (("cue", item),), energy, estimate, reasons
    )


def adapter_result(status=EstimateStatus.INFERRED):
    diagnostics = {
        name: diagnostic(name, status)
        for name in ("thermal", "contact", "potential", "kinetic")
    }
    return PhysicalFactorAdapterResult(
        track_id=1,
        timestamp_seconds=1.0,
        config=PhysicalFactorAdapterConfig(),
        profile=PhysicalRiskProfile(
            diagnostics["thermal"].estimate,
            diagnostics["contact"].estimate,
            diagnostics["potential"].estimate,
            diagnostics["kinetic"].estimate,
        ),
        potential=diagnostics["potential"],
        kinetic=diagnostics["kinetic"],
        contact=diagnostics["contact"],
        thermal=diagnostics["thermal"],
    )


def scenario(mode=EvidenceMode.SENSOR, family="thermal"):
    return RestaurantScenario("fixture", family, "fixture", mode, 0)


def test_contact_proxy_readiness_is_explicit_and_provenance_is_still_required():
    result = adapter_result()
    result = replace(
        result, contact=replace(result.contact, basis="visual_exposed_edge_proxy"),
        config=PhysicalFactorAdapterConfig(allow_visual_contact_proxy=True),
    )
    readiness = evaluate_perception_readiness(scenario(family="contact"), result,
                                             scope=EvaluationScope.FACTOR_SPECIFIC)
    assert readiness.ready
    assert readiness.factors[0].basis == "visual_exposed_edge_proxy"
    prior = replace(result, contact=replace(diagnostic("contact", EstimateStatus.PRIOR),
                                          basis="visual_exposed_edge_proxy"))
    assert not evaluate_perception_readiness(scenario(family="contact"), prior,
                                             scope=EvaluationScope.FACTOR_SPECIFIC).ready
    with pytest.raises(ValueError, match="opt-in"):
        evaluate_perception_readiness(scenario(), replace(result, config=PhysicalFactorAdapterConfig()))


def test_sensor_four_factor_readiness_requires_all_inferred_or_measured():
    result = evaluate_perception_readiness(scenario(), adapter_result())
    assert result.ready
    assert result.aggregate_phi_reportable
    assert not result.rejected_factors


def test_sensor_mode_rejects_prior_even_when_factor_is_numeric():
    result = evaluate_perception_readiness(
        scenario(), adapter_result(EstimateStatus.PRIOR)
    )
    assert not result.ready
    assert set(result.rejected_factors) == {
        "thermal",
        "contact",
        "potential",
        "kinetic",
    }
    assert "not valid for evidence_mode=sensor" in result.factors[0].reasons[0]


def test_prior_and_oracle_modes_remain_distinct():
    prior = evaluate_perception_readiness(
        scenario(EvidenceMode.PRIOR), adapter_result(EstimateStatus.PRIOR)
    )
    oracle = evaluate_perception_readiness(
        scenario(EvidenceMode.ORACLE), adapter_result(EstimateStatus.ORACLE)
    )
    wrong_oracle = evaluate_perception_readiness(
        scenario(EvidenceMode.ORACLE), adapter_result(EstimateStatus.INFERRED)
    )
    assert prior.ready
    assert oracle.ready
    assert not wrong_oracle.ready


def test_factor_specific_acceptance_never_authorizes_aggregate_phi():
    result = adapter_result()
    result = replace(
        result,
        contact=diagnostic("contact", EstimateStatus.UNAVAILABLE),
        potential=diagnostic("potential", EstimateStatus.UNAVAILABLE),
        kinetic=diagnostic("kinetic", EstimateStatus.UNAVAILABLE),
    )
    readiness = evaluate_perception_readiness(
        scenario(family="thermal"),
        result,
        scope=EvaluationScope.FACTOR_SPECIFIC,
    )
    assert readiness.ready
    assert not readiness.aggregate_phi_reportable
    assert tuple(item.factor for item in readiness.factors) == ("thermal",)


def test_fallback_and_unavailable_inputs_are_rejected():
    fallback = evaluate_perception_readiness(
        scenario(EvidenceMode.PRIOR), adapter_result(EstimateStatus.FALLBACK)
    )
    unavailable = evaluate_perception_readiness(
        scenario(), adapter_result(EstimateStatus.UNAVAILABLE)
    )
    assert not fallback.ready
    assert not unavailable.ready


def test_unknown_hazard_family_rejected():
    with pytest.raises(ValueError, match="hazard_family"):
        evaluate_perception_readiness(
            scenario(family="unknown"), adapter_result()
        )

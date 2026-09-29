from dataclasses import replace
import json

import pytest

from robot.evaluation.factor_specific_cra import diagnose_factor_specific_cra
from robot.perception.physical_attributes import (
    EstimateStatus,
    PhysicalAttributes,
    ScalarEstimate,
)
from robot.risk.cra import PhysicalRiskFactor
from robot.risk.physical_factor_adapter import adapt_physical_factors


def cue(value, *, source="fixture", status=EstimateStatus.INFERRED):
    return ScalarEstimate(
        value,
        max(0.0, value - 0.1),
        min(1.0, value + 0.1),
        0.8,
        "1",
        source,
        2.0,
        status,
    )


def thermal_only_result():
    base = PhysicalAttributes.unknown(track_id=4, timestamp_seconds=2.0)
    attributes = replace(
        base,
        thermal=replace(base.thermal, heat_source_likelihood=cue(0.7)),
    )
    return adapt_physical_factors(
        attributes,
        timestamp_seconds=2.0,
        thermal_interaction_likelihood=cue(0.4, source="route_fixture"),
    )


def test_unknown_factors_are_reported_instead_of_zeroed_for_thermal_evaluation():
    report = diagnose_factor_specific_cra(thermal_only_result())
    thermal = report.factor(PhysicalRiskFactor.THERMAL)
    assert thermal.available
    assert thermal.nominal_risk == pytest.approx(0.4)
    assert not thermal.aggregate_phi_reportable
    assert set(report.unknown_fallback_factors) == {
        PhysicalRiskFactor.CONTACT,
        PhysicalRiskFactor.POTENTIAL,
        PhysicalRiskFactor.KINETIC,
    }
    assert report.aggregate_saturated
    assert report.saturation_involves_unknown_fallback
    for factor in report.unknown_fallback_factors:
        item = report.factor(factor)
        assert not item.available
        assert item.rejection_reasons
        assert item.conservative_risk > item.nominal_risk


def test_report_preserves_factor_input_provenance_and_scope(tmp_path):
    report = diagnose_factor_specific_cra(thermal_only_result())
    thermal = report.factor(PhysicalRiskFactor.THERMAL)
    assert {item.status for item in thermal.inputs} == {"inferred"}
    assert {item.source for item in thermal.inputs} == {
        "fixture",
        "route_fixture",
    }
    path = tmp_path / "factor-report.json"
    report.write_json(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["aggregate"]["saturation_involves_unknown_fallback"]
    assert not payload["factor_specific"]["thermal"]["aggregate_phi_reportable"]
    assert "remain unknown" in payload["scope_note"]


def test_saturation_threshold_and_input_validation():
    result = thermal_only_result()
    report = diagnose_factor_specific_cra(result, saturation_threshold=1.0)
    assert report.aggregate_saturated
    with pytest.raises(ValueError, match="saturation_threshold"):
        diagnose_factor_specific_cra(result, saturation_threshold=0)
    with pytest.raises(TypeError, match="PhysicalFactorAdapterResult"):
        diagnose_factor_specific_cra(object())

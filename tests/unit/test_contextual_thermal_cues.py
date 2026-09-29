import pytest

from robot.perception.contextual_thermal_cues import (
    CONTEXTUAL_THERMAL_SOURCE,
    estimate_contextual_thermal_cues,
)
from robot.perception.physical_attributes import EstimateStatus, ScalarEstimate


def cue(value: float, *, timestamp: float = 2.0) -> ScalarEstimate:
    return ScalarEstimate(
        value=value,
        lower_bound=max(0.0, value - 0.05),
        upper_bound=min(1.0, value + 0.05),
        confidence=0.8,
        unit="1",
        source="fixture",
        timestamp_seconds=timestamp,
        status=EstimateStatus.INFERRED,
    )


def unavailable() -> ScalarEstimate:
    return ScalarEstimate.unavailable(
        unit="1", source="fixture_unavailable", timestamp_seconds=2.0
    )


def estimate(plume: float, context: float, association: float):
    return estimate_contextual_thermal_cues(
        plume_likelihood=cue(plume),
        source_context_likelihood=cue(context),
        plume_source_association_likelihood=cue(association),
        timestamp_seconds=2.1,
    )


def test_heat_evidence_is_continuous_and_monotonic():
    weak = estimate(0.2, 0.9, 0.8)
    medium = estimate(0.5, 0.9, 0.8)
    strong = estimate(0.8, 0.9, 0.8)

    assert weak.thermal.heat_source_likelihood.value == pytest.approx(0.2)
    assert medium.thermal.heat_source_likelihood.value == pytest.approx(0.5)
    assert strong.thermal.heat_source_likelihood.value == pytest.approx(0.8)
    assert weak.thermal.heat_source_likelihood.source == CONTEXTUAL_THERMAL_SOURCE


def test_general_source_context_does_not_require_a_soup_classification():
    result = estimate_contextual_thermal_cues(
        plume_likelihood=cue(0.7),
        source_context_likelihood=cue(0.8),
        plume_source_association_likelihood=cue(0.6),
        timestamp_seconds=2.1,
        plume_label="smoke_like",
    )

    assert result.context_available
    assert result.thermal.heat_source_likelihood.value == pytest.approx(0.6)
    assert not result.thermal.open_container_likelihood.is_available
    assert not result.thermal.steam_likelihood.is_available
    assert any(
        "no hot/not-hot" in item
        for item in result.thermal.heat_source_likelihood.evidence
    )


def test_missing_required_visual_evidence_remains_unavailable():
    result = estimate_contextual_thermal_cues(
        plume_likelihood=unavailable(),
        source_context_likelihood=cue(0.8),
        plume_source_association_likelihood=cue(0.7),
        timestamp_seconds=2.1,
    )

    assert not result.context_available
    assert not result.thermal.heat_source_likelihood.is_available
    assert result.rejection_reasons == ("plume_likelihood: unavailable",)

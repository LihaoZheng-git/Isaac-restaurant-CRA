import pytest

from robot.perception.hot_soup_cues import (
    HOT_SOUP_SOURCE,
    HotSoupCueConfig,
    estimate_hot_soup_cues,
)
from robot.perception.physical_attributes import EstimateStatus, ScalarEstimate


def cue(value=0.8, *, timestamp=2.0, source="fixture"):
    return ScalarEstimate(
        value=value,
        lower_bound=max(0, value - 0.1),
        upper_bound=min(1, value + 0.1),
        confidence=0.8,
        unit="1",
        source=source,
        timestamp_seconds=timestamp,
        status=EstimateStatus.INFERRED,
    )


def unavailable(*, timestamp=2.0):
    return ScalarEstimate.unavailable(
        unit="1", source="unavailable", timestamp_seconds=timestamp
    )


def test_restaurant_wrapper_produces_continuous_contextual_thermal_proxy():
    result = estimate_hot_soup_cues(
        plume_likelihood=cue(0.9),
        open_container_likelihood=cue(0.8),
        plume_above_container_likelihood=cue(0.85),
        timestamp_seconds=2.1,
        plume_label="steam",
    )
    assert result.accepted_context
    assert result.thermal.heat_source_likelihood.value == pytest.approx(0.8)
    assert result.thermal.heat_source_likelihood.source == HOT_SOUP_SOURCE
    assert result.thermal.source_type.value == "contextual_visible_plume_proxy"
    assert result.thermal.steam_likelihood.value == pytest.approx(0.9)
    assert any(
        "not a temperature" in item
        for item in result.thermal.heat_source_likelihood.evidence
    )


def test_generic_smoke_like_plume_alone_stays_unavailable():
    result = estimate_hot_soup_cues(
        plume_likelihood=cue(0.9),
        open_container_likelihood=unavailable(),
        plume_above_container_likelihood=unavailable(),
        timestamp_seconds=2.1,
        plume_label="smoke_like",
    )
    assert not result.accepted_context
    assert not result.thermal.heat_source_likelihood.is_available
    assert not result.thermal.steam_likelihood.is_available
    assert "source_context_likelihood: unavailable" in result.rejection_reasons


def test_smoke_like_context_can_support_proxy_but_is_not_relabeled_as_steam():
    result = estimate_hot_soup_cues(
        plume_likelihood=cue(0.9),
        open_container_likelihood=cue(0.8),
        plume_above_container_likelihood=cue(0.85),
        timestamp_seconds=2.1,
        plume_label="smoke_like",
    )
    assert result.thermal.heat_source_likelihood.is_available
    assert not result.thermal.steam_likelihood.is_available
    assert any(
        "does not prove steam" in item
        for item in result.thermal.heat_source_likelihood.evidence
    )


def test_stale_required_evidence_rejects_context():
    result = estimate_hot_soup_cues(
        plume_likelihood=cue(timestamp=0),
        open_container_likelihood=cue(),
        plume_above_container_likelihood=cue(),
        timestamp_seconds=2.1,
    )
    assert not result.thermal.heat_source_likelihood.is_available
    assert result.rejection_reasons == ("plume_likelihood: stale evidence",)


def test_prior_dependency_remains_prior_in_derived_heat_proxy():
    original = cue(0.8)
    prior = ScalarEstimate(
        original.value,
        original.lower_bound,
        original.upper_bound,
        original.confidence,
        original.unit,
        original.source,
        original.timestamp_seconds,
        EstimateStatus.PRIOR,
    )
    result = estimate_hot_soup_cues(
        plume_likelihood=cue(0.9),
        open_container_likelihood=prior,
        plume_above_container_likelihood=cue(0.85),
        timestamp_seconds=2.1,
    )
    assert result.thermal.heat_source_likelihood.status is EstimateStatus.PRIOR
    assert result.thermal.source_type.status is EstimateStatus.PRIOR


def test_weak_supported_context_keeps_numeric_score_and_descriptive_source():
    result = estimate_hot_soup_cues(
        plume_likelihood=cue(0.2),
        open_container_likelihood=cue(0.9),
        plume_above_container_likelihood=cue(0.8),
        timestamp_seconds=2.1,
    )
    assert result.thermal.heat_source_likelihood.value == pytest.approx(0.2)
    assert result.thermal.source_type.value == "contextual_visible_plume_proxy"
    assert "not a risk threshold" in result.thermal.source_type.evidence[-1]


@pytest.mark.parametrize("label", ["fire", "", "hot"])
def test_unsupported_plume_labels_rejected(label):
    with pytest.raises(ValueError, match="plume_label"):
        estimate_hot_soup_cues(
            plume_likelihood=cue(),
            open_container_likelihood=cue(),
            plume_above_container_likelihood=cue(),
            timestamp_seconds=2.1,
            plume_label=label,
        )


def test_invalid_configuration_rejected():
    with pytest.raises(ValueError):
        HotSoupCueConfig(maximum_evidence_age_seconds=0)

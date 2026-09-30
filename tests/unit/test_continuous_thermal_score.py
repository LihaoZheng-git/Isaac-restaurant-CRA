import pytest

from robot.perception.continuous_thermal_score import (
    ContinuousThermalScoreConfig,
    estimate_continuous_thermal_score,
)
from robot.perception.physical_attributes import EstimateStatus, ScalarEstimate
from robot.perception.plume_features import ContinuousPlumeFeatures


def cue(
    value,
    *,
    confidence=0.8,
    status=EstimateStatus.INFERRED,
    bounded=True,
):
    return ScalarEstimate(
        value,
        max(0, value - 0.05),
        min(1, value + 0.05) if bounded else value + 0.05,
        confidence,
        "1",
        "fixture",
        2.0,
        status,
    )


def features(
    area,
    extent,
    persistence,
    *,
    context=0.9,
    association=0.9,
    model_confidence=0.8,
):
    return ContinuousPlumeFeatures(
        relative_mask_area=cue(area, bounded=False),
        vertical_extent=cue(extent, bounded=False),
        persistence=cue(persistence),
        source_context_likelihood=cue(context),
        spatial_association_likelihood=cue(association),
        segmentation_confidence=cue(model_confidence, confidence=model_confidence),
        invoked_frame_count=4,
        positive_frame_count=4,
        rejection_reasons=(),
    )


def test_stronger_more_persistent_plume_increases_continuous_score():
    faint = estimate_continuous_thermal_score(
        features(0.1, 0.3, 0.25), timestamp_seconds=2.1
    )
    dense = estimate_continuous_thermal_score(
        features(0.9, 1.8, 1.0), timestamp_seconds=2.1
    )
    assert faint.plume_strength.value < dense.plume_strength.value
    assert faint.thermal.heat_source_likelihood.value < (
        dense.thermal.heat_source_likelihood.value
    )
    assert dense.thermal.source_type.value == "continuous_visual_plume_proxy"


def test_weak_source_association_limits_thermal_evidence():
    result = estimate_continuous_thermal_score(
        features(0.9, 1.8, 1.0, association=0.2),
        timestamp_seconds=2.1,
    )
    assert result.plume_strength.value > 0.8
    assert result.thermal.heat_source_likelihood.value == pytest.approx(0.2)


def test_model_confidence_changes_confidence_not_score_magnitude():
    high = estimate_continuous_thermal_score(
        features(0.5, 1.0, 0.75, model_confidence=0.9),
        timestamp_seconds=2.1,
    )
    low = estimate_continuous_thermal_score(
        features(0.5, 1.0, 0.75, model_confidence=0.3),
        timestamp_seconds=2.1,
    )
    assert high.plume_strength.value == low.plume_strength.value
    assert high.thermal.heat_source_likelihood.value == (
        low.thermal.heat_source_likelihood.value
    )
    assert low.thermal.heat_source_likelihood.confidence == pytest.approx(0.3)


def test_missing_persistence_keeps_thermal_evidence_unavailable():
    item = features(0.5, 1.0, 0.75)
    missing = ScalarEstimate.unavailable(
        unit="1",
        source="fixture",
        timestamp_seconds=2.0,
    )
    item = ContinuousPlumeFeatures(
        item.relative_mask_area,
        item.vertical_extent,
        missing,
        item.source_context_likelihood,
        item.spatial_association_likelihood,
        item.segmentation_confidence,
        3,
        2,
        ("unvalidated negative",),
    )
    result = estimate_continuous_thermal_score(item, timestamp_seconds=2.1)
    assert not result.thermal.heat_source_likelihood.is_available
    assert result.rejection_reasons == ("persistence: unavailable",)


def test_generic_plume_does_not_populate_steam_temperature_or_container_fields():
    result = estimate_continuous_thermal_score(
        features(0.5, 1.0, 0.75), timestamp_seconds=2.1
    )
    assert not result.thermal.steam_likelihood.is_available
    assert not result.thermal.open_container_likelihood.is_available
    assert not result.thermal.carried_likelihood.is_available
    assert not result.thermal.spill_likelihood.is_available
    assert any(
        "not a temperature" in item
        for item in result.thermal.heat_source_likelihood.evidence
    )


def test_oracle_feature_dependency_remains_oracle():
    item = features(0.5, 1.0, 0.75)
    oracle = cue(0.75, status=EstimateStatus.ORACLE)
    item = ContinuousPlumeFeatures(
        item.relative_mask_area,
        item.vertical_extent,
        oracle,
        item.source_context_likelihood,
        item.spatial_association_likelihood,
        item.segmentation_confidence,
        3,
        3,
        (),
    )
    result = estimate_continuous_thermal_score(item, timestamp_seconds=2.1)
    assert result.thermal.heat_source_likelihood.status is EstimateStatus.ORACLE


def test_weights_must_be_nonnegative_and_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        ContinuousThermalScoreConfig(
            relative_area_weight=0.5,
            vertical_extent_weight=0.5,
            persistence_weight=0.5,
        )


def test_stale_features_do_not_receive_a_fresh_output_timestamp():
    result = estimate_continuous_thermal_score(
        features(0.5, 1.0, 0.75),
        timestamp_seconds=3.0,
        config=ContinuousThermalScoreConfig(maximum_evidence_age_seconds=0.5),
    )
    assert not result.thermal.heat_source_likelihood.is_available
    assert all("stale evidence" in reason for reason in result.rejection_reasons)

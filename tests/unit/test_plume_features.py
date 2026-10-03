import numpy as np
import pytest
from dataclasses import replace

from robot.perception.physical_attributes import EstimateStatus, ScalarEstimate
from robot.perception.plume_features import (
    PlumeFeatureConfig,
    PlumeSourceObservation,
    extract_continuous_plume_features,
)
from robot.perception.plume_segmentation import (
    PlumeSegmentationFrame,
    VisualPlumeInstance,
)
from robot.perception.semantic_detection import PixelBoundingBox


def cue(value, timestamp, status=EstimateStatus.INFERRED):
    return ScalarEstimate(
        value,
        max(0, value - 0.05),
        min(1, value + 0.05),
        0.8,
        "1",
        "fixture",
        timestamp,
        status,
    )


def segmentation(frame_id, timestamp, mask=None, status=EstimateStatus.INFERRED):
    instances = ()
    if mask is not None:
        ys, xs = np.nonzero(mask)
        instances = (
            VisualPlumeInstance(
                "steam",
                "steam",
                0.8,
                PixelBoundingBox(xs.min(), ys.min(), xs.max() + 1, ys.max() + 1),
                mask,
                0,
            ),
        )
    return PlumeSegmentationFrame(
        frame_id,
        timestamp,
        20,
        20,
        instances,
        "fixture-segmenter",
        0.01,
        status,
    )


def observation(
    frame_id,
    timestamp,
    mask,
    *,
    negative_validated=False,
    context_timestamp=None,
    association_timestamp=None,
):
    return PlumeSourceObservation(
        segmentation(frame_id, timestamp, mask),
        PixelBoundingBox(5, 10, 15, 20),
        cue(0.8, timestamp if context_timestamp is None else context_timestamp),
        cue(
            0.7,
            timestamp if association_timestamp is None else association_timestamp,
        ),
        0 if mask is not None else None,
        negative_validated,
    )


def plume(height, width=6):
    mask = np.zeros((20, 20), dtype=bool)
    mask[10 - height : 10, 10 - width // 2 : 10 + width // 2] = True
    return mask


def test_validated_empty_quality_enables_low_visible_cue_without_claiming_cold():
    items = tuple(replace(observation(i, 1+i*.1, None, negative_validated=True),
                          negative_observation_confidence=.55) for i in range(3))
    features = extract_continuous_plume_features(items, timestamp_seconds=1.2)
    assert features.relative_mask_area.value == 0
    assert features.vertical_extent.value == 0
    assert features.persistence.value == 0
    assert features.segmentation_confidence.value == .55
    assert features.persistence.confidence == .55
    assert "no cold-object claim" in features.relative_mask_area.evidence[0]


def test_negative_confidence_cannot_appear_without_domain_validation():
    with pytest.raises(ValueError, match="domain validation"):
        replace(observation(0, 1, None), negative_observation_confidence=.55)


def test_larger_and_taller_masks_produce_larger_descriptive_features():
    small = extract_continuous_plume_features(
        tuple(observation(i, 1 + i * 0.1, plume(2, 4)) for i in range(3)),
        timestamp_seconds=1.3,
    )
    large = extract_continuous_plume_features(
        tuple(observation(i, 1 + i * 0.1, plume(8, 8)) for i in range(3)),
        timestamp_seconds=1.3,
    )
    assert small.relative_mask_area.value < large.relative_mask_area.value
    assert small.vertical_extent.value < large.vertical_extent.value
    assert large.persistence.value == pytest.approx(1.0)
    assert large.source_context_likelihood.value == pytest.approx(0.8)
    assert large.spatial_association_likelihood.value == pytest.approx(0.7)


def test_validated_empty_masks_make_intermittent_plume_less_persistent():
    persistent = tuple(
        observation(i, 1 + i * 0.1, plume(4), negative_validated=True)
        for i in range(4)
    )
    intermittent = tuple(
        observation(
            i,
            1 + i * 0.1,
            plume(4) if i in (0, 2) else None,
            negative_validated=True,
        )
        for i in range(4)
    )
    strong = extract_continuous_plume_features(persistent, timestamp_seconds=1.4)
    weak = extract_continuous_plume_features(intermittent, timestamp_seconds=1.4)
    assert strong.persistence.value == 1
    assert weak.persistence.value == 0.5


def test_unvalidated_empty_masks_do_not_become_zero_evidence():
    items = (
        observation(0, 1.0, plume(4)),
        observation(1, 1.1, None),
        observation(2, 1.2, plume(4)),
    )
    actual = extract_continuous_plume_features(items, timestamp_seconds=1.3)
    assert not actual.persistence.is_available
    assert "empty masks are not validated negative evidence" in actual.rejection_reasons


def test_model_confidence_is_kept_separate_from_feature_magnitude():
    actual = extract_continuous_plume_features(
        tuple(observation(i, 1 + i * 0.1, plume(4)) for i in range(3)),
        timestamp_seconds=1.3,
    )
    assert actual.segmentation_confidence.value == pytest.approx(0.8)
    assert actual.relative_mask_area.value != actual.segmentation_confidence.value
    assert "not plume strength" in actual.segmentation_confidence.evidence[0]


def test_oracle_masks_preserve_oracle_provenance():
    items = []
    for i in range(3):
        timestamp = 1 + i * 0.1
        items.append(
            PlumeSourceObservation(
                segmentation(i, timestamp, plume(4), EstimateStatus.ORACLE),
                PixelBoundingBox(5, 10, 15, 20),
                cue(0.8, timestamp, EstimateStatus.ORACLE),
                cue(0.7, timestamp, EstimateStatus.ORACLE),
                0,
            )
        )
    actual = extract_continuous_plume_features(
        tuple(items), timestamp_seconds=1.3
    )
    assert actual.relative_mask_area.status is EstimateStatus.ORACLE
    assert actual.persistence.status is EstimateStatus.ORACLE


def test_stale_frames_are_rejected_together():
    actual = extract_continuous_plume_features(
        (observation(0, 0.0, plume(4)),),
        timestamp_seconds=2.0,
        config=PlumeFeatureConfig(maximum_evidence_age_seconds=0.5),
    )
    assert not actual.relative_mask_area.is_available
    assert actual.rejection_reasons == ("stale segmentation frame",)


@pytest.mark.parametrize(
    "timestamp_argument,feature,reason",
    [
        ({"context_timestamp": 0.0}, "source_context_likelihood", "stale cue"),
        (
            {"association_timestamp": 2.2},
            "spatial_association_likelihood",
            "future cue",
        ),
    ],
)
def test_fresh_masks_cannot_refresh_stale_or_future_context_cues(
    timestamp_argument, feature, reason
):
    observations = tuple(
        observation(
            i,
            1.8 + i * 0.1,
            plume(4),
            **timestamp_argument,
        )
        for i in range(3)
    )

    actual = extract_continuous_plume_features(
        observations,
        timestamp_seconds=2.0,
        config=PlumeFeatureConfig(maximum_evidence_age_seconds=0.5),
    )

    assert actual.relative_mask_area.is_available
    assert not getattr(actual, feature).is_available
    assert f"{feature}: {reason}" in actual.rejection_reasons

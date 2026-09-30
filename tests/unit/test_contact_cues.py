import numpy as np
import pytest

from robot.perception.contact_cues import (
    DangerousPartSegmentation,
    estimate_contact_cues,
)
from robot.perception.physical_attributes import EstimateStatus, ScalarEstimate


def segmentation(confidence=0.8, timestamp=2):
    object_mask = np.ones((4, 4), dtype=np.bool_)
    danger = np.zeros((4, 4), dtype=np.bool_)
    danger[:2] = True
    return DangerousPartSegmentation(
        object_mask, danger, "knife_blade", confidence, timestamp, "fixture"
    )


def cue(value, source="specialist"):
    return ScalarEstimate(
        value,
        max(0, value - 0.1),
        min(1, value + 0.1),
        0.7,
        "1",
        source,
        2,
        EstimateStatus.INFERRED,
    )


def test_mask_exposure_is_separate_from_sharpness_material_and_orientation():
    result = estimate_contact_cues(segmentation(), timestamp_seconds=2)
    assert result.dangerous_part_exposure.value == 0.5
    assert result.dangerous_part.value == "knife_blade"
    assert not result.sharpness.is_available
    assert not result.hardness.is_available
    assert not result.orientation_toward_route.is_available


def test_supplied_specialists_retain_provenance():
    result = estimate_contact_cues(
        segmentation(),
        sharpness=cue(0.9, "sharpness_net"),
        hardness=cue(0.8, "material_net"),
        orientation_toward_route=cue(0.4, "calibrated_geometry"),
        timestamp_seconds=2,
    )
    assert result.sharpness.source == "sharpness_net"
    assert result.hardness.source == "material_net"


def test_missing_segmentation_is_fully_explicit():
    result = estimate_contact_cues(None, timestamp_seconds=2)
    assert all(
        not item.is_available
        for item in (
            result.sharpness,
            result.hardness,
            result.dangerous_part_exposure,
            result.orientation_toward_route,
            result.dangerous_part,
        )
    )


def test_part_outside_object_is_rejected():
    item = segmentation()
    object_mask = item.object_mask.copy()
    object_mask[0, 0] = False
    with pytest.raises(ValueError, match="inside object"):
        estimate_contact_cues(
            DangerousPartSegmentation(
                object_mask, item.dangerous_part_mask, "blade", 0.8, 2, "fixture"
            ),
            timestamp_seconds=2,
        )


@pytest.mark.parametrize("confidence", [-1, 2, float("nan")])
def test_bad_confidence_is_rejected(confidence):
    with pytest.raises(ValueError):
        estimate_contact_cues(segmentation(confidence), timestamp_seconds=2)


def test_pixel_ratio_bounds_widen_when_confidence_is_low():
    high = estimate_contact_cues(segmentation(1), timestamp_seconds=2)
    low = estimate_contact_cues(segmentation(0.5), timestamp_seconds=2)
    assert high.dangerous_part_exposure.interval_width == 0
    assert low.dangerous_part_exposure.interval_width > 0


def test_contact_mask_keeps_source_timestamp_instead_of_becoming_fresh():
    result = estimate_contact_cues(
        segmentation(timestamp=1), timestamp_seconds=2
    )

    assert result.dangerous_part_exposure.timestamp_seconds == 1
    assert result.dangerous_part.timestamp_seconds == 1

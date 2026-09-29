import numpy as np
import pytest

from robot.perception.semantic_detection import (
    PixelBoundingBox,
    SemanticDetection,
    SemanticDetectionFrame,
)
from robot.perception.thermal_detection_adapter import (
    thermal_cue_frame_from_detections,
)


def detection(name, confidence, class_id):
    mask = np.ones((4, 5), dtype=np.bool_)
    return SemanticDetection(
        class_id, name, confidence, PixelBoundingBox(0, 0, 5, 4), mask
    )


def frame(*detections):
    return SemanticDetectionFrame(1, 2.0, 4, 5, detections, "d_fire_candidate", 0.01)


def test_fire_and_smoke_detector_does_not_relabel_smoke_as_steam():
    result = thermal_cue_frame_from_detections(
        frame(detection("fire", 0.7, 0), detection("smoke", 0.9, 1))
    )
    assert result.flame_likelihood == 0.7
    assert result.smoke_likelihood == 0.9
    assert result.steam_likelihood is None


def test_no_fire_detection_remains_unavailable():
    result = thermal_cue_frame_from_detections(frame(detection("smoke", 0.8, 1)))
    assert result.flame_likelihood is None
    assert result.smoke_likelihood == 0.8
    assert result.steam_likelihood is None


def test_empty_detection_frame_does_not_fabricate_negative_evidence():
    result = thermal_cue_frame_from_detections(frame())
    assert result.flame_likelihood is None
    assert result.smoke_likelihood is None
    assert result.steam_likelihood is None


def test_overlapping_class_mapping_is_rejected():
    with pytest.raises(ValueError, match="overlap"):
        thermal_cue_frame_from_detections(
            frame(), flame_classes=("fire",), smoke_classes=("fire",)
        )

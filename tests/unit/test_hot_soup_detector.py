from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from robot.perception.hot_soup_cues import HotSoupCueConfig
from robot.perception.hot_soup_detector import (
    HOT_SOUP_DETECTOR_SOURCE,
    HotSoupDetection,
    HotSoupDetectionFrame,
    HotSoupDetectorConfig,
    YoloHotSoupDetector,
    estimate_hot_soup_from_detections,
)
from robot.perception.physical_attributes import EstimateStatus
from robot.perception.semantic_detection import PixelBoundingBox


def detection(class_id, class_name, confidence, box):
    return HotSoupDetection(
        class_id,
        class_name,
        confidence,
        PixelBoundingBox(*box),
    )


def frame(*detections, timestamp=2.0):
    return HotSoupDetectionFrame(
        source_frame_id=7,
        source_timestamp_seconds=timestamp,
        image_height=100,
        image_width=100,
        detections=detections,
        model_name="project-hot-soup.pt",
        inference_wall_seconds=0.01,
    )


def test_plume_above_container_produces_contextual_visual_proxy():
    result = estimate_hot_soup_from_detections(
        frame(
            detection(0, "visible_hot_soup_plume", 0.9, (40, 10, 60, 45)),
            detection(1, "open_container", 0.8, (30, 45, 70, 80)),
        ),
        timestamp_seconds=2.1,
    )
    thermal = result.cue_result.thermal
    assert result.spatial_association_score == pytest.approx(1.0)
    assert thermal.heat_source_likelihood.value == pytest.approx(0.8)
    assert thermal.heat_source_likelihood.source != HOT_SOUP_DETECTOR_SOURCE
    assert thermal.source_type.value == "contextual_visible_plume_proxy"
    assert not thermal.steam_likelihood.is_available


def test_oracle_box_fixture_preserves_oracle_provenance():
    item = frame(
        detection(0, "visible_hot_soup_plume", 0.9, (40, 10, 60, 45)),
        detection(1, "open_container", 0.8, (30, 45, 70, 80)),
    )
    item = replace(item, evidence_status=EstimateStatus.ORACLE)
    result = estimate_hot_soup_from_detections(item, timestamp_seconds=2.1)
    assert result.cue_result.thermal.heat_source_likelihood.status is (
        EstimateStatus.ORACLE
    )


def test_missing_container_remains_unavailable_instead_of_becoming_zero():
    result = estimate_hot_soup_from_detections(
        frame(detection(0, "visible_hot_soup_plume", 0.9, (40, 10, 60, 45))),
        timestamp_seconds=2.1,
    )
    assert result.candidate_pair_count == 0
    assert not result.cue_result.thermal.heat_source_likelihood.is_available
    assert "source_context_likelihood: unavailable" in (
        result.cue_result.rejection_reasons
    )


def test_observed_bad_geometry_is_zero_relation_not_missing_evidence():
    result = estimate_hot_soup_from_detections(
        frame(
            detection(0, "visible_hot_soup_plume", 0.9, (0, 70, 20, 90)),
            detection(1, "open_container", 0.8, (60, 10, 90, 40)),
        ),
        timestamp_seconds=2.1,
    )
    assert result.candidate_pair_count == 1
    assert result.spatial_association_score == 0
    assert result.cue_result.accepted_context
    assert result.cue_result.thermal.heat_source_likelihood.value == 0
    assert (
        result.cue_result.thermal.source_type.value
        == "contextual_visible_plume_proxy"
    )
    assert "not a risk threshold" in (
        result.cue_result.thermal.source_type.evidence[-1]
    )


def test_partial_horizontal_overlap_retains_continuous_evidence():
    weak = estimate_hot_soup_from_detections(
        frame(
            detection(0, "visible_hot_soup_plume", 0.9, (68, 10, 88, 45)),
            detection(1, "open_container", 0.8, (30, 45, 70, 80)),
        ),
        timestamp_seconds=2.1,
    )
    stronger = estimate_hot_soup_from_detections(
        frame(
            detection(0, "visible_hot_soup_plume", 0.9, (66, 10, 86, 45)),
            detection(1, "open_container", 0.8, (30, 45, 70, 80)),
        ),
        timestamp_seconds=2.1,
    )

    assert weak.spatial_association_score == pytest.approx(0.4)
    assert stronger.spatial_association_score == pytest.approx(0.8)
    assert weak.cue_result.thermal.heat_source_likelihood.value == pytest.approx(0.4)
    assert stronger.cue_result.thermal.heat_source_likelihood.value == pytest.approx(
        0.8
    )


def test_vertical_relation_changes_smoothly_with_plume_position():
    near = estimate_hot_soup_from_detections(
        frame(
            detection(0, "visible_hot_soup_plume", 0.9, (40, 35, 60, 65)),
            detection(1, "open_container", 0.8, (30, 45, 70, 80)),
        ),
        timestamp_seconds=2.1,
    )
    clearly_above = estimate_hot_soup_from_detections(
        frame(
            detection(0, "visible_hot_soup_plume", 0.9, (40, 20, 60, 50)),
            detection(1, "open_container", 0.8, (30, 45, 70, 80)),
        ),
        timestamp_seconds=2.1,
    )

    assert 0 < near.spatial_association_score < clearly_above.spatial_association_score
    assert clearly_above.spatial_association_score == pytest.approx(1.0)


def test_stale_detection_frame_is_rejected_by_existing_cue_contract():
    result = estimate_hot_soup_from_detections(
        frame(
            detection(0, "visible_hot_soup_plume", 0.9, (40, 10, 60, 45)),
            detection(1, "open_container", 0.8, (30, 45, 70, 80)),
            timestamp=0,
        ),
        timestamp_seconds=2.1,
        cue_config=HotSoupCueConfig(maximum_evidence_age_seconds=1),
    )
    assert not result.cue_result.accepted_context
    assert all("stale evidence" in reason for reason in result.cue_result.rejection_reasons)


class FakeBackend:
    def __init__(self, result):
        self.result = result
        self.kwargs = None

    def predict(self, **kwargs):
        self.kwargs = kwargs
        return [self.result]


def test_yolo_detection_backend_converts_boxes_without_fake_masks():
    raw = SimpleNamespace(
        names={0: "visible_hot_soup_plume", 1: "open_container", 2: "person"},
        boxes=SimpleNamespace(
            xyxy=np.array([[4, 1, 8, 5], [2, 5, 10, 9], [0, 0, 2, 4]]),
            conf=np.array([0.9, 0.8, 0.7]),
            cls=np.array([0.0, 1.0, 2.0]),
        ),
    )
    backend = FakeBackend(raw)
    detector = YoloHotSoupDetector(
        HotSoupDetectorConfig("project-hot-soup.pt", device="cpu"),
        model_backend=backend,
    )
    actual = detector.detect(
        np.zeros((10, 12, 3), dtype=np.uint8),
        source_frame_id=3,
        source_timestamp_seconds=4.0,
    )
    assert [item.semantic_class for item in actual.detections] == [
        "visible_hot_soup_plume",
        "open_container",
    ]
    assert backend.kwargs["save"] is False
    assert not hasattr(actual.detections[0], "mask")


@pytest.mark.parametrize(
    "class_name", ["steam", "smoke", "soup", "person"]
)
def test_detection_contract_rejects_classes_outside_fixed_vocabulary(class_name):
    with pytest.raises(ValueError, match="unsupported"):
        detection(0, class_name, 0.8, (1, 1, 2, 2))

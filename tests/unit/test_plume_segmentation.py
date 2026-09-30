import numpy as np
import pytest

from robot.perception.physical_attributes import EstimateStatus
from robot.perception.plume_segmentation import (
    PlumeClassMap,
    adapt_plume_segmentations,
)
from robot.perception.semantic_detection import (
    PixelBoundingBox,
    SemanticDetection,
    SemanticDetectionFrame,
)


def detection(name, confidence, mask):
    ys, xs = np.nonzero(mask)
    return SemanticDetection(
        0,
        name,
        confidence,
        PixelBoundingBox(xs.min(), ys.min(), xs.max() + 1, ys.max() + 1),
        mask,
    )


def frame(*detections):
    return SemanticDetectionFrame(
        source_frame_id=4,
        source_timestamp_seconds=1.5,
        image_height=10,
        image_width=12,
        detections=detections,
        model_name="candidate-segmentation",
        inference_wall_seconds=0.02,
    )


def test_explicit_mapping_preserves_steam_and_smoke_distinction():
    steam = np.zeros((10, 12), dtype=bool)
    steam[1:5, 4:8] = True
    smoke = np.zeros((10, 12), dtype=bool)
    smoke[2:7, 1:3] = True
    person = np.zeros((10, 12), dtype=bool)
    person[4:9, 8:11] = True
    actual = adapt_plume_segmentations(
        frame(
            detection("steam", 0.8, steam),
            detection("smoke", 0.7, smoke),
            detection("person", 0.9, person),
        ),
        class_map=PlumeClassMap(
            {"steam": "steam", "smoke": "smoke_like"}
        ),
        checkpoint_sha256="a" * 64,
        model_source="https://example.invalid/model-card",
    )
    assert [item.plume_label for item in actual.instances] == [
        "steam",
        "smoke_like",
    ]
    assert actual.instances[1].source_class == "smoke"
    assert actual.checkpoint_sha256 == "a" * 64
    assert actual.evidence_status is EstimateStatus.INFERRED
    assert not actual.instances[0].mask.flags.writeable


def test_empty_invoked_frame_is_distinct_from_missing_evidence():
    actual = adapt_plume_segmentations(
        frame(),
        class_map=PlumeClassMap({"steam": "steam"}),
    )
    assert actual.observed_plume_count == 0
    assert actual.instances == ()


def test_generic_smoke_cannot_be_silently_mapped_to_steam():
    mapping = PlumeClassMap({"smoke": "smoke_like"})
    assert mapping.source_to_plume_label["smoke"] == "smoke_like"
    with pytest.raises(ValueError, match="plume label"):
        PlumeClassMap({"smoke": "hot_soup"})


def test_unavailable_is_not_a_valid_invoked_frame_status():
    with pytest.raises(ValueError, match="cannot be unavailable"):
        adapt_plume_segmentations(
            frame(),
            class_map=PlumeClassMap({"steam": "steam"}),
            evidence_status=EstimateStatus.UNAVAILABLE,
        )

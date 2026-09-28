import numpy as np
import pytest

from robot.perception.contact_detection_adapter import (
    ContactPartAssociationConfig,
    associate_contact_parts_to_objects,
)
from robot.perception.semantic_detection import (
    PixelBoundingBox,
    SemanticDetection,
    SemanticDetectionFrame,
)


def detection(name, mask, confidence=0.8):
    ys, xs = np.nonzero(mask)
    return SemanticDetection(
        0,
        name,
        confidence,
        PixelBoundingBox(xs.min(), ys.min(), xs.max() + 1, ys.max() + 1),
        mask,
    )


def frame(*detections, frame_id=3, timestamp=2.0, model="fixture"):
    return SemanticDetectionFrame(
        frame_id,
        timestamp,
        20,
        20,
        detections,
        model,
        0.0,
    )


def mask(y0, y1, x0, x1):
    result = np.zeros((20, 20), dtype=bool)
    result[y0:y1, x0:x1] = True
    return result


def test_supported_edge_is_attached_to_parent_object_mask():
    objects = frame(detection("knife", mask(4, 16, 4, 16)), model="objects.pt")
    parts = frame(
        detection("exposed_edge", mask(5, 7, 5, 15), 0.7),
        model="paco-edge.pt",
    )

    result = associate_contact_parts_to_objects(objects, parts)
    item = result.segmentations_by_object_detection_index[0]
    assert item.dangerous_part_label == "exposed_edge"
    assert item.confidence == pytest.approx(0.7)
    assert item.dangerous_part_mask.sum() == 20
    assert item.object_mask.sum() == 144
    assert result.diagnostics.accepted_part_count == 1


def test_unsupported_pointed_tip_is_reported_not_promoted():
    objects = frame(detection("tool", mask(4, 16, 4, 16)))
    parts = frame(detection("pointed_tip", mask(5, 7, 5, 15)))

    result = associate_contact_parts_to_objects(objects, parts)
    assert result.segmentations_by_object_detection_index == {}
    assert result.diagnostics.unsupported_part_indices == (0,)


def test_missing_detection_does_not_create_zero_exposure():
    objects = frame(detection("knife", mask(4, 16, 4, 16)))
    result = associate_contact_parts_to_objects(objects, frame())
    assert result.segmentations_by_object_detection_index == {}
    assert result.diagnostics.part_detection_count == 0


def test_part_with_two_equally_good_parents_is_ambiguous():
    parent = detection("tool", mask(4, 16, 4, 16))
    parts = frame(detection("exposed_edge", mask(5, 7, 5, 15)))
    result = associate_contact_parts_to_objects(frame(parent, parent), parts)
    assert result.segmentations_by_object_detection_index == {}
    assert result.diagnostics.ambiguous_part_indices == (0,)


def test_part_must_be_mostly_contained_by_parent():
    objects = frame(detection("tool", mask(4, 8, 4, 8)))
    parts = frame(detection("exposed_edge", mask(0, 10, 0, 10)))
    result = associate_contact_parts_to_objects(objects, parts)
    assert result.diagnostics.unmatched_part_indices == (0,)


def test_frames_must_represent_same_camera_capture():
    objects = frame(detection("tool", mask(4, 8, 4, 8)))
    with pytest.raises(ValueError, match="frame IDs"):
        associate_contact_parts_to_objects(objects, frame(frame_id=4))
    with pytest.raises(ValueError, match="timestamps"):
        associate_contact_parts_to_objects(objects, frame(timestamp=2.1))


def test_only_taxonomy_classes_can_be_configured():
    with pytest.raises(ValueError, match="taxonomy"):
        ContactPartAssociationConfig(supported_geometry_classes=("sharp",))

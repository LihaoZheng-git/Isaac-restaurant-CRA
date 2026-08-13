"""Unit tests for project-owned semantic perception values."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from robot.perception.semantic_detection import (
    PixelBoundingBox,
    SemanticDetection,
    SemanticDetectionFrame,
    normalize_detection_class,
)
from robot.perception.visualization import (
    render_semantic_detection_overlay,
)


def make_detection(
    *,
    semantic_class: str = "Dining Chair",
    confidence: float = 0.9,
    image_shape: tuple[int, int] = (6, 8),
) -> SemanticDetection:
    """Return one small valid instance for focused tests."""

    mask = np.zeros(image_shape, dtype=np.bool_)
    mask[2:5, 3:7] = True
    return SemanticDetection(
        class_id=56,
        semantic_class=semantic_class,
        confidence=confidence,
        bounding_box=PixelBoundingBox(3.0, 2.0, 7.0, 5.0),
        mask=mask,
    )


def make_frame(
    *detections: SemanticDetection,
) -> SemanticDetectionFrame:
    """Return a frame matching :func:`make_detection`."""

    return SemanticDetectionFrame(
        source_frame_id=7,
        source_timestamp_seconds=1.25,
        image_height=6,
        image_width=8,
        detections=detections,
        model_name="yolo11n-seg.pt",
        inference_wall_seconds=0.012,
    )


def test_normalize_detection_class_produces_stable_identifier() -> None:
    assert normalize_detection_class("  Dining-Chair  ") == "dining_chair"


@pytest.mark.parametrize("value", ["", "   "])
def test_normalize_detection_class_rejects_empty_name(value: str) -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        normalize_detection_class(value)


def test_pixel_bounding_box_reports_geometry() -> None:
    box = PixelBoundingBox(1, 2, 6, 10)

    assert box.width == pytest.approx(5.0)
    assert box.height == pytest.approx(8.0)
    assert box.area == pytest.approx(40.0)
    assert box.center_xy == pytest.approx((3.5, 6.0))


@pytest.mark.parametrize(
    "coordinates",
    [
        (-1.0, 0.0, 2.0, 2.0),
        (0.0, -1.0, 2.0, 2.0),
        (2.0, 0.0, 2.0, 2.0),
        (3.0, 0.0, 2.0, 2.0),
        (0.0, 0.0, float("nan"), 2.0),
    ],
)
def test_pixel_bounding_box_rejects_invalid_geometry(
    coordinates: tuple[float, float, float, float],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        PixelBoundingBox(*coordinates)


def test_semantic_detection_normalizes_and_owns_immutable_mask() -> None:
    source_mask = np.zeros((6, 8), dtype=np.bool_)
    source_mask[2:5, 3:7] = True
    detection = SemanticDetection(
        class_id=56,
        semantic_class="Dining-Chair",
        confidence=0.875,
        bounding_box=PixelBoundingBox(3, 2, 7, 5),
        mask=source_mask,
    )

    source_mask[:, :] = False

    assert detection.semantic_class == "dining_chair"
    assert detection.mask_pixel_count == 12
    assert not detection.mask.flags.writeable
    with pytest.raises(ValueError):
        detection.mask[2, 3] = False
    with pytest.raises(FrozenInstanceError):
        detection.confidence = 0.1  # type: ignore[misc]


@pytest.mark.parametrize("confidence", [-0.01, 1.01, float("inf")])
def test_semantic_detection_rejects_invalid_confidence(
    confidence: float,
) -> None:
    with pytest.raises(ValueError):
        make_detection(confidence=confidence)


def test_semantic_detection_requires_boolean_two_dimensional_mask() -> None:
    box = PixelBoundingBox(0, 0, 2, 2)
    with pytest.raises(TypeError, match="boolean"):
        SemanticDetection(0, "chair", 0.9, box, np.ones((2, 2)))
    with pytest.raises(ValueError, match="height, width"):
        SemanticDetection(
            0,
            "chair",
            0.9,
            box,
            np.ones((1, 2, 2), dtype=np.bool_),
        )


def test_detection_frame_validates_and_queries_instances() -> None:
    chair = make_detection()
    person = make_detection(semantic_class="person", confidence=0.8)
    frame = make_frame(chair, person, chair)

    assert isinstance(frame.detections, tuple)
    assert frame.class_counts == {"dining_chair": 2, "person": 1}
    assert frame.detections_for("Dining-Chair") == (chair, chair)
    with pytest.raises(TypeError):
        frame.class_counts["person"] = 4  # type: ignore[index]


def test_detection_frame_rejects_mask_shape_mismatch() -> None:
    detection = make_detection(image_shape=(5, 8))

    with pytest.raises(ValueError, match="mask must match"):
        make_frame(detection)


def test_detection_frame_rejects_box_outside_source_image() -> None:
    mask = np.zeros((6, 8), dtype=np.bool_)
    detection = SemanticDetection(
        class_id=0,
        semantic_class="person",
        confidence=0.9,
        bounding_box=PixelBoundingBox(1, 1, 9, 5),
        mask=mask,
    )

    with pytest.raises(ValueError, match="inside the source image"):
        make_frame(detection)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_frame_id", -1),
        ("source_timestamp_seconds", -0.1),
        ("image_height", 0),
        ("image_width", 0),
        ("inference_wall_seconds", -0.1),
        ("model_name", " "),
    ],
)
def test_detection_frame_rejects_invalid_metadata(
    field: str,
    value: object,
) -> None:
    arguments: dict[str, object] = {
        "source_frame_id": 0,
        "source_timestamp_seconds": 0.0,
        "image_height": 6,
        "image_width": 8,
        "detections": (),
        "model_name": "model.pt",
        "inference_wall_seconds": 0.0,
    }
    arguments[field] = value

    with pytest.raises((TypeError, ValueError)):
        SemanticDetectionFrame(**arguments)  # type: ignore[arg-type]


def test_overlay_renders_mask_box_and_label_without_mutating_input() -> None:
    image = np.full((6, 8, 3), 40, dtype=np.uint8)
    original = image.copy()
    frame = make_frame(make_detection())

    overlay = render_semantic_detection_overlay(
        image,
        frame,
        mask_opacity=0.5,
        line_width=1,
    )

    assert overlay.shape == image.shape
    assert overlay.dtype == np.uint8
    assert np.array_equal(image, original)
    assert not np.array_equal(overlay, original)


def test_overlay_rejects_frame_image_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        render_semantic_detection_overlay(
            np.zeros((7, 8, 3), dtype=np.uint8),
            make_frame(),
        )

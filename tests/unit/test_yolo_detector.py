"""Unit tests for the optional Ultralytics segmentation adapter."""

from types import SimpleNamespace

import numpy as np
import pytest

from robot.perception.yolo_detector import (
    YoloSegmentationConfig,
    YoloSegmentationDetector,
    rgb_to_bgr_uint8,
)


class FakeModel:
    """Record the adapter call and return controlled model output."""

    def __init__(self, results: object) -> None:
        self.results = results
        self.calls: list[dict[str, object]] = []

    def predict(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self.results


def make_result(
    *,
    image_shape: tuple[int, int] = (4, 5),
) -> SimpleNamespace:
    """Return two instance-segmentation outputs."""

    height, width = image_shape
    masks = np.zeros((2, height, width), dtype=np.float32)
    masks[0, 0:3, 0:2] = 0.8
    masks[1, 1:4, 2:5] = 0.9
    boxes = SimpleNamespace(
        xyxy=np.asarray(
            [
                [-1.0, 0.0, 2.0, 3.0],
                [2.0, 1.0, 8.0, 4.0],
            ],
            dtype=np.float32,
        ),
        conf=np.asarray([0.91, 0.82], dtype=np.float32),
        cls=np.asarray([0.0, 56.0], dtype=np.float32),
    )
    return SimpleNamespace(
        boxes=boxes,
        masks=SimpleNamespace(data=masks),
        names={0: "Person", 56: "Dining-Chair"},
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("confidence_threshold", -0.1),
        ("confidence_threshold", 1.1),
        ("iou_threshold", -0.1),
        ("iou_threshold", 1.1),
        ("image_size", 0),
        ("maximum_detections", 0),
        ("device", ""),
        ("model_path", ""),
    ],
)
def test_configuration_rejects_invalid_values(
    field: str,
    value: object,
) -> None:
    arguments: dict[str, object] = {field: value}
    with pytest.raises((TypeError, ValueError)):
        YoloSegmentationConfig(**arguments)  # type: ignore[arg-type]


def test_configuration_normalizes_allowed_classes() -> None:
    config = YoloSegmentationConfig(
        allowed_classes=("Dining-Chair", " Person "),
    )

    assert config.allowed_classes == ("dining_chair", "person")


def test_configuration_rejects_string_and_duplicate_allowlists() -> None:
    with pytest.raises(TypeError, match="iterable of class names"):
        YoloSegmentationConfig(
            allowed_classes="person",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="duplicates"):
        YoloSegmentationConfig(
            allowed_classes=("Dining-Chair", "dining chair"),
        )


def test_rgb_to_bgr_uint8_converts_channels_without_mutation() -> None:
    image = np.asarray([[[1, 2, 3, 255], [4, 5, 6, 255]]], dtype=np.uint8)
    original = image.copy()

    converted = rgb_to_bgr_uint8(image)

    assert np.array_equal(converted, [[[3, 2, 1], [6, 5, 4]]])
    assert converted.dtype == np.uint8
    assert converted.flags.c_contiguous
    assert np.array_equal(image, original)


def test_rgb_to_bgr_uint8_scales_normalized_float_image() -> None:
    converted = rgb_to_bgr_uint8(
        np.asarray([[[0.0, 0.5, 1.0]]], dtype=np.float32)
    )

    assert np.array_equal(converted, [[[255, 127, 0]]])


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((2, 3), dtype=np.uint8),
        np.zeros((2, 3, 2), dtype=np.uint8),
        np.asarray([[[float("nan"), 0.0, 0.0]]]),
        np.asarray([[[-0.1, 0.0, 0.0]]]),
    ],
)
def test_rgb_to_bgr_uint8_rejects_invalid_images(image: np.ndarray) -> None:
    with pytest.raises((TypeError, ValueError)):
        rgb_to_bgr_uint8(image)


def test_detector_requires_initialization() -> None:
    detector = YoloSegmentationDetector()

    assert not detector.is_initialized
    with pytest.raises(RuntimeError, match="Initialize"):
        detector.detect(
            np.zeros((4, 5, 3), dtype=np.uint8),
            source_frame_id=0,
            source_timestamp_seconds=0.0,
        )


def test_detector_rejects_empty_image_before_calling_backend() -> None:
    model = FakeModel([make_result(image_shape=(0, 5))])
    detector = YoloSegmentationDetector(model_backend=model)

    with pytest.raises(ValueError, match="height and width"):
        detector.detect(
            np.zeros((0, 5, 3), dtype=np.uint8),
            source_frame_id=0,
            source_timestamp_seconds=0.0,
        )

    assert model.calls == []


def test_detector_converts_segmentation_result_and_model_arguments() -> None:
    model = FakeModel([make_result()])
    config = YoloSegmentationConfig(
        model_path="custom-seg.pt",
        confidence_threshold=0.4,
        iou_threshold=0.6,
        image_size=320,
        device="cpu",
        maximum_detections=7,
    )
    detector = YoloSegmentationDetector(config, model_backend=model)
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    image[0, 0] = [11, 22, 33]

    frame = detector.detect(
        image,
        source_frame_id=12,
        source_timestamp_seconds=4.5,
    )

    assert detector.is_initialized
    assert detector.model_name == "custom-seg.pt"
    assert frame.source_frame_id == 12
    assert frame.source_timestamp_seconds == pytest.approx(4.5)
    assert frame.image_height == 4
    assert frame.image_width == 5
    assert frame.model_name == "custom-seg.pt"
    assert frame.class_counts == {"dining_chair": 1, "person": 1}
    assert frame.detections[0].bounding_box.x_min == 0.0
    assert frame.detections[1].bounding_box.x_max == 5.0
    assert frame.detections[0].mask_pixel_count == 6
    assert frame.detections[1].mask_pixel_count == 9

    assert len(model.calls) == 1
    call = model.calls[0]
    assert np.array_equal(call["source"][0, 0], [33, 22, 11])
    assert call["conf"] == pytest.approx(0.4)
    assert call["iou"] == pytest.approx(0.6)
    assert call["imgsz"] == 320
    assert call["device"] == "cpu"
    assert call["max_det"] == 7
    assert call["retina_masks"] is True
    assert call["verbose"] is False
    assert call["save"] is False


def test_detector_filters_normalized_allowed_classes() -> None:
    model = FakeModel([make_result()])
    detector = YoloSegmentationDetector(
        YoloSegmentationConfig(allowed_classes=("Dining Chair",)),
        model_backend=model,
    )

    frame = detector.detect(
        np.zeros((4, 5, 3), dtype=np.uint8),
        source_frame_id=0,
        source_timestamp_seconds=0.0,
    )

    assert [item.semantic_class for item in frame.detections] == [
        "dining_chair"
    ]


def test_detector_accepts_empty_result_without_masks() -> None:
    empty_result = SimpleNamespace(
        boxes=SimpleNamespace(
            xyxy=np.empty((0, 4)),
            conf=np.empty((0,)),
            cls=np.empty((0,)),
        ),
        masks=None,
        names={},
    )
    detector = YoloSegmentationDetector(
        model_backend=FakeModel([empty_result])
    )

    frame = detector.detect(
        np.zeros((4, 5, 3), dtype=np.uint8),
        source_frame_id=0,
        source_timestamp_seconds=0.0,
    )

    assert frame.detections == ()


def test_detector_rejects_boxes_without_instance_masks() -> None:
    result = make_result()
    result.masks = None
    detector = YoloSegmentationDetector(model_backend=FakeModel([result]))

    with pytest.raises(RuntimeError, match="without instance masks"):
        detector.detect(
            np.zeros((4, 5, 3), dtype=np.uint8),
            source_frame_id=0,
            source_timestamp_seconds=0.0,
        )


def test_detector_rejects_mask_shape_mismatch() -> None:
    result = make_result()
    result.masks.data = np.zeros((2, 2, 2), dtype=np.float32)
    detector = YoloSegmentationDetector(model_backend=FakeModel([result]))

    with pytest.raises(RuntimeError, match="source image shape"):
        detector.detect(
            np.zeros((4, 5, 3), dtype=np.uint8),
            source_frame_id=0,
            source_timestamp_seconds=0.0,
        )


def test_detector_rejects_multiple_results_for_one_image() -> None:
    detector = YoloSegmentationDetector(
        model_backend=FakeModel([make_result(), make_result()])
    )

    with pytest.raises(RuntimeError, match="exactly one"):
        detector.detect(
            np.zeros((4, 5, 3), dtype=np.uint8),
            source_frame_id=0,
            source_timestamp_seconds=0.0,
        )


@pytest.mark.parametrize("bad_class_id", [float("nan"), 1.5])
def test_detector_rejects_invalid_class_ids(bad_class_id: float) -> None:
    result = make_result()
    result.boxes.cls[0] = bad_class_id
    detector = YoloSegmentationDetector(model_backend=FakeModel([result]))

    with pytest.raises(RuntimeError, match="finite integers"):
        detector.detect(
            np.zeros((4, 5, 3), dtype=np.uint8),
            source_frame_id=0,
            source_timestamp_seconds=0.0,
        )

"""Deterministic visualization of project-owned semantic detections."""

import zlib

import numpy as np
from PIL import Image, ImageDraw

from robot.perception.semantic_detection import SemanticDetectionFrame
from robot.perception.yolo_detector import rgb_to_bgr_uint8


_PALETTE = (
    (35, 220, 120),
    (255, 110, 70),
    (80, 170, 255),
    (235, 200, 55),
    (190, 90, 240),
    (40, 215, 220),
)


def _class_color(semantic_class: str) -> tuple[int, int, int]:
    index = zlib.crc32(semantic_class.encode("utf-8")) % len(_PALETTE)
    return _PALETTE[index]


def render_semantic_detection_overlay(
    image_rgb: object,
    frame: SemanticDetectionFrame,
    *,
    mask_opacity: float = 0.35,
    line_width: int = 3,
) -> np.ndarray:
    """Return an RGB image containing masks, boxes, and class labels."""

    if not isinstance(frame, SemanticDetectionFrame):
        raise TypeError("frame must be a SemanticDetectionFrame.")
    if isinstance(mask_opacity, bool) or not isinstance(
        mask_opacity,
        (int, float),
    ):
        raise TypeError("mask_opacity must be a real number.")
    opacity = float(mask_opacity)
    if not 0.0 <= opacity <= 1.0:
        raise ValueError("mask_opacity must lie in [0, 1].")
    if isinstance(line_width, bool) or not isinstance(line_width, int):
        raise TypeError("line_width must be an integer.")
    if line_width <= 0:
        raise ValueError("line_width must be greater than zero.")

    # Reuse the detector's rigorously tested conversion, then reverse BGR
    # back to RGB.  This also owns the returned storage.
    image = np.ascontiguousarray(rgb_to_bgr_uint8(image_rgb)[:, :, ::-1])
    if image.shape[:2] != (frame.image_height, frame.image_width):
        raise ValueError("image_rgb dimensions must match the detection frame.")

    blended = image.astype(np.float64)
    for detection in frame.detections:
        color = np.asarray(
            _class_color(detection.semantic_class),
            dtype=np.float64,
        )
        mask = detection.mask
        blended[mask] = (1.0 - opacity) * blended[mask] + opacity * color
    annotated = np.clip(blended, 0.0, 255.0).astype(np.uint8)

    pil_image = Image.fromarray(annotated)
    draw = ImageDraw.Draw(pil_image)
    for detection in frame.detections:
        color = _class_color(detection.semantic_class)
        box = detection.bounding_box
        draw.rectangle(
            (box.x_min, box.y_min, box.x_max, box.y_max),
            outline=color,
            width=line_width,
        )
        label = f"{detection.semantic_class} {detection.confidence:.2f}"
        text_box = draw.textbbox((box.x_min, box.y_min), label)
        text_height = text_box[3] - text_box[1]
        label_top = max(0.0, box.y_min - text_height - 4.0)
        label_box = draw.textbbox((box.x_min + 2.0, label_top + 2.0), label)
        draw.rectangle(
            (
                box.x_min,
                label_top,
                label_box[2] + 2.0,
                label_box[3] + 2.0,
            ),
            fill=color,
        )
        draw.text(
            (box.x_min + 2.0, label_top + 2.0),
            label,
            fill=(0, 0, 0),
        )
    return np.asarray(pil_image, dtype=np.uint8).copy()

"""Dependency-free semantic instance-detection data structures.

The safety stack should not depend on a particular vision framework.  These
immutable NumPy-backed values form the boundary between an image model such
as YOLO and later camera--LiDAR association.  Model-specific tensors and
result objects must be converted before they cross this boundary.
"""

import math
from collections import Counter
from dataclasses import dataclass
from numbers import Integral, Real
from types import MappingProxyType
from typing import Mapping

import numpy as np


def _finite_real(value: object, *, name: str) -> float:
    """Return a finite real number while rejecting booleans."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _nonnegative_integer(value: object, *, name: str) -> int:
    """Return a nonnegative integer while rejecting booleans."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} cannot be negative.")
    return result


def normalize_detection_class(value: str) -> str:
    """Return a stable identifier for a visual class name."""

    if not isinstance(value, str):
        raise TypeError("semantic_class must be a string.")
    normalized = "_".join(
        value.strip().casefold().replace("-", " ").split()
    )
    if not normalized:
        raise ValueError("semantic_class cannot be empty.")
    return normalized


@dataclass(frozen=True)
class PixelBoundingBox:
    """One axis-aligned bounding box in source-image pixel coordinates."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        x_min = _finite_real(self.x_min, name="x_min")
        y_min = _finite_real(self.y_min, name="y_min")
        x_max = _finite_real(self.x_max, name="x_max")
        y_max = _finite_real(self.y_max, name="y_max")
        if x_min < 0.0 or y_min < 0.0:
            raise ValueError("Bounding-box minima cannot be negative.")
        if x_max <= x_min or y_max <= y_min:
            raise ValueError(
                "Bounding-box maxima must be greater than their minima."
            )
        object.__setattr__(self, "x_min", x_min)
        object.__setattr__(self, "y_min", y_min)
        object.__setattr__(self, "x_max", x_max)
        object.__setattr__(self, "y_max", y_max)

    @property
    def width(self) -> float:
        """Return the box width in pixels."""

        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        """Return the box height in pixels."""

        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        """Return the box area in square pixels."""

        return self.width * self.height

    @property
    def center_xy(self) -> tuple[float, float]:
        """Return the box center in pixel coordinates."""

        return (
            0.5 * (self.x_min + self.x_max),
            0.5 * (self.y_min + self.y_max),
        )


@dataclass(frozen=True)
class SemanticDetection:
    """One classified image instance with an immutable pixel mask."""

    class_id: int
    semantic_class: str
    confidence: float
    bounding_box: PixelBoundingBox
    mask: np.ndarray

    def __post_init__(self) -> None:
        class_id = _nonnegative_integer(self.class_id, name="class_id")
        semantic_class = normalize_detection_class(self.semantic_class)
        confidence = _finite_real(self.confidence, name="confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must lie in [0, 1].")
        if not isinstance(self.bounding_box, PixelBoundingBox):
            raise TypeError("bounding_box must be a PixelBoundingBox.")
        mask = np.asarray(self.mask)
        if mask.ndim != 2:
            raise ValueError("mask must have shape (height, width).")
        if mask.dtype != np.bool_:
            raise TypeError("mask must have boolean dtype.")
        mask = np.array(mask, dtype=np.bool_, copy=True, order="C")
        mask.setflags(write=False)
        object.__setattr__(self, "class_id", class_id)
        object.__setattr__(self, "semantic_class", semantic_class)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "mask", mask)

    @property
    def mask_pixel_count(self) -> int:
        """Return the number of source-image pixels in this instance."""

        return int(np.count_nonzero(self.mask))


@dataclass(frozen=True)
class SemanticDetectionFrame:
    """Validated semantic instances produced from one camera frame."""

    source_frame_id: int
    source_timestamp_seconds: float
    image_height: int
    image_width: int
    detections: tuple[SemanticDetection, ...]
    model_name: str
    inference_wall_seconds: float

    def __post_init__(self) -> None:
        frame_id = _nonnegative_integer(
            self.source_frame_id,
            name="source_frame_id",
        )
        timestamp = _finite_real(
            self.source_timestamp_seconds,
            name="source_timestamp_seconds",
        )
        height = _nonnegative_integer(self.image_height, name="image_height")
        width = _nonnegative_integer(self.image_width, name="image_width")
        inference_time = _finite_real(
            self.inference_wall_seconds,
            name="inference_wall_seconds",
        )
        if timestamp < 0.0:
            raise ValueError("source_timestamp_seconds cannot be negative.")
        if height == 0 or width == 0:
            raise ValueError("Image dimensions must be greater than zero.")
        if inference_time < 0.0:
            raise ValueError("inference_wall_seconds cannot be negative.")
        if not isinstance(self.model_name, str):
            raise TypeError("model_name must be a string.")
        model_name = self.model_name.strip()
        if not model_name:
            raise ValueError("model_name cannot be empty.")
        try:
            detections = tuple(self.detections)
        except TypeError as error:
            raise TypeError("detections must be an iterable.") from error
        for detection in detections:
            if not isinstance(detection, SemanticDetection):
                raise TypeError(
                    "Every detection must be a SemanticDetection."
                )
            if detection.mask.shape != (height, width):
                raise ValueError(
                    "Every detection mask must match the source image shape."
                )
            box = detection.bounding_box
            if box.x_max > width or box.y_max > height:
                raise ValueError(
                    "Detection bounding boxes must lie inside the source image."
                )
        object.__setattr__(self, "source_frame_id", frame_id)
        object.__setattr__(self, "source_timestamp_seconds", timestamp)
        object.__setattr__(self, "image_height", height)
        object.__setattr__(self, "image_width", width)
        object.__setattr__(self, "detections", detections)
        object.__setattr__(self, "model_name", model_name)
        object.__setattr__(self, "inference_wall_seconds", inference_time)

    @property
    def class_counts(self) -> Mapping[str, int]:
        """Return immutable per-class instance counts."""

        counts = Counter(
            detection.semantic_class for detection in self.detections
        )
        return MappingProxyType(dict(sorted(counts.items())))

    def detections_for(self, semantic_class: str) -> tuple[SemanticDetection, ...]:
        """Return detections matching one normalized class name."""

        normalized = normalize_detection_class(semantic_class)
        return tuple(
            detection
            for detection in self.detections
            if detection.semantic_class == normalized
        )

"""Optional Ultralytics YOLO11n instance-segmentation adapter.

Ultralytics is imported only when :meth:`YoloSegmentationDetector.initialize`
is called.  Importing the project or running ordinary unit tests therefore
does not require the external vision package, CUDA, or model weights.
"""

import math
import time
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from robot.perception.semantic_detection import (
    PixelBoundingBox,
    SemanticDetection,
    SemanticDetectionFrame,
    normalize_detection_class,
)


class YoloModelBackend(Protocol):
    """Small backend surface used by the adapter and test doubles."""

    def predict(self, **kwargs: Any) -> object:
        """Return model-specific results for one input image."""


def _finite_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    return result


@dataclass(frozen=True)
class YoloSegmentationConfig:
    """Inference configuration for the Safe-SAGE visual-label baseline."""

    model_path: str | Path = "yolo11n-seg.pt"
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.70
    image_size: int = 640
    device: str = "0"
    maximum_detections: int = 100
    allowed_classes: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.model_path, str) and not self.model_path.strip():
            raise ValueError("model_path cannot be empty.")
        try:
            model_path = str(Path(self.model_path).expanduser())
        except TypeError as error:
            raise TypeError("model_path must be a path-like value.") from error
        if not model_path.strip():
            raise ValueError("model_path cannot be empty.")
        confidence = _finite_real(
            self.confidence_threshold,
            name="confidence_threshold",
        )
        iou = _finite_real(self.iou_threshold, name="iou_threshold")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence_threshold must lie in [0, 1].")
        if not 0.0 <= iou <= 1.0:
            raise ValueError("iou_threshold must lie in [0, 1].")
        image_size = _positive_integer(self.image_size, name="image_size")
        maximum_detections = _positive_integer(
            self.maximum_detections,
            name="maximum_detections",
        )
        if not isinstance(self.device, str):
            raise TypeError("device must be a string.")
        device = self.device.strip()
        if not device:
            raise ValueError("device cannot be empty.")
        allowed_classes = self.allowed_classes
        if allowed_classes is not None:
            if isinstance(allowed_classes, (str, bytes)):
                raise TypeError(
                    "allowed_classes must be an iterable of class names."
                )
            try:
                normalized_classes = tuple(
                    normalize_detection_class(value)
                    for value in allowed_classes
                )
            except TypeError as error:
                raise TypeError("allowed_classes must be an iterable.") from error
            if len(set(normalized_classes)) != len(normalized_classes):
                raise ValueError("allowed_classes cannot contain duplicates.")
            allowed_classes = normalized_classes
        object.__setattr__(self, "model_path", model_path)
        object.__setattr__(self, "confidence_threshold", confidence)
        object.__setattr__(self, "iou_threshold", iou)
        object.__setattr__(self, "image_size", image_size)
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "maximum_detections", maximum_detections)
        object.__setattr__(self, "allowed_classes", allowed_classes)


def rgb_to_bgr_uint8(image_rgb: object) -> np.ndarray:
    """Convert an RGB image to contiguous BGR ``uint8`` without mutation."""

    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("image_rgb must have shape (height, width, 3 or 4).")
    image = image[:, :, :3]
    if np.issubdtype(image.dtype, np.floating):
        if not np.all(np.isfinite(image)):
            raise ValueError("image_rgb must contain finite values.")
        image = np.array(image, dtype=np.float64, copy=True)
        maximum = float(np.max(image)) if image.size else 0.0
        minimum = float(np.min(image)) if image.size else 0.0
        if minimum < 0.0:
            raise ValueError("Floating RGB values cannot be negative.")
        if maximum <= 1.0:
            image *= 255.0
    elif not np.issubdtype(image.dtype, np.integer):
        raise TypeError("image_rgb must contain integer or floating values.")
    image_uint8 = np.clip(image, 0.0, 255.0).astype(np.uint8, copy=False)
    return np.ascontiguousarray(image_uint8[:, :, ::-1])


def _to_numpy(value: object, *, name: str) -> np.ndarray:
    """Copy one tensor-like result to CPU NumPy form."""

    candidate = value
    if hasattr(candidate, "detach"):
        candidate = candidate.detach()
    if hasattr(candidate, "cpu"):
        candidate = candidate.cpu()
    if hasattr(candidate, "numpy"):
        candidate = candidate.numpy()
    try:
        result = np.asarray(candidate)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} could not be converted to NumPy.") from error
    return np.array(result, copy=True)


def _class_name(names: object, class_id: int) -> str:
    """Resolve one class ID from Ultralytics' mapping or sequence."""

    try:
        if isinstance(names, dict):
            value = names[class_id]
        else:
            value = names[class_id]  # type: ignore[index]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(
            f"YOLO did not provide a name for class ID {class_id}."
        ) from error
    return normalize_detection_class(value)


def _single_result(results: object) -> object:
    """Return exactly one result from one-image inference."""

    try:
        materialized = tuple(results)  # type: ignore[arg-type]
    except TypeError as error:
        raise RuntimeError("YOLO predict() did not return an iterable.") from error
    if len(materialized) != 1:
        raise RuntimeError(
            "One-image inference must produce exactly one YOLO result."
        )
    return materialized[0]


class YoloSegmentationDetector:
    """Convert YOLO11n-seg output into project-owned semantic instances."""

    def __init__(
        self,
        config: YoloSegmentationConfig | None = None,
        *,
        model_backend: YoloModelBackend | None = None,
    ) -> None:
        self.config = config or YoloSegmentationConfig()
        if not isinstance(self.config, YoloSegmentationConfig):
            raise TypeError("config must be a YoloSegmentationConfig.")
        self._model: YoloModelBackend | None = model_backend

    @property
    def is_initialized(self) -> bool:
        """Whether model weights have been loaded."""

        return self._model is not None

    @property
    def model_name(self) -> str:
        """Return the configured model filename for reports."""

        return Path(self.config.model_path).name

    def initialize(self) -> None:
        """Load Ultralytics and the configured segmentation weights."""

        if self.is_initialized:
            return
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "Ultralytics is unavailable. Activate the project's YOLO "
                "environment before initializing visual perception."
            ) from error
        try:
            self._model = YOLO(self.config.model_path)
        except Exception as error:
            raise RuntimeError(
                f"Could not load YOLO weights: {self.config.model_path}"
            ) from error

    def detect(
        self,
        image_rgb: object,
        *,
        source_frame_id: int,
        source_timestamp_seconds: float,
    ) -> SemanticDetectionFrame:
        """Run one RGB frame and return dependency-free instance masks."""

        if self._model is None:
            raise RuntimeError("Initialize the YOLO detector before detect().")
        image_bgr = rgb_to_bgr_uint8(image_rgb)
        height, width = image_bgr.shape[:2]
        if height == 0 or width == 0:
            raise ValueError("image_rgb height and width must be positive.")
        start = time.perf_counter()
        results = self._model.predict(
            source=image_bgr,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.image_size,
            device=self.config.device,
            max_det=self.config.maximum_detections,
            retina_masks=True,
            verbose=False,
            save=False,
        )
        inference_wall_seconds = time.perf_counter() - start
        result = _single_result(results)
        detections = self._convert_result(
            result,
            image_height=height,
            image_width=width,
        )
        return SemanticDetectionFrame(
            source_frame_id=source_frame_id,
            source_timestamp_seconds=source_timestamp_seconds,
            image_height=height,
            image_width=width,
            detections=detections,
            model_name=self.model_name,
            inference_wall_seconds=inference_wall_seconds,
        )

    def _convert_result(
        self,
        result: object,
        *,
        image_height: int,
        image_width: int,
    ) -> tuple[SemanticDetection, ...]:
        boxes_object = getattr(result, "boxes", None)
        names = getattr(result, "names", None)
        if boxes_object is None:
            return ()
        xyxy = _to_numpy(
            getattr(boxes_object, "xyxy", None),
            name="boxes.xyxy",
        )
        confidences = _to_numpy(
            getattr(boxes_object, "conf", None),
            name="boxes.conf",
        ).reshape(-1)
        class_ids = _to_numpy(
            getattr(boxes_object, "cls", None),
            name="boxes.cls",
        ).reshape(-1)
        if xyxy.ndim != 2 or xyxy.shape[1] != 4:
            raise RuntimeError("YOLO boxes.xyxy must have shape (N, 4).")
        detection_count = xyxy.shape[0]
        if confidences.shape != (detection_count,) or class_ids.shape != (
            detection_count,
        ):
            raise RuntimeError("YOLO box outputs have inconsistent lengths.")
        if detection_count == 0:
            return ()
        masks_object = getattr(result, "masks", None)
        if masks_object is None:
            raise RuntimeError(
                "YOLO returned boxes without instance masks. Use segmentation "
                "weights such as yolo11n-seg.pt."
            )
        masks = _to_numpy(
            getattr(masks_object, "data", None),
            name="masks.data",
        )
        if masks.shape != (detection_count, image_height, image_width):
            raise RuntimeError(
                "YOLO masks must match the number of boxes and source image "
                f"shape; received {masks.shape}."
            )
        allowed = (
            None
            if self.config.allowed_classes is None
            else set(self.config.allowed_classes)
        )
        detections: list[SemanticDetection] = []
        for index in range(detection_count):
            raw_class_id = float(class_ids[index])
            if not math.isfinite(raw_class_id):
                raise RuntimeError("YOLO class IDs must be finite integers.")
            class_id = int(raw_class_id)
            if raw_class_id != class_id:
                raise RuntimeError("YOLO class IDs must be finite integers.")
            semantic_class = _class_name(names, class_id)
            if allowed is not None and semantic_class not in allowed:
                continue
            coordinates = np.asarray(xyxy[index], dtype=np.float64)
            if not np.all(np.isfinite(coordinates)):
                raise RuntimeError("YOLO bounding boxes must be finite.")
            x_min = float(np.clip(coordinates[0], 0.0, image_width))
            y_min = float(np.clip(coordinates[1], 0.0, image_height))
            x_max = float(np.clip(coordinates[2], 0.0, image_width))
            y_max = float(np.clip(coordinates[3], 0.0, image_height))
            detections.append(
                SemanticDetection(
                    class_id=class_id,
                    semantic_class=semantic_class,
                    confidence=float(confidences[index]),
                    bounding_box=PixelBoundingBox(
                        x_min=x_min,
                        y_min=y_min,
                        x_max=x_max,
                        y_max=y_max,
                    ),
                    mask=np.asarray(masks[index]) >= 0.5,
                )
            )
        return tuple(detections)

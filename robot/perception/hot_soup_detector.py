"""Fixed-label hot-soup detector boundary and contextual box association.

The optional Ultralytics backend detects only a visible hot-soup plume and an
open container. Box geometry then supplies the contextual "plume above
container" cue. None of these outputs measure temperature or prove steam.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from numbers import Integral, Real
from pathlib import Path
import time
from typing import Any, Protocol

import numpy as np

from robot.perception.hot_soup_cues import (
    HotSoupCueConfig,
    HotSoupCueResult,
    estimate_hot_soup_cues,
)
from robot.perception.physical_attributes import EstimateStatus, ScalarEstimate
from robot.perception.semantic_detection import (
    PixelBoundingBox,
    normalize_detection_class,
)
from robot.perception.yolo_detector import rgb_to_bgr_uint8


HOT_SOUP_DETECTOR_SOURCE = "fixed_label_hot_soup_detector"
HOT_SOUP_DETECTION_CLASSES = frozenset(
    {"visible_hot_soup_plume", "open_container"}
)


class HotSoupModelBackend(Protocol):
    def predict(self, **kwargs: Any) -> object:
        """Return model-specific detections for one image."""


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


@dataclass(frozen=True)
class HotSoupDetectorConfig:
    model_path: str | Path
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.70
    image_size: int = 640
    device: str = "0"
    maximum_detections: int = 50

    def __post_init__(self) -> None:
        path = str(Path(self.model_path).expanduser())
        if not path.strip():
            raise ValueError("model_path cannot be empty")
        confidence = _finite(self.confidence_threshold, "confidence_threshold")
        iou = _finite(self.iou_threshold, "iou_threshold")
        if not 0 <= confidence <= 1 or not 0 <= iou <= 1:
            raise ValueError("confidence and IoU thresholds must lie in [0, 1]")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be nonempty text")
        object.__setattr__(self, "model_path", path)
        object.__setattr__(self, "confidence_threshold", confidence)
        object.__setattr__(self, "iou_threshold", iou)
        object.__setattr__(self, "image_size", _positive_integer(self.image_size, "image_size"))
        object.__setattr__(
            self,
            "maximum_detections",
            _positive_integer(self.maximum_detections, "maximum_detections"),
        )
        object.__setattr__(self, "device", self.device.strip())


@dataclass(frozen=True)
class HotSoupDetection:
    class_id: int
    semantic_class: str
    confidence: float
    bounding_box: PixelBoundingBox

    def __post_init__(self) -> None:
        if isinstance(self.class_id, bool) or not isinstance(self.class_id, Integral):
            raise TypeError("class_id must be an integer")
        class_id = int(self.class_id)
        if class_id < 0:
            raise ValueError("class_id cannot be negative")
        semantic_class = normalize_detection_class(self.semantic_class)
        if semantic_class not in HOT_SOUP_DETECTION_CLASSES:
            raise ValueError("unsupported hot-soup detector class")
        confidence = _finite(self.confidence, "confidence")
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must lie in [0, 1]")
        if not isinstance(self.bounding_box, PixelBoundingBox):
            raise TypeError("bounding_box must be a PixelBoundingBox")
        object.__setattr__(self, "class_id", class_id)
        object.__setattr__(self, "semantic_class", semantic_class)
        object.__setattr__(self, "confidence", confidence)


@dataclass(frozen=True)
class HotSoupDetectionFrame:
    source_frame_id: int
    source_timestamp_seconds: float
    image_height: int
    image_width: int
    detections: tuple[HotSoupDetection, ...]
    model_name: str
    inference_wall_seconds: float
    checkpoint_sha256: str | None = None
    evidence_status: EstimateStatus = EstimateStatus.INFERRED

    def __post_init__(self) -> None:
        if isinstance(self.source_frame_id, bool) or not isinstance(
            self.source_frame_id, Integral
        ):
            raise TypeError("source_frame_id must be an integer")
        frame_id = int(self.source_frame_id)
        if frame_id < 0:
            raise ValueError("source_frame_id cannot be negative")
        timestamp = _finite(self.source_timestamp_seconds, "source timestamp")
        if timestamp < 0:
            raise ValueError("source timestamp cannot be negative")
        height = _positive_integer(self.image_height, "image_height")
        width = _positive_integer(self.image_width, "image_width")
        inference = _finite(self.inference_wall_seconds, "inference_wall_seconds")
        if inference < 0:
            raise ValueError("inference_wall_seconds cannot be negative")
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("model_name must be nonempty text")
        detections = tuple(self.detections)
        for detection in detections:
            if not isinstance(detection, HotSoupDetection):
                raise TypeError("detections must contain HotSoupDetection values")
            box = detection.bounding_box
            if box.x_max > width or box.y_max > height:
                raise ValueError("detection boxes must stay inside the image")
        digest = self.checkpoint_sha256
        if digest is not None:
            digest = digest.lower()
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError("checkpoint_sha256 must be a hexadecimal SHA-256 digest")
        try:
            evidence_status = EstimateStatus(self.evidence_status)
        except (TypeError, ValueError) as error:
            raise ValueError("evidence_status must be an EstimateStatus") from error
        if evidence_status is EstimateStatus.UNAVAILABLE:
            raise ValueError("a detection frame cannot have unavailable provenance")
        object.__setattr__(self, "source_frame_id", frame_id)
        object.__setattr__(self, "source_timestamp_seconds", timestamp)
        object.__setattr__(self, "image_height", height)
        object.__setattr__(self, "image_width", width)
        object.__setattr__(self, "detections", detections)
        object.__setattr__(self, "model_name", self.model_name.strip())
        object.__setattr__(self, "inference_wall_seconds", inference)
        object.__setattr__(self, "checkpoint_sha256", digest)
        object.__setattr__(self, "evidence_status", evidence_status)


@dataclass(frozen=True)
class HotSoupAssociationConfig:
    """Continuous plume-to-container geometry score configuration.

    ``minimum_horizontal_overlap_fraction`` is retained as a compatibility
    name. It is the overlap that receives a full score, not a rejection
    threshold; smaller observed overlaps retain proportionally weaker evidence.
    """

    minimum_horizontal_overlap_fraction: float = 0.25
    maximum_vertical_gap_container_heights: float = 2.0
    vertical_separation_full_score_container_heights: float = 0.75
    detector_score_half_width: float = 0.10

    def __post_init__(self) -> None:
        overlap = _finite(
            self.minimum_horizontal_overlap_fraction,
            "minimum_horizontal_overlap_fraction",
        )
        gap = _finite(
            self.maximum_vertical_gap_container_heights,
            "maximum_vertical_gap_container_heights",
        )
        vertical = _finite(
            self.vertical_separation_full_score_container_heights,
            "vertical_separation_full_score_container_heights",
        )
        half_width = _finite(self.detector_score_half_width, "detector_score_half_width")
        if not 0 < overlap <= 1:
            raise ValueError("minimum horizontal overlap must lie in (0, 1]")
        if gap <= 0 or vertical <= 0:
            raise ValueError("vertical association scales must be positive")
        if not 0 <= half_width <= 1:
            raise ValueError("detector_score_half_width must lie in [0, 1]")
        object.__setattr__(self, "minimum_horizontal_overlap_fraction", overlap)
        object.__setattr__(self, "maximum_vertical_gap_container_heights", gap)
        object.__setattr__(
            self,
            "vertical_separation_full_score_container_heights",
            vertical,
        )
        object.__setattr__(self, "detector_score_half_width", half_width)


@dataclass(frozen=True)
class HotSoupDetectionCueResult:
    cue_result: HotSoupCueResult
    selected_plume_index: int | None
    selected_container_index: int | None
    candidate_pair_count: int
    spatial_association_score: float | None


class YoloHotSoupDetector:
    """Optional fixed-label YOLO detection adapter; no model is bundled."""

    def __init__(
        self,
        config: HotSoupDetectorConfig,
        *,
        model_backend: HotSoupModelBackend | None = None,
    ) -> None:
        if not isinstance(config, HotSoupDetectorConfig):
            raise TypeError("config must be HotSoupDetectorConfig")
        self.config = config
        self._model = model_backend
        self.checkpoint_sha256: str | None = None

    @property
    def is_initialized(self) -> bool:
        return self._model is not None

    def initialize(self) -> None:
        if self.is_initialized:
            return
        path = Path(self.config.model_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError("Ultralytics is unavailable in this environment") from error
        self._model = YOLO(str(path))
        self.checkpoint_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    def detect(
        self,
        image_rgb: object,
        *,
        source_frame_id: int,
        source_timestamp_seconds: float,
    ) -> HotSoupDetectionFrame:
        if self._model is None:
            raise RuntimeError("initialize the hot-soup detector before detect()")
        image_bgr = rgb_to_bgr_uint8(image_rgb)
        height, width = image_bgr.shape[:2]
        start = time.perf_counter()
        raw_results = self._model.predict(
            source=image_bgr,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.image_size,
            device=self.config.device,
            max_det=self.config.maximum_detections,
            verbose=False,
            save=False,
        )
        elapsed = time.perf_counter() - start
        results = tuple(raw_results)
        if len(results) != 1:
            raise RuntimeError("one-image inference must return exactly one result")
        detections = _convert_yolo_result(results[0], width=width, height=height)
        return HotSoupDetectionFrame(
            source_frame_id=source_frame_id,
            source_timestamp_seconds=source_timestamp_seconds,
            image_height=height,
            image_width=width,
            detections=detections,
            model_name=Path(self.config.model_path).name,
            inference_wall_seconds=elapsed,
            checkpoint_sha256=self.checkpoint_sha256,
        )


def _as_numpy(value: object, name: str) -> np.ndarray:
    candidate = value
    if hasattr(candidate, "detach"):
        candidate = candidate.detach()
    if hasattr(candidate, "cpu"):
        candidate = candidate.cpu()
    if hasattr(candidate, "numpy"):
        candidate = candidate.numpy()
    try:
        return np.asarray(candidate)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} could not be converted to NumPy") from error


def _convert_yolo_result(
    result: object, *, width: int, height: int
) -> tuple[HotSoupDetection, ...]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return ()
    xyxy = _as_numpy(getattr(boxes, "xyxy", None), "boxes.xyxy")
    confidence = _as_numpy(getattr(boxes, "conf", None), "boxes.conf").reshape(-1)
    class_ids = _as_numpy(getattr(boxes, "cls", None), "boxes.cls").reshape(-1)
    if xyxy.ndim != 2 or xyxy.shape[1] != 4:
        raise RuntimeError("YOLO boxes.xyxy must have shape (N, 4)")
    if confidence.shape != (len(xyxy),) or class_ids.shape != (len(xyxy),):
        raise RuntimeError("YOLO box outputs have inconsistent lengths")
    names = getattr(result, "names", None)
    detections = []
    for index, coordinates in enumerate(xyxy):
        raw_class_id = float(class_ids[index])
        class_id = int(raw_class_id)
        if not math.isfinite(raw_class_id) or raw_class_id != class_id:
            raise RuntimeError("YOLO class IDs must be finite integers")
        try:
            raw_name = names[class_id]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError("YOLO result is missing a class name") from error
        class_name = normalize_detection_class(raw_name)
        if class_name not in HOT_SOUP_DETECTION_CLASSES:
            continue
        values = np.asarray(coordinates, dtype=float)
        if not np.isfinite(values).all():
            raise RuntimeError("YOLO bounding boxes must be finite")
        x_min = float(np.clip(values[0], 0, width))
        y_min = float(np.clip(values[1], 0, height))
        x_max = float(np.clip(values[2], 0, width))
        y_max = float(np.clip(values[3], 0, height))
        if x_max <= x_min or y_max <= y_min:
            continue
        detections.append(
            HotSoupDetection(
                class_id=class_id,
                semantic_class=class_name,
                confidence=float(confidence[index]),
                bounding_box=PixelBoundingBox(x_min, y_min, x_max, y_max),
            )
        )
    return tuple(detections)


def _score_pair(
    plume: PixelBoundingBox,
    container: PixelBoundingBox,
    config: HotSoupAssociationConfig,
) -> float:
    overlap_width = max(
        0.0, min(plume.x_max, container.x_max) - max(plume.x_min, container.x_min)
    )
    horizontal_fraction = overlap_width / min(plume.width, container.width)
    gap = max(0.0, container.y_min - plume.y_max)
    maximum_gap = config.maximum_vertical_gap_container_heights * container.height
    gap_score = max(0.0, 1.0 - gap / maximum_gap)
    overlap_score = min(
        1.0,
        horizontal_fraction / config.minimum_horizontal_overlap_fraction,
    )
    vertical_separation = (
        container.center_xy[1] - plume.center_xy[1]
    ) / container.height
    vertical_score = min(
        1.0,
        max(0.0, vertical_separation)
        / config.vertical_separation_full_score_container_heights,
    )
    return min(overlap_score, vertical_score, gap_score)


def _cue(
    value: float,
    *,
    timestamp: float,
    confidence: float,
    half_width: float,
    status: EstimateStatus,
    evidence: tuple[str, ...],
) -> ScalarEstimate:
    return ScalarEstimate(
        value=value,
        lower_bound=max(0.0, value - half_width),
        upper_bound=min(1.0, value + half_width),
        confidence=confidence,
        unit="1",
        source=HOT_SOUP_DETECTOR_SOURCE,
        timestamp_seconds=timestamp,
        status=status,
        evidence=evidence,
    )


def _unavailable(frame: HotSoupDetectionFrame, reason: str) -> ScalarEstimate:
    return ScalarEstimate.unavailable(
        unit="1",
        source=HOT_SOUP_DETECTOR_SOURCE,
        timestamp_seconds=frame.source_timestamp_seconds,
        evidence=(
            f"model={frame.model_name}",
            f"frame_id={frame.source_frame_id}",
            reason,
            "absence of a detection is not evidence of hazard absence",
        ),
    )


def estimate_hot_soup_from_detections(
    frame: HotSoupDetectionFrame,
    *,
    timestamp_seconds: float,
    association_config: HotSoupAssociationConfig | None = None,
    cue_config: HotSoupCueConfig | None = None,
) -> HotSoupDetectionCueResult:
    """Associate fixed-label boxes and invoke the thermal cue contract."""
    if not isinstance(frame, HotSoupDetectionFrame):
        raise TypeError("frame must be HotSoupDetectionFrame")
    settings = association_config or HotSoupAssociationConfig()
    plumes = [
        (index, item)
        for index, item in enumerate(frame.detections)
        if item.semantic_class == "visible_hot_soup_plume"
    ]
    containers = [
        (index, item)
        for index, item in enumerate(frame.detections)
        if item.semantic_class == "open_container"
    ]
    pairs = []
    for plume_index, plume in plumes:
        for container_index, container in containers:
            spatial = _score_pair(plume.bounding_box, container.bounding_box, settings)
            joint = min(plume.confidence, container.confidence, spatial)
            pairs.append((joint, spatial, plume_index, container_index, plume, container))

    plume_cue = _unavailable(frame, "no visible hot-soup plume detection")
    container_cue = _unavailable(frame, "no open-container detection")
    above_cue = _unavailable(frame, "no plume-container pair was available")
    selected_plume = None
    selected_container = None
    spatial_score = None
    if pairs:
        joint, spatial_score, selected_plume, selected_container, plume, container = max(
            pairs, key=lambda item: item[0]
        )
        evidence = (
            f"model={frame.model_name}",
            f"frame_id={frame.source_frame_id}",
            f"plume_detection_index={selected_plume}",
            f"container_detection_index={selected_container}",
            "detector scores are not calibrated temperatures or probabilities",
        )
        plume_cue = _cue(
            plume.confidence,
            timestamp=frame.source_timestamp_seconds,
            confidence=plume.confidence,
            half_width=settings.detector_score_half_width,
            status=frame.evidence_status,
            evidence=evidence,
        )
        container_cue = _cue(
            container.confidence,
            timestamp=frame.source_timestamp_seconds,
            confidence=container.confidence,
            half_width=settings.detector_score_half_width,
            status=frame.evidence_status,
            evidence=evidence,
        )
        above_cue = _cue(
            spatial_score,
            timestamp=frame.source_timestamp_seconds,
            confidence=joint,
            half_width=settings.detector_score_half_width,
            status=frame.evidence_status,
            evidence=evidence
            + (
                "spatial score uses horizontal overlap and vertical box ordering",
            ),
        )

    cue_result = estimate_hot_soup_cues(
        plume_likelihood=plume_cue,
        open_container_likelihood=container_cue,
        plume_above_container_likelihood=above_cue,
        timestamp_seconds=timestamp_seconds,
        plume_label="visible_plume",
        config=cue_config,
    )
    return HotSoupDetectionCueResult(
        cue_result=cue_result,
        selected_plume_index=selected_plume,
        selected_container_index=selected_container,
        candidate_pair_count=len(pairs),
        spatial_association_score=spatial_score,
    )

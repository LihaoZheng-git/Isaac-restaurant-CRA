"""Model-agnostic boundary for visible plume instance masks.

This module converts an existing semantic-segmentation frame into a fixed,
auditable plume vocabulary.  It deliberately does not infer temperature,
thermal severity, or a hot/not-hot class.  Model confidence remains epistemic
confidence; later feature extraction supplies the continuous visual evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping

import numpy as np

from robot.perception.physical_attributes import EstimateStatus
from robot.perception.semantic_detection import (
    PixelBoundingBox,
    SemanticDetectionFrame,
    normalize_detection_class,
)


PLUME_SEGMENTATION_SOURCE = "fixed_vocabulary_plume_segmentation"
PLUME_VISUAL_LABELS = frozenset({"steam", "smoke_like", "visible_plume"})


@dataclass(frozen=True)
class PlumeClassMap:
    """Explicit source-class to project plume-label mapping.

    An explicit map prevents a generic smoke class from silently becoming a
    steam observation.  ``steam`` should be used only when the source model was
    trained with that fixed label.
    """

    source_to_plume_label: Mapping[str, str]

    def __post_init__(self) -> None:
        normalized: dict[str, str] = {}
        for source, target in self.source_to_plume_label.items():
            source_name = normalize_detection_class(source)
            target_name = normalize_detection_class(target)
            if target_name not in PLUME_VISUAL_LABELS:
                raise ValueError(
                    f"plume label must be one of {sorted(PLUME_VISUAL_LABELS)}"
                )
            if source_name in normalized:
                raise ValueError("source plume classes must be unique")
            normalized[source_name] = target_name
        if not normalized:
            raise ValueError("at least one plume class mapping is required")
        object.__setattr__(
            self,
            "source_to_plume_label",
            MappingProxyType(normalized),
        )


@dataclass(frozen=True)
class VisualPlumeInstance:
    """One visible plume mask with model provenance."""

    plume_label: str
    source_class: str
    confidence: float
    bounding_box: PixelBoundingBox
    mask: np.ndarray
    source_detection_index: int

    def __post_init__(self) -> None:
        label = normalize_detection_class(self.plume_label)
        source_class = normalize_detection_class(self.source_class)
        if label not in PLUME_VISUAL_LABELS:
            raise ValueError("unsupported plume label")
        confidence = float(self.confidence)
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("confidence must lie in [0, 1]")
        if not isinstance(self.bounding_box, PixelBoundingBox):
            raise TypeError("bounding_box must be a PixelBoundingBox")
        if isinstance(self.source_detection_index, bool) or not isinstance(
            self.source_detection_index, int
        ):
            raise TypeError("source_detection_index must be an integer")
        if self.source_detection_index < 0:
            raise ValueError("source_detection_index cannot be negative")
        mask = np.asarray(self.mask)
        if mask.ndim != 2 or mask.dtype != np.bool_:
            raise TypeError("mask must be a two-dimensional boolean array")
        mask = np.array(mask, dtype=np.bool_, copy=True, order="C")
        mask.setflags(write=False)
        object.__setattr__(self, "plume_label", label)
        object.__setattr__(self, "source_class", source_class)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "mask", mask)

    @property
    def mask_pixel_count(self) -> int:
        return int(np.count_nonzero(self.mask))


@dataclass(frozen=True)
class PlumeSegmentationFrame:
    """Visible plume masks retained separately from thermal interpretation."""

    source_frame_id: int
    source_timestamp_seconds: float
    image_height: int
    image_width: int
    instances: tuple[VisualPlumeInstance, ...]
    model_name: str
    inference_wall_seconds: float
    evidence_status: EstimateStatus
    checkpoint_sha256: str | None = None
    model_source: str | None = None

    def __post_init__(self) -> None:
        status = EstimateStatus(self.evidence_status)
        if status is EstimateStatus.UNAVAILABLE:
            raise ValueError("an invoked segmentation frame cannot be unavailable")
        digest = self.checkpoint_sha256
        if digest is not None:
            digest = digest.casefold()
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("checkpoint_sha256 must be a hexadecimal SHA-256")
        for instance in self.instances:
            if not isinstance(instance, VisualPlumeInstance):
                raise TypeError("instances must contain VisualPlumeInstance values")
            if instance.mask.shape != (self.image_height, self.image_width):
                raise ValueError("plume masks must match the source image shape")
        object.__setattr__(self, "evidence_status", status)
        object.__setattr__(self, "checkpoint_sha256", digest)
        object.__setattr__(self, "instances", tuple(self.instances))

    @property
    def observed_plume_count(self) -> int:
        return len(self.instances)


def adapt_plume_segmentations(
    frame: SemanticDetectionFrame,
    *,
    class_map: PlumeClassMap,
    evidence_status: EstimateStatus = EstimateStatus.INFERRED,
    checkpoint_sha256: str | None = None,
    model_source: str | None = None,
) -> PlumeSegmentationFrame:
    """Filter one segmentation frame through an explicit plume vocabulary.

    Empty results mean the model was invoked but produced no accepted plume
    mask.  They are kept distinct from a missing frame; later estimators decide
    whether an empty result is validated negative evidence or must remain
    unavailable.
    """

    if not isinstance(frame, SemanticDetectionFrame):
        raise TypeError("frame must be a SemanticDetectionFrame")
    if not isinstance(class_map, PlumeClassMap):
        raise TypeError("class_map must be a PlumeClassMap")
    instances = []
    for index, detection in enumerate(frame.detections):
        target = class_map.source_to_plume_label.get(detection.semantic_class)
        if target is None:
            continue
        instances.append(
            VisualPlumeInstance(
                plume_label=target,
                source_class=detection.semantic_class,
                confidence=detection.confidence,
                bounding_box=detection.bounding_box,
                mask=detection.mask,
                source_detection_index=index,
            )
        )
    return PlumeSegmentationFrame(
        source_frame_id=frame.source_frame_id,
        source_timestamp_seconds=frame.source_timestamp_seconds,
        image_height=frame.image_height,
        image_width=frame.image_width,
        instances=tuple(instances),
        model_name=frame.model_name,
        inference_wall_seconds=frame.inference_wall_seconds,
        evidence_status=evidence_status,
        checkpoint_sha256=checkpoint_sha256,
        model_source=model_source,
    )

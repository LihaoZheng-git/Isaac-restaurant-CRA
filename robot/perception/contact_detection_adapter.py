"""Associate fixed-label contact-part masks with parent object instances."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping

import numpy as np

from robot.perception.contact_cues import DangerousPartSegmentation
from robot.perception.paco_contact_dataset import GENERAL_CONTACT_GEOMETRIES
from robot.perception.semantic_detection import SemanticDetectionFrame


@dataclass(frozen=True)
class ContactPartAssociationConfig:
    """Conservative mask-containment and ambiguity thresholds."""

    supported_geometry_classes: tuple[str, ...] = ("exposed_edge",)
    minimum_part_inside_object_fraction: float = 0.8
    minimum_best_fraction_margin: float = 0.1

    def __post_init__(self) -> None:
        classes = tuple(self.supported_geometry_classes)
        if not classes or len(classes) != len(set(classes)):
            raise ValueError("supported geometry classes must be nonempty and unique")
        if any(item not in GENERAL_CONTACT_GEOMETRIES for item in classes):
            raise ValueError("supported geometry class is outside the project taxonomy")
        for name in (
            "minimum_part_inside_object_fraction",
            "minimum_best_fraction_margin",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "supported_geometry_classes", classes)


@dataclass(frozen=True)
class ContactPartAssociationDiagnostics:
    """Association counts without implying physical contact properties."""

    part_detection_count: int
    accepted_part_count: int
    unsupported_part_indices: tuple[int, ...]
    unmatched_part_indices: tuple[int, ...]
    ambiguous_part_indices: tuple[int, ...]
    mixed_geometry_object_indices: tuple[int, ...]


@dataclass(frozen=True)
class ContactPartAssociationResult:
    """Parent-detection keyed segmentations and audit diagnostics."""

    segmentations_by_object_detection_index: Mapping[
        int, DangerousPartSegmentation
    ]
    diagnostics: ContactPartAssociationDiagnostics

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "segmentations_by_object_detection_index",
            MappingProxyType(dict(self.segmentations_by_object_detection_index)),
        )


def associate_contact_parts_to_objects(
    object_frame: SemanticDetectionFrame,
    part_frame: SemanticDetectionFrame,
    *,
    config: ContactPartAssociationConfig | None = None,
) -> ContactPartAssociationResult:
    """Attach supported part masks to same-frame object masks.

    The result keys use indices from ``object_frame`` and can therefore be
    passed directly to ``build_track_specialist_evidence`` after the object
    frame has gone through camera--LiDAR fusion. A missing part detection does
    not produce a zero-exposure record.
    """

    if not isinstance(object_frame, SemanticDetectionFrame):
        raise TypeError("object_frame must be SemanticDetectionFrame")
    if not isinstance(part_frame, SemanticDetectionFrame):
        raise TypeError("part_frame must be SemanticDetectionFrame")
    _validate_aligned_frames(object_frame, part_frame)
    settings = config or ContactPartAssociationConfig()
    supported = set(settings.supported_geometry_classes)
    unsupported: list[int] = []
    unmatched: list[int] = []
    ambiguous: list[int] = []
    accepted: dict[int, list[int]] = {}

    for part_index, part in enumerate(part_frame.detections):
        if part.semantic_class not in supported:
            unsupported.append(part_index)
            continue
        part_count = part.mask_pixel_count
        if part_count == 0:
            unmatched.append(part_index)
            continue
        candidates = []
        for object_index, parent in enumerate(object_frame.detections):
            inside = int(np.count_nonzero(part.mask & parent.mask))
            fraction = inside / part_count
            if fraction >= settings.minimum_part_inside_object_fraction:
                candidates.append((fraction, object_index))
        candidates.sort(reverse=True)
        if not candidates:
            unmatched.append(part_index)
            continue
        if (
            len(candidates) > 1
            and candidates[0][0] - candidates[1][0]
            < settings.minimum_best_fraction_margin
        ):
            ambiguous.append(part_index)
            continue
        accepted.setdefault(candidates[0][1], []).append(part_index)

    segmentations: dict[int, DangerousPartSegmentation] = {}
    mixed: list[int] = []
    accepted_count = 0
    for object_index, part_indices in sorted(accepted.items()):
        geometry_classes = {
            part_frame.detections[index].semantic_class for index in part_indices
        }
        if len(geometry_classes) != 1:
            mixed.append(object_index)
            continue
        parent = object_frame.detections[object_index]
        danger_mask = np.zeros_like(parent.mask)
        confidences = []
        for part_index in part_indices:
            part = part_frame.detections[part_index]
            danger_mask |= part.mask & parent.mask
            confidences.append(part.confidence)
        if not np.any(danger_mask):
            continue
        accepted_count += len(part_indices)
        segmentations[object_index] = DangerousPartSegmentation(
            object_mask=parent.mask,
            dangerous_part_mask=danger_mask,
            dangerous_part_label=next(iter(geometry_classes)),
            confidence=min(confidences),
            timestamp_seconds=part_frame.source_timestamp_seconds,
            model_id=part_frame.model_name,
        )

    return ContactPartAssociationResult(
        segmentations_by_object_detection_index=segmentations,
        diagnostics=ContactPartAssociationDiagnostics(
            part_detection_count=len(part_frame.detections),
            accepted_part_count=accepted_count,
            unsupported_part_indices=tuple(unsupported),
            unmatched_part_indices=tuple(unmatched),
            ambiguous_part_indices=tuple(ambiguous),
            mixed_geometry_object_indices=tuple(mixed),
        ),
    )


def _validate_aligned_frames(
    object_frame: SemanticDetectionFrame,
    part_frame: SemanticDetectionFrame,
) -> None:
    if object_frame.source_frame_id != part_frame.source_frame_id:
        raise ValueError("object and part frame IDs must match")
    if not math.isclose(
        object_frame.source_timestamp_seconds,
        part_frame.source_timestamp_seconds,
        abs_tol=1e-9,
    ):
        raise ValueError("object and part timestamps must match")
    if (
        object_frame.image_height != part_frame.image_height
        or object_frame.image_width != part_frame.image_width
    ):
        raise ValueError("object and part image dimensions must match")

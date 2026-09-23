"""Adapt fixed-label detector outputs into explicit thermal cue frames."""

from __future__ import annotations

from typing import Iterable

from robot.perception.semantic_detection import (
    SemanticDetectionFrame,
    normalize_detection_class,
)
from robot.perception.thermal_cues import ThermalCueFrame


def thermal_cue_frame_from_detections(
    frame: SemanticDetectionFrame,
    *,
    flame_classes: Iterable[str] = ("fire", "flame"),
    smoke_classes: Iterable[str] = ("smoke",),
) -> ThermalCueFrame:
    """Extract flame and smoke scores while leaving steam unsupported."""
    if not isinstance(frame, SemanticDetectionFrame):
        raise TypeError("frame must be a SemanticDetectionFrame")
    flames = {normalize_detection_class(item) for item in flame_classes}
    smoke = {normalize_detection_class(item) for item in smoke_classes}
    if not flames or not smoke:
        raise ValueError("flame and smoke class sets must be nonempty")
    overlap = flames & smoke
    if overlap:
        raise ValueError(f"flame and smoke classes overlap: {sorted(overlap)}")

    flame_score = max(
        (
            detection.confidence
            for detection in frame.detections
            if detection.semantic_class in flames
        ),
        default=0.0,
    )
    smoke_score = max(
        (
            detection.confidence
            for detection in frame.detections
            if detection.semantic_class in smoke
        ),
        default=0.0,
    )
    return ThermalCueFrame(
        timestamp_seconds=frame.source_timestamp_seconds,
        flame_likelihood=flame_score,
        steam_likelihood=None,
        smoke_likelihood=smoke_score,
        model_id=frame.model_name,
    )

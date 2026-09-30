"""Continuous temporal features from visible plume segmentation masks.

The features describe image evidence only.  They do not estimate temperature
or thermal danger, and segmentation confidence is retained separately from
feature magnitude.  Empty model results count as negative observations only
when that behavior has been validated explicitly for the backend and domain.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from statistics import median

from robot.perception.physical_attributes import (
    EstimateStatus,
    ScalarEstimate,
    combine_estimate_statuses,
)
from robot.perception.plume_segmentation import PlumeSegmentationFrame
from robot.perception.semantic_detection import PixelBoundingBox


PLUME_FEATURE_SOURCE = "continuous_visual_plume_features"


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class PlumeSourceObservation:
    """One invoked segmentation frame relative to one candidate source."""

    frame: PlumeSegmentationFrame
    source_box: PixelBoundingBox
    source_context_likelihood: ScalarEstimate
    spatial_association_likelihood: ScalarEstimate
    selected_instance_index: int | None
    negative_observation_validated: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.frame, PlumeSegmentationFrame):
            raise TypeError("frame must be a PlumeSegmentationFrame")
        if not isinstance(self.source_box, PixelBoundingBox):
            raise TypeError("source_box must be a PixelBoundingBox")
        if (
            self.source_box.x_max > self.frame.image_width
            or self.source_box.y_max > self.frame.image_height
        ):
            raise ValueError("source_box must lie inside the source image")
        for name in (
            "source_context_likelihood",
            "spatial_association_likelihood",
        ):
            item = getattr(self, name)
            if not isinstance(item, ScalarEstimate) or item.unit != "1":
                raise TypeError(f"{name} must be a dimensionless ScalarEstimate")
            if item.is_available and not 0 <= item.lower_bound <= item.upper_bound <= 1:
                raise ValueError(f"{name} bounds must lie in [0, 1]")
        index = self.selected_instance_index
        if index is not None:
            if isinstance(index, bool) or not isinstance(index, Integral):
                raise TypeError("selected_instance_index must be an integer or None")
            index = int(index)
            if not 0 <= index < len(self.frame.instances):
                raise ValueError("selected_instance_index is outside the frame")
            object.__setattr__(self, "selected_instance_index", index)
        if not isinstance(self.negative_observation_validated, bool):
            raise TypeError("negative_observation_validated must be boolean")

    @property
    def selected_instance(self):
        if self.selected_instance_index is None:
            return None
        return self.frame.instances[self.selected_instance_index]


@dataclass(frozen=True)
class PlumeFeatureConfig:
    """Temporal window and descriptive feature uncertainty."""

    minimum_frame_count: int = 3
    maximum_evidence_age_seconds: float = 1.0
    relative_measurement_uncertainty: float = 0.15

    def __post_init__(self) -> None:
        if isinstance(self.minimum_frame_count, bool) or not isinstance(
            self.minimum_frame_count, Integral
        ):
            raise TypeError("minimum_frame_count must be an integer")
        if self.minimum_frame_count <= 0:
            raise ValueError("minimum_frame_count must be positive")
        age = _finite(
            self.maximum_evidence_age_seconds,
            "maximum_evidence_age_seconds",
        )
        uncertainty = _finite(
            self.relative_measurement_uncertainty,
            "relative_measurement_uncertainty",
        )
        if age <= 0:
            raise ValueError("maximum_evidence_age_seconds must be positive")
        if not 0 <= uncertainty <= 1:
            raise ValueError("relative_measurement_uncertainty must lie in [0, 1]")
        object.__setattr__(self, "minimum_frame_count", int(self.minimum_frame_count))
        object.__setattr__(self, "maximum_evidence_age_seconds", age)
        object.__setattr__(self, "relative_measurement_uncertainty", uncertainty)


@dataclass(frozen=True)
class ContinuousPlumeFeatures:
    """Auditable visual plume features for a single candidate source."""

    relative_mask_area: ScalarEstimate
    vertical_extent: ScalarEstimate
    persistence: ScalarEstimate
    source_context_likelihood: ScalarEstimate
    spatial_association_likelihood: ScalarEstimate
    segmentation_confidence: ScalarEstimate
    invoked_frame_count: int
    positive_frame_count: int
    rejection_reasons: tuple[str, ...]


def _unavailable(timestamp: float, feature: str, *evidence: str) -> ScalarEstimate:
    return ScalarEstimate.unavailable(
        unit="1",
        source=f"{PLUME_FEATURE_SOURCE}:{feature}",
        timestamp_seconds=timestamp,
        evidence=evidence,
    )


def _estimate(
    value: float,
    *,
    timestamp: float,
    feature: str,
    confidence: float,
    status: EstimateStatus,
    uncertainty: float,
    bounded: bool,
    evidence: tuple[str, ...],
) -> ScalarEstimate:
    width = uncertainty * max(value, 0.1)
    upper = value + width
    if bounded:
        upper = min(1.0, upper)
    return ScalarEstimate(
        value=value,
        lower_bound=max(0.0, value - width),
        upper_bound=upper,
        confidence=confidence,
        unit="1",
        source=f"{PLUME_FEATURE_SOURCE}:{feature}",
        timestamp_seconds=timestamp,
        status=status,
        evidence=evidence,
    )


def _aggregate_input_cue(
    observations: tuple[PlumeSourceObservation, ...],
    name: str,
    *,
    timestamp: float,
    uncertainty: float,
) -> ScalarEstimate:
    items = tuple(getattr(item, name) for item in observations)
    if any(not item.is_available for item in items):
        return _unavailable(
            timestamp,
            name,
            f"{name} unavailable in at least one invoked frame",
        )
    values = [item.value for item in items]
    assert all(value is not None for value in values)
    value = float(median(values))
    return _estimate(
        value,
        timestamp=timestamp,
        feature=name,
        confidence=min(item.confidence for item in items),
        status=combine_estimate_statuses(item.status for item in items),
        uncertainty=uncertainty,
        bounded=True,
        evidence=(
            f"temporal median across {len(items)} invoked frames",
            "context and spatial relation remain separate from plume appearance",
        ),
    )


def extract_continuous_plume_features(
    observations: tuple[PlumeSourceObservation, ...],
    *,
    timestamp_seconds: float,
    config: PlumeFeatureConfig | None = None,
) -> ContinuousPlumeFeatures:
    """Extract mask geometry and persistence without predicting thermal risk."""

    timestamp = _finite(timestamp_seconds, "timestamp_seconds")
    if timestamp < 0:
        raise ValueError("timestamp_seconds cannot be negative")
    settings = config or PlumeFeatureConfig()
    if not isinstance(settings, PlumeFeatureConfig):
        raise TypeError("config must be a PlumeFeatureConfig")
    try:
        items = tuple(observations)
    except TypeError as error:
        raise TypeError("observations must be iterable") from error
    if any(not isinstance(item, PlumeSourceObservation) for item in items):
        raise TypeError("observations must contain PlumeSourceObservation values")
    if not items:
        unavailable = _unavailable(timestamp, "no_frames", "no invoked frames")
        return ContinuousPlumeFeatures(
            unavailable,
            unavailable,
            unavailable,
            unavailable,
            unavailable,
            unavailable,
            0,
            0,
            ("no invoked frames",),
        )
    ordered = tuple(sorted(items, key=lambda item: item.frame.source_timestamp_seconds))
    frame_ids = [item.frame.source_frame_id for item in ordered]
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError("source frame IDs must be unique")
    future = [
        item
        for item in ordered
        if item.frame.source_timestamp_seconds > timestamp
    ]
    stale = [
        item
        for item in ordered
        if timestamp - item.frame.source_timestamp_seconds
        > settings.maximum_evidence_age_seconds
    ]
    reasons = []
    if future:
        reasons.append("future segmentation frame")
    if stale:
        reasons.append("stale segmentation frame")
    if reasons:
        unavailable = _unavailable(timestamp, "frame_window", *reasons)
        return ContinuousPlumeFeatures(
            unavailable,
            unavailable,
            unavailable,
            unavailable,
            unavailable,
            unavailable,
            len(ordered),
            0,
            tuple(reasons),
        )

    positive = tuple(item for item in ordered if item.selected_instance is not None)
    statuses = [item.frame.evidence_status for item in positive]
    confidences = [item.selected_instance.confidence for item in positive]
    area_ratios = [
        item.selected_instance.mask_pixel_count / item.source_box.area
        for item in positive
    ]
    extent_ratios = [
        item.selected_instance.bounding_box.height / item.source_box.height
        for item in positive
    ]
    geometry_evidence = (
        f"positive_frames={len(positive)}/{len(ordered)}",
        "mask area and vertical extent are normalized by the candidate source box",
        "features are visual descriptors, not temperature or burn severity",
    )
    if positive:
        status = combine_estimate_statuses(statuses)
        confidence = min(confidences)
        relative_area = _estimate(
            float(median(area_ratios)),
            timestamp=timestamp,
            feature="relative_mask_area",
            confidence=confidence,
            status=status,
            uncertainty=settings.relative_measurement_uncertainty,
            bounded=False,
            evidence=geometry_evidence,
        )
        vertical_extent = _estimate(
            float(median(extent_ratios)),
            timestamp=timestamp,
            feature="vertical_extent",
            confidence=confidence,
            status=status,
            uncertainty=settings.relative_measurement_uncertainty,
            bounded=False,
            evidence=geometry_evidence,
        )
        model_confidence = _estimate(
            float(median(confidences)),
            timestamp=timestamp,
            feature="segmentation_confidence",
            confidence=confidence,
            status=status,
            uncertainty=0.0,
            bounded=True,
            evidence=(
                "model confidence is epistemic confidence, not plume strength",
            ),
        )
    else:
        relative_area = _unavailable(
            timestamp, "relative_mask_area", "no associated plume mask"
        )
        vertical_extent = _unavailable(
            timestamp, "vertical_extent", "no associated plume mask"
        )
        model_confidence = _unavailable(
            timestamp, "segmentation_confidence", "no associated plume mask"
        )

    persistence = _unavailable(
        timestamp,
        "persistence",
        "insufficient or unvalidated negative observations",
    )
    if len(ordered) < settings.minimum_frame_count:
        reasons.append(
            f"persistence requires {settings.minimum_frame_count} invoked frames"
        )
    elif any(
        item.selected_instance is None and not item.negative_observation_validated
        for item in ordered
    ):
        reasons.append("empty masks are not validated negative evidence")
    else:
        value = len(positive) / len(ordered)
        status = combine_estimate_statuses(
            item.frame.evidence_status for item in ordered
        )
        persistence = _estimate(
            value,
            timestamp=timestamp,
            feature="persistence",
            confidence=min(confidences) if confidences else 0.0,
            status=status,
            uncertainty=settings.relative_measurement_uncertainty,
            bounded=True,
            evidence=(
                f"associated plume masks in {len(positive)} of {len(ordered)} frames",
                "empty frames counted only after explicit negative-evidence validation",
            ),
        )

    context = _aggregate_input_cue(
        ordered,
        "source_context_likelihood",
        timestamp=timestamp,
        uncertainty=settings.relative_measurement_uncertainty,
    )
    association = _aggregate_input_cue(
        ordered,
        "spatial_association_likelihood",
        timestamp=timestamp,
        uncertainty=settings.relative_measurement_uncertainty,
    )
    return ContinuousPlumeFeatures(
        relative_mask_area=relative_area,
        vertical_extent=vertical_extent,
        persistence=persistence,
        source_context_likelihood=context,
        spatial_association_likelihood=association,
        segmentation_confidence=model_confidence,
        invoked_frame_count=len(ordered),
        positive_frame_count=len(positive),
        rejection_reasons=tuple(reasons),
    )

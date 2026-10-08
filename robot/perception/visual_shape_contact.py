"""Continuous *visible silhouette* contact cues, independent of class names.

A sufficiently resolved round silhouette supplies positive low edge/tip cue
evidence. This does not establish physical smoothness, hardness, hidden-surface
safety, or injury severity. Border truncation and fragmented masks are rejected;
unrecognized occlusion and interior hazards remain limitations of this 2D proxy.
"""

from dataclasses import dataclass
import math

import cv2
import numpy as np

from robot.perception.physical_attributes import (
    EstimateStatus,
    ScalarEstimate,
    combine_estimate_statuses,
)
from robot.perception.semantic_detection import SemanticDetection


SHAPE_CONTACT_SOURCE = "visual_silhouette_contact_proxy"
SHAPE_CONTACT_BASIS = "visual_silhouette_geometry_proxy"


@dataclass(frozen=True)
class SilhouetteContactConfig:
    minimum_pixels: int = 200
    minimum_detection_confidence: float = 0.5
    minimum_component_fraction: float = 0.98
    border_margin_pixels: int = 2
    polygon_tolerance_fraction: float = 0.01
    confidence_cap: float = 0.8
    uncertainty_floor: float = 0.05

    def __post_init__(self):
        for name in ("minimum_pixels", "border_margin_pixels"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "minimum_detection_confidence",
            "minimum_component_fraction",
            "polygon_tolerance_fraction",
            "confidence_cap",
            "uncertainty_floor",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not math.isfinite(value)
                or not 0 < value <= 1
            ):
                raise ValueError(f"{name} must lie in (0, 1]")
        if self.polygon_tolerance_fraction > 0.05:
            raise ValueError("polygon tolerance must not erase large features")


def estimate_silhouette_contact_proxy(
    detection: SemanticDetection,
    *,
    timestamp_seconds: float,
    config: SilhouetteContactConfig | None = None,
    evidence_status: EstimateStatus = EstimateStatus.INFERRED,
) -> ScalarEstimate:
    """Score observed outline irregularity/elongation/corners continuously.

    Maximum of circularity deficit, PCA elongation, concavity and large polygon
    turns. Weights/thresholds and intervals are engineering assumptions, not a
    trained or calibrated injury model. A class name never changes the result.
    """
    if not isinstance(detection, SemanticDetection):
        raise TypeError("detection must be SemanticDetection")
    settings = config or SilhouetteContactConfig()
    if not isinstance(settings, SilhouetteContactConfig):
        raise TypeError("config must be SilhouetteContactConfig")
    timestamp = float(timestamp_seconds)
    if not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError("timestamp_seconds must be finite and nonnegative")
    status = EstimateStatus(evidence_status)
    if status is EstimateStatus.UNAVAILABLE:
        raise ValueError("observed masks cannot have unavailable status")

    def unknown(reason):
        return ScalarEstimate.unavailable(
            unit="1",
            source=SHAPE_CONTACT_SOURCE,
            timestamp_seconds=timestamp,
            evidence=(reason,),
        )

    mask = detection.mask
    count = int(mask.sum())
    if count < settings.minimum_pixels:
        return unknown("insufficient resolved silhouette pixels")
    if detection.confidence < settings.minimum_detection_confidence:
        return unknown("object detection confidence below acceptance threshold")
    margin = settings.border_margin_pixels
    if (
        mask[:margin].any()
        or mask[-margin:].any()
        or mask[:, :margin].any()
        or mask[:, -margin:].any()
    ):
        return unknown("silhouette touches image border; object may be truncated")
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    fraction = float(stats[largest, cv2.CC_STAT_AREA]) / count
    if fraction < settings.minimum_component_fraction:
        return unknown("fragmented silhouette; no sufficiently dominant component")
    component = (labels == largest).astype(np.uint8)
    contours, _ = cv2.findContours(component, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 1:
        return unknown("silhouette contains holes or unresolved interior boundaries")
    contour = contours[0]
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if area <= 0 or perimeter <= 0:
        return unknown("degenerate silhouette")
    circularity = min(1.0, 4 * math.pi * area / perimeter**2)
    eigenvalues = np.linalg.eigvalsh(np.cov(np.argwhere(component), rowvar=False))
    elongation = 1 - math.sqrt(max(0.0, eigenvalues[0]) / max(eigenvalues[-1], 1e-12))
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    concavity = 1 - area / max(hull_area, area)
    polygon = cv2.approxPolyDP(
        contour, settings.polygon_tolerance_fraction * perimeter, True
    ).reshape(-1, 2)
    if len(polygon) < 3:
        return unknown("silhouette has too few resolved polygon vertices")
    previous = np.roll(polygon, 1, axis=0) - polygon
    following = np.roll(polygon, -1, axis=0) - polygon
    cosine = np.sum(previous * following, axis=1) / (
        np.linalg.norm(previous, axis=1) * np.linalg.norm(following, axis=1)
    )
    maximum_turn = math.pi - float(np.arccos(np.clip(cosine, -1, 1)).min())
    corner_cue = max(0.0, (maximum_turn - math.pi / 6) / (5 * math.pi / 6))
    value = min(1.0, max(1 - circularity, elongation, concavity, corner_cue))
    confidence = min(detection.confidence * fraction, settings.confidence_cap)
    uncertainty = settings.uncertainty_floor + 0.2 * (1 - confidence) + (1 - fraction)
    return ScalarEstimate(
        value,
        max(0.0, value - uncertainty),
        min(1.0, value + uncertainty),
        confidence,
        "1",
        SHAPE_CONTACT_SOURCE,
        timestamp,
        status,
        (
            f"resolved_pixels={count}; dominant_component_fraction={fraction:.6g}",
            f"circularity={circularity:.6g}; elongation={elongation:.6g}; concavity={concavity:.6g}; maximum_turn_rad={maximum_turn:.6g}",
            "score=max(circularity deficit, elongation, concavity, large-corner cue)",
            "2D visible-outline proxy only; hidden surfaces, material hardness, edge radius and physical sharpness unavailable",
            "segmentation confidence is not a guarantee of complete visibility; occlusion can be missed",
        ),
    )


def combine_shape_and_part_contact(
    shape: ScalarEstimate,
    part: ScalarEstimate | None,
) -> ScalarEstimate:
    """Keep the greater accepted visual cue; never suppress a detected edge.

    Missing shape evidence leaves a supplied specialist unchanged. Missing
    specialist evidence does not invalidate a positively observed silhouette.
    """
    for item in (shape, part):
        if item is not None and (
            not isinstance(item, ScalarEstimate) or item.unit != "1"
        ):
            raise ValueError("contact proxies must be dimensionless ScalarEstimate")
        if (
            item is not None
            and item.is_available
            and not 0 <= item.lower_bound <= item.upper_bound <= 1
        ):
            raise ValueError("contact proxy bounds must lie in [0, 1]")
    if part is None or not part.is_available:
        return shape
    if not shape.is_available:
        return part
    return ScalarEstimate(
        *(
            max(getattr(shape, name), getattr(part, name))
            for name in ("value", "lower_bound", "upper_bound")
        ),
        min(shape.confidence, part.confidence),
        "1",
        SHAPE_CONTACT_SOURCE,
        min(shape.timestamp_seconds, part.timestamp_seconds),
        combine_estimate_statuses((shape.status, part.status)),
        (
            f"maximum of {shape.source} and {part.source}",
            *shape.evidence,
            *part.evidence,
        ),
    )

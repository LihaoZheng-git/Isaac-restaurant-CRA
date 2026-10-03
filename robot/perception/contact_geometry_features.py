"""Continuous visual geometry descriptors for potentially harmful parts.

The descriptors are measurable from a part mask and support the project's
general edge/tip/protrusion/jagged taxonomy.  They are not physical sharpness,
material hardness, contact probability, or injury severity.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from robot.perception.contact_cues import DangerousPartSegmentation
from robot.perception.paco_contact_dataset import GENERAL_CONTACT_GEOMETRIES
from robot.perception.physical_attributes import (
    CategoricalEstimate,
    EstimateStatus,
    ScalarEstimate,
    combine_estimate_statuses,
)


CONTACT_GEOMETRY_FEATURE_SOURCE = "visual_contact_geometry_features"


@dataclass(frozen=True)
class ContactGeometryFeatures:
    """Mask-derived descriptors with explicit physical-property limits."""

    geometry_class: CategoricalEstimate
    exposure_fraction: ScalarEstimate
    elongation: ScalarEstimate
    boundary_complexity: ScalarEstimate
    sharpness: ScalarEstimate
    hardness: ScalarEstimate
    orientation_toward_route: ScalarEstimate


def _unavailable(timestamp: float, name: str, reason: str) -> ScalarEstimate:
    return ScalarEstimate.unavailable(
        unit="1",
        source=f"{CONTACT_GEOMETRY_FEATURE_SOURCE}:{name}",
        timestamp_seconds=timestamp,
        evidence=(reason,),
    )


def _estimate(
    value: float,
    *,
    timestamp: float,
    name: str,
    confidence: float,
    status: EstimateStatus,
    evidence: tuple[str, ...],
) -> ScalarEstimate:
    uncertainty = min(0.25, (1.0 - confidence) * 0.5)
    return ScalarEstimate(
        value=value,
        lower_bound=max(0.0, value - uncertainty),
        upper_bound=min(1.0, value + uncertainty),
        confidence=confidence,
        unit="1",
        source=f"{CONTACT_GEOMETRY_FEATURE_SOURCE}:{name}",
        timestamp_seconds=timestamp,
        status=status,
        evidence=evidence,
    )


def _mask_perimeter(mask: np.ndarray) -> int:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    horizontal = np.count_nonzero(padded[:, 1:] != padded[:, :-1])
    vertical = np.count_nonzero(padded[1:, :] != padded[:-1, :])
    return int(horizontal + vertical)


def _elongation(mask: np.ndarray) -> float:
    coordinates = np.argwhere(mask).astype(float)
    if len(coordinates) < 2:
        return 0.0
    covariance = np.cov(coordinates, rowvar=False, bias=True)
    eigenvalues = np.linalg.eigvalsh(covariance)
    major = max(0.0, float(eigenvalues[-1]))
    minor = max(0.0, float(eigenvalues[0]))
    if major <= 1e-12:
        return 0.0
    return min(1.0, max(0.0, 1.0 - math.sqrt(minor / major)))


def estimate_contact_geometry_features(
    segmentation: DangerousPartSegmentation,
    *,
    timestamp_seconds: float,
    evidence_status: EstimateStatus = EstimateStatus.INFERRED,
) -> ContactGeometryFeatures:
    """Extract resolution-aware shape descriptors from one accepted part mask."""

    if not isinstance(segmentation, DangerousPartSegmentation):
        raise TypeError("segmentation must be a DangerousPartSegmentation")
    timestamp = float(timestamp_seconds)
    if not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError("timestamp_seconds must be finite and nonnegative")
    if segmentation.timestamp_seconds > timestamp:
        raise ValueError("segmentation cannot be newer than result")
    evidence_timestamp = float(segmentation.timestamp_seconds)
    status = EstimateStatus(evidence_status)
    if status is EstimateStatus.UNAVAILABLE:
        raise ValueError("an observed segmentation cannot have unavailable status")
    label = "_".join(
        str(segmentation.dangerous_part_label)
        .casefold()
        .replace("-", " ")
        .split()
    )
    if label not in GENERAL_CONTACT_GEOMETRIES:
        raise ValueError("dangerous-part label is outside the general taxonomy")
    confidence = float(segmentation.confidence)
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("segmentation confidence must lie in [0, 1]")
    object_mask = np.asarray(segmentation.object_mask)
    part_mask = np.asarray(segmentation.dangerous_part_mask)
    if (
        object_mask.dtype != np.bool_
        or part_mask.dtype != np.bool_
        or object_mask.ndim != 2
        or object_mask.shape != part_mask.shape
    ):
        raise ValueError("contact masks must be aligned two-dimensional booleans")
    object_count = int(np.count_nonzero(object_mask))
    part_count = int(np.count_nonzero(part_mask))
    if object_count == 0 or part_count == 0:
        raise ValueError("object and dangerous-part masks must be nonempty")
    if np.any(part_mask & ~object_mask):
        raise ValueError("dangerous-part mask must lie inside object mask")

    perimeter = _mask_perimeter(part_mask)
    compactness = 4.0 * math.pi * part_count / max(float(perimeter * perimeter), 1.0)
    complexity = 1.0 - min(1.0, max(0.0, compactness))
    elongation = _elongation(part_mask)
    exposure = part_count / object_count
    common_evidence = (
        f"model_id={segmentation.model_id}",
        f"geometry_class={label}",
        f"part_pixels={part_count}; object_pixels={object_count}; perimeter={perimeter}",
        "mask geometry is not physical sharpness, hardness, or injury severity",
    )
    return ContactGeometryFeatures(
        geometry_class=CategoricalEstimate(
            value=label,
            confidence=confidence,
            source=CONTACT_GEOMETRY_FEATURE_SOURCE,
            timestamp_seconds=evidence_timestamp,
            status=status,
            evidence=common_evidence,
        ),
        exposure_fraction=_estimate(
            exposure,
            timestamp=evidence_timestamp,
            name="exposure_fraction",
            confidence=confidence,
            status=status,
            evidence=common_evidence,
        ),
        elongation=_estimate(
            elongation,
            timestamp=evidence_timestamp,
            name="elongation",
            confidence=confidence,
            status=status,
            evidence=common_evidence
            + ("principal-axis elongation is a 2D visual descriptor",),
        ),
        boundary_complexity=_estimate(
            complexity,
            timestamp=evidence_timestamp,
            name="boundary_complexity",
            confidence=confidence,
            status=status,
            evidence=common_evidence
            + ("isoperimetric boundary complexity is resolution-sensitive",),
        ),
        sharpness=_unavailable(
            timestamp,
            "sharpness",
            "2D mask geometry cannot establish edge radius or physical sharpness",
        ),
        hardness=_unavailable(
            timestamp,
            "hardness",
            "visual geometry cannot establish material hardness",
        ),
        orientation_toward_route=_unavailable(
            timestamp,
            "orientation_toward_route",
            "route orientation requires calibrated 3D geometry",
        ),
    )


def estimate_exposed_edge_proxy(
    features: ContactGeometryFeatures, *, elongation_weight: float = 0.6,
) -> ScalarEstimate:
    """Continuous visible-edge proxy, never physical sharpness or hardness.

    Exposure scales a weighted mask-shape descriptor (0.6 elongation, 0.4
    boundary complexity). The fixed weights are engineering assumptions,
    not a calibrated injury model. Unsupported geometry remains unavailable.
    """
    source = "visual_exposed_edge_geometry_proxy"
    if not math.isfinite(elongation_weight) or not 0 <= elongation_weight <= 1:
        raise ValueError("elongation_weight must lie in [0, 1]")
    items = (features.exposure_fraction, features.elongation, features.boundary_complexity)
    timestamp = min(item.timestamp_seconds for item in items)
    if features.geometry_class.value != "exposed_edge" or any(not item.is_available for item in items):
        return ScalarEstimate.unavailable(
            unit="1", source=source, timestamp_seconds=timestamp,
            evidence=("only accepted exposed-edge masks support this bounded proxy",),
        )
    values = [
        getattr(items[0], bound) * (elongation_weight * getattr(items[1], bound)
                                  + (1 - elongation_weight) * getattr(items[2], bound))
        for bound in ("value", "lower_bound", "upper_bound")
    ]
    return ScalarEstimate(
        *values, min(item.confidence for item in items), "1", source, timestamp,
        combine_estimate_statuses(item.status for item in items),
        tuple(f"{item.source}: value={item.value}; bounds=({item.lower_bound},{item.upper_bound}); status={item.status.value}" for item in items)
        + (f"exposure * ({elongation_weight} elongation + {1-elongation_weight} boundary complexity)",
           "2D visual proxy; sharpness, hardness, edge radius and 3D route orientation remain unknown"),
    )

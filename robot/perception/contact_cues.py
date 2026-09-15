"""Build contact evidence from fixed-label visual specialist outputs."""

from dataclasses import dataclass
import math

import numpy as np

from robot.perception.physical_attributes import (
    CategoricalEstimate,
    ContactAttributes,
    EstimateStatus,
    ScalarEstimate,
)


CONTACT_MASK_SOURCE = "fixed_label_dangerous_part_segmentation"


@dataclass(frozen=True)
class DangerousPartSegmentation:
    """One object mask and a specialist blade/tip/tine mask."""

    object_mask: np.ndarray
    dangerous_part_mask: np.ndarray
    dangerous_part_label: str
    confidence: float
    timestamp_seconds: float
    model_id: str


def estimate_contact_cues(
    segmentation: DangerousPartSegmentation | None,
    *,
    sharpness: ScalarEstimate | None = None,
    hardness: ScalarEstimate | None = None,
    orientation_toward_route: ScalarEstimate | None = None,
    timestamp_seconds: float,
) -> ContactAttributes:
    """Compute part exposure; keep other specialist evidence independent."""
    timestamp = float(timestamp_seconds)

    def unavailable(source: str) -> ScalarEstimate:
        return ScalarEstimate.unavailable(
            unit="1", source=source, timestamp_seconds=timestamp
        )

    part = CategoricalEstimate.unavailable(
        source=CONTACT_MASK_SOURCE, timestamp_seconds=timestamp
    )
    exposure = unavailable(CONTACT_MASK_SOURCE)
    if segmentation is not None:
        if not isinstance(segmentation, DangerousPartSegmentation):
            raise TypeError("segmentation must be DangerousPartSegmentation")
        object_mask = np.asarray(segmentation.object_mask)
        danger_mask = np.asarray(segmentation.dangerous_part_mask)
        if object_mask.dtype != np.bool_ or danger_mask.dtype != np.bool_:
            raise ValueError("contact masks must be boolean")
        if object_mask.shape != danger_mask.shape or object_mask.ndim != 2:
            raise ValueError("contact masks must be aligned 2D arrays")
        object_count = int(object_mask.sum())
        danger_count = int(danger_mask.sum())
        if object_count == 0:
            raise ValueError("object mask cannot be empty")
        if np.any(danger_mask & ~object_mask):
            raise ValueError("dangerous-part mask must lie inside object mask")
        confidence = float(segmentation.confidence)
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("segmentation confidence must lie in [0, 1]")
        if segmentation.timestamp_seconds > timestamp:
            raise ValueError("segmentation cannot be newer than result")
        ratio = danger_count / object_count
        uncertainty = min(0.25, (1 - confidence) * 0.5)
        exposure = ScalarEstimate(
            ratio,
            max(0, ratio - uncertainty),
            min(1, ratio + uncertainty),
            confidence,
            "1",
            CONTACT_MASK_SOURCE,
            timestamp,
            EstimateStatus.INFERRED,
            (
                f"model_id={segmentation.model_id}",
                f"dangerous pixels={danger_count}; object pixels={object_count}",
                "pixel exposure is not physical sharpness or contact probability",
            ),
        )
        part = CategoricalEstimate(
            segmentation.dangerous_part_label,
            confidence,
            CONTACT_MASK_SOURCE,
            timestamp,
            EstimateStatus.INFERRED,
            (f"model_id={segmentation.model_id}",),
        )
    supplied = {
        "sharpness": sharpness or unavailable("sharpness_specialist_unavailable"),
        "hardness": hardness or unavailable("material_specialist_unavailable"),
        "orientation": orientation_toward_route
        or unavailable("calibrated_part_orientation_unavailable"),
    }
    for name, item in supplied.items():
        if not isinstance(item, ScalarEstimate) or item.unit != "1":
            raise ValueError(f"{name} must be a dimensionless ScalarEstimate")
        if item.timestamp_seconds > timestamp:
            raise ValueError(f"{name} cannot be newer than result")
    return ContactAttributes(
        sharpness=supplied["sharpness"],
        hardness=supplied["hardness"],
        dangerous_part_exposure=exposure,
        orientation_toward_route=supplied["orientation"],
        dangerous_part=part,
    )

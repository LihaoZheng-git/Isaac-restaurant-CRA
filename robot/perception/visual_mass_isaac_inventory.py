"""Controlled Isaac visual-mass inventory and segmentation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class IsaacMassObject:
    """One independent synthetic object with explicit offline oracle labels."""

    object_id: str
    semantic_class: str
    dimensions_m: tuple[float, float, float]
    authored_mass_kg: float
    color: tuple[float, float, float]
    shape: str

    def __post_init__(self) -> None:
        if not self.object_id or not self.object_id.replace("-", "").isalnum():
            raise ValueError("object_id must contain letters, digits, or hyphens")
        if self.semantic_class not in {"bottle", "bowl", "cup", "pan", "plate"}:
            raise ValueError("unsupported fixture class")
        if self.shape not in {"cube", "cylinder"}:
            raise ValueError("shape must be cube or cylinder")
        values = (*self.dimensions_m, self.authored_mass_kg, *self.color)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("fixture values must be finite")
        if any(value <= 0 for value in self.dimensions_m) or self.authored_mass_kg <= 0:
            raise ValueError("dimensions and mass must be positive")
        if any(channel < 0 or channel > 1 for channel in self.color):
            raise ValueError("color channels must be in [0, 1]")


ISAAC_MASS_TEST_INVENTORY = (
    IsaacMassObject("bottle-01", "bottle", (0.10, 0.10, 0.27), 0.38, (0.15, 0.55, 0.82), "cylinder"),
    IsaacMassObject("bottle-02", "bottle", (0.13, 0.13, 0.32), 0.72, (0.18, 0.72, 0.38), "cylinder"),
    IsaacMassObject("bowl-01", "bowl", (0.24, 0.24, 0.10), 0.44, (0.82, 0.30, 0.12), "cylinder"),
    IsaacMassObject("bowl-02", "bowl", (0.31, 0.31, 0.13), 0.86, (0.25, 0.42, 0.85), "cylinder"),
    IsaacMassObject("cup-01", "cup", (0.09, 0.09, 0.12), 0.25, (0.88, 0.78, 0.18), "cylinder"),
    IsaacMassObject("cup-02", "cup", (0.12, 0.12, 0.16), 0.48, (0.58, 0.20, 0.72), "cylinder"),
    IsaacMassObject("pan-01", "pan", (0.34, 0.25, 0.07), 0.91, (0.16, 0.18, 0.20), "cube"),
    IsaacMassObject("pan-02", "pan", (0.42, 0.30, 0.09), 1.42, (0.34, 0.35, 0.37), "cube"),
    IsaacMassObject("plate-01", "plate", (0.25, 0.25, 0.035), 0.51, (0.90, 0.90, 0.86), "cylinder"),
    IsaacMassObject("plate-02", "plate", (0.31, 0.31, 0.045), 0.79, (0.66, 0.82, 0.92), "cylinder"),
)


def mask_for_semantic_class(
    segmentation: np.ndarray,
    id_to_labels: Mapping[Any, Any],
    semantic_class: str,
) -> np.ndarray:
    """Return a boolean mask for one class from a Replicator ID image."""

    pixels = np.asarray(segmentation)
    if pixels.ndim == 3 and pixels.shape[2] == 1:
        pixels = pixels[:, :, 0]
    if pixels.ndim != 2:
        raise ValueError("segmentation must be a two-dimensional ID image")
    matching_ids = []
    for identifier, labels in id_to_labels.items():
        label = labels.get("class") if isinstance(labels, Mapping) else labels
        if label == semantic_class:
            matching_ids.append(int(identifier))
    if not matching_ids:
        raise ValueError(f"semantic class missing from segmentation: {semantic_class}")
    mask = np.isin(pixels, matching_ids)
    if not mask.any():
        raise ValueError(f"semantic class has no visible pixels: {semantic_class}")
    return mask

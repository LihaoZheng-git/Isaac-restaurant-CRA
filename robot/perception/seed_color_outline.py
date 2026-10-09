"""Bounded RGB outline refinement for a uniformly colored observed object.

The detector mask seeds a local connected color component. This can recover
same-colored protrusions omitted by a class-trained mask; it cannot handle
arbitrary materials, touching similar objects, or hidden geometry. Explicit
domain opt-in is required by callers. No authored colors or masks are inputs.
"""

from dataclasses import dataclass, replace
import math

import cv2
import numpy as np

from robot.perception.semantic_detection import PixelBoundingBox, SemanticDetection


@dataclass(frozen=True)
class SeedColorOutlineConfig:
    hue_tolerance: float = 12.0  # OpenCV hue is circular, period 180.
    minimum_saturation: int = 60
    minimum_value: int = 35
    box_margin_fraction: float = 0.6
    minimum_seed_pixels: int = 200
    minimum_seed_color_fraction: float = 0.8
    maximum_growth_fraction: float = 0.5
    confidence_cap: float = 0.8

    def __post_init__(self):
        for name in ("minimum_saturation", "minimum_value", "minimum_seed_pixels"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.minimum_saturation > 255 or self.minimum_value > 255:
            raise ValueError("HSV saturation/value must be within [1, 255]")
        for name in (
            "hue_tolerance",
            "box_margin_fraction",
            "minimum_seed_color_fraction",
            "maximum_growth_fraction",
            "confidence_cap",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.hue_tolerance >= 90 or any(
            getattr(self, name) > 1
            for name in (
                "box_margin_fraction",
                "minimum_seed_color_fraction",
                "maximum_growth_fraction",
                "confidence_cap",
            )
        ):
            raise ValueError("outline settings exceed bounded domain")


@dataclass(frozen=True)
class SeedColorOutlineResult:
    detection: SemanticDetection | None
    rejection_reasons: tuple[str, ...]
    added_pixel_count: int
    seed_color_fraction: float
    source: str = "rgb_seed_color_connected_outline"


def refine_seed_color_outline(
    rgb: np.ndarray,
    detection: SemanticDetection,
    *,
    uniform_color_domain_validated: bool,
    config: SeedColorOutlineConfig | None = None,
) -> SeedColorOutlineResult:
    """Refine visible RGB geometry, with bounded growth and ambiguity rejection."""
    settings = config or SeedColorOutlineConfig()
    if not isinstance(settings, SeedColorOutlineConfig):
        raise TypeError("config must be SeedColorOutlineConfig")
    if not isinstance(detection, SemanticDetection):
        raise TypeError("detection must be SemanticDetection")
    if not isinstance(uniform_color_domain_validated, bool):
        raise TypeError("uniform_color_domain_validated must be bool")
    image = np.asarray(rgb)
    if image.dtype != np.uint8 or image.shape != (*detection.mask.shape, 3):
        raise ValueError("rgb must be matching HxWx3 uint8")

    def reject(reason, fraction=0.0, added=0):
        return SeedColorOutlineResult(None, (reason,), added, fraction)

    if not uniform_color_domain_validated:
        return reject("uniform-color local separation domain has not been validated")
    mask = detection.mask
    if mask.sum() < settings.minimum_seed_pixels:
        return reject("insufficient seed mask pixels")
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    usable = (
        mask
        & (hsv[:, :, 1] >= settings.minimum_saturation)
        & (hsv[:, :, 2] >= settings.minimum_value)
    )
    if usable.sum() < settings.minimum_seed_pixels:
        return reject("insufficient chromatic seed pixels")
    # Circular mean avoids the red hue wrap; concentration is checked below.
    angles = hsv[:, :, 0][usable] * (2 * math.pi / 180)
    hue = (
        math.atan2(np.sin(angles).mean(), np.cos(angles).mean())
        * 180
        / (2 * math.pi)
        % 180
    )
    distance = abs(hsv[:, :, 0].astype(float) - hue)
    distance = np.minimum(distance, 180 - distance)
    color = (
        (distance <= settings.hue_tolerance)
        & (hsv[:, :, 1] >= settings.minimum_saturation)
        & (hsv[:, :, 2] >= settings.minimum_value)
    )
    fraction = float((mask & color).sum() / mask.sum())
    if fraction < settings.minimum_seed_color_fraction:
        return reject("seed does not support a dominant uniform color", fraction)
    ys, xs = np.where(mask)
    margin = math.ceil(max(np.ptp(xs), np.ptp(ys)) * settings.box_margin_fraction)
    y0, y1 = max(0, ys.min() - margin), min(mask.shape[0], ys.max() + margin + 1)
    x0, x1 = max(0, xs.min() - margin), min(mask.shape[1], xs.max() + margin + 1)
    roi = np.zeros(mask.shape, bool)
    roi[y0:y1, x0:x1] = True
    count, labels, _, _ = cv2.connectedComponentsWithStats(
        (color & roi).astype(np.uint8), 8
    )
    hits = np.bincount(labels[mask], minlength=count)
    hits[0] = 0
    component = labels == hits.argmax()
    if hits.max() < settings.minimum_seed_color_fraction * mask.sum():
        return reject("seed color is fragmented or ambiguously connected", fraction)
    if (
        component[y0].any()
        or component[y1 - 1].any()
        or component[:, x0].any()
        or component[:, x1 - 1].any()
    ):
        return reject(
            "color component reaches ROI/image boundary; possible touching object",
            fraction,
        )
    refined = mask | component
    added = int(refined.sum() - mask.sum())
    if added > settings.maximum_growth_fraction * mask.sum():
        return reject("outline growth exceeds validated local domain", fraction, added)
    ys, xs = np.where(refined)
    result = replace(
        detection,
        mask=refined,
        confidence=min(detection.confidence, settings.confidence_cap),
        bounding_box=PixelBoundingBox(
            float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)
        ),
    )
    return SeedColorOutlineResult(result, (), added, fraction)

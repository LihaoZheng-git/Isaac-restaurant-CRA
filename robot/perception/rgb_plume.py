"""Bounded fixed-camera RGB plume-like foreground estimation.

Requires an observed clean reference and a supplied source box. It detects
bright, low-chroma changes above that source, not aerosol identity or heat.
Grey foreground objects can confound it. Moving-camera use is unsupported.
"""

from dataclasses import dataclass
import math
import time

import cv2
import numpy as np

from robot.perception.physical_attributes import EstimateStatus, ScalarEstimate
from robot.perception.plume_segmentation import PlumeSegmentationFrame, VisualPlumeInstance
from robot.perception.semantic_detection import PixelBoundingBox


RGB_PLUME_SOURCE = "fixed_camera_rgb_plume_like_foreground"


@dataclass(frozen=True)
class RgbPlumeConfig:
    minimum_brightness_change: float = 12.0
    maximum_chroma_fraction: float = 0.35
    minimum_component_pixels: int = 20
    source_width_multiplier: float = 1.5
    source_height_multiplier: float = 4.0
    maximum_background_change: float = 8.0
    confidence_cap: float = 0.55

    def __post_init__(self) -> None:
        for name in ("minimum_brightness_change", "source_width_multiplier",
                     "source_height_multiplier", "maximum_background_change"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("maximum_chroma_fraction", "confidence_cap"):
            if not math.isfinite(getattr(self, name)) or not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
        if isinstance(self.minimum_component_pixels, bool) or not isinstance(self.minimum_component_pixels, int) or self.minimum_component_pixels < 1:
            raise ValueError("minimum_component_pixels must be a positive integer")


@dataclass(frozen=True)
class RgbPlumeResult:
    frame: PlumeSegmentationFrame | None
    spatial_association: ScalarEstimate
    rejection_reasons: tuple[str, ...]


def _rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("images must be HxWx3 uint8 RGB")
    return image


class StationaryRgbPlumeEstimator:
    def __init__(self, reference_rgb: np.ndarray, *, source_box: PixelBoundingBox,
                 config: RgbPlumeConfig | None = None) -> None:
        self.reference = np.array(_rgb(reference_rgb), copy=True)
        self.source_box = source_box
        self.config = config or RgbPlumeConfig()
        height, width = self.reference.shape[:2]
        if source_box.x_max > width or source_box.y_max > height:
            raise ValueError("source box must lie within reference image")
        box = source_box
        cx = (box.x_min + box.x_max) / 2
        half_width = box.width * self.config.source_width_multiplier / 2
        self.roi = np.zeros((height, width), dtype=bool)
        self.roi[max(0, int(box.y_min - box.height * self.config.source_height_multiplier)):int(box.y_min),
                 max(0, int(cx-half_width)):min(width, int(math.ceil(cx+half_width)))] = True
        if not np.any(self.roi) or not np.any(~self.roi):
            raise ValueError("source box must allow plume and background regions")

    def estimate(self, image_rgb: np.ndarray, *, source_frame_id: int,
                 timestamp_seconds: float) -> RgbPlumeResult:
        started = time.perf_counter()
        image = _rgb(image_rgb)
        if image.shape != self.reference.shape:
            raise ValueError("image must match reference dimensions")
        if (isinstance(source_frame_id, bool) or not isinstance(source_frame_id, int)
                or source_frame_id < 0 or not math.isfinite(timestamp_seconds) or timestamp_seconds < 0):
            raise ValueError("frame id and timestamp must be valid nonnegative values")
        difference = image.astype(float) - self.reference.astype(float)
        background_change = float(np.abs(difference[~self.roi]).mean())
        if background_change > self.config.maximum_background_change:
            reason = "background changed; fixed-camera reference is invalid"
            return RgbPlumeResult(None, ScalarEstimate.unavailable(
                unit="1", source=RGB_PLUME_SOURCE, timestamp_seconds=timestamp_seconds,
                evidence=(reason,),
            ), (reason,))
        brightness = image.mean(axis=2)
        chroma = np.ptp(image.astype(float), axis=2) / np.maximum(brightness, 1)
        candidate = (self.roi & (difference.mean(axis=2) >= self.config.minimum_brightness_change)
                     & (chroma <= self.config.maximum_chroma_fraction))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), connectivity=8)
        mask = np.zeros_like(candidate)
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] >= self.config.minimum_component_pixels:
                mask |= labels == label
        instances = ()
        association = ScalarEstimate.unavailable(
            unit="1", source=RGB_PLUME_SOURCE, timestamp_seconds=timestamp_seconds,
            evidence=("no plume-like foreground; absence needs domain validation",),
        )
        if np.any(mask):
            yy, xx = np.nonzero(mask)
            box = PixelBoundingBox(float(xx.min()), float(yy.min()), float(xx.max()+1), float(yy.max()+1))
            instances = (VisualPlumeInstance("visible_plume", "plume_like_foreground",
                         self.config.confidence_cap, box, mask, 0),)
            aligned_fraction = float(np.mean((xx >= self.source_box.x_min) & (xx < self.source_box.x_max)))
            gap = max(0., self.source_box.y_min - box.y_max)
            value = aligned_fraction * math.exp(-gap / self.source_box.height)
            association = ScalarEstimate(value, max(0., value-.15), min(1., value+.15),
                self.config.confidence_cap, "1", RGB_PLUME_SOURCE, timestamp_seconds,
                EstimateStatus.INFERRED, ("horizontal mask overlap and vertical gap to supplied source box",
                                         "source identity/open-container context must be supplied separately"))
        frame = PlumeSegmentationFrame(source_frame_id, timestamp_seconds, *image.shape[:2],
            instances, RGB_PLUME_SOURCE, time.perf_counter()-started, EstimateStatus.INFERRED,
            model_source="classical RGB reference subtraction; no learned weights")
        return RgbPlumeResult(frame, association, ())

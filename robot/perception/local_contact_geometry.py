"""Local convex boundary cues; no category, texture or physical-sharpness label.

This engineering proxy compares signed turning angles at three arc-length
supports. A localized outward corner persists across supports; smooth curvature
usually accumulates with support length. It is not an injury model or a
reimplementation of a published curvature-scale-space detector.
"""

from dataclasses import dataclass
import math

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d


@dataclass(frozen=True)
class LocalContactGeometryConfig:
    sample_count: int = 512
    minimum_smoothing_pixels: float = .8
    smoothing_perimeter_fraction: float = .003
    minimum_support_pixels: float = 3.
    support_perimeter_fractions: tuple[float, float, float] = (.015, .03, .06)
    turn_start_degrees: float = 25.
    turn_full_degrees: float = 155.
    persistence_start: float = .35
    persistence_full: float = .75

    def __post_init__(self):
        if (isinstance(self.sample_count, bool) or not isinstance(self.sample_count, int)
                or not 64 <= self.sample_count <= 2048):
            raise ValueError("sample_count must be an integer in [64, 2048]")
        for name in ("minimum_smoothing_pixels", "smoothing_perimeter_fraction",
                     "minimum_support_pixels", "turn_start_degrees", "turn_full_degrees",
                     "persistence_start", "persistence_full"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        fractions = tuple(self.support_perimeter_fractions)
        if (len(fractions) != 3 or any(isinstance(v, bool) or not math.isfinite(v)
            or not 0 < v < .2 for v in fractions) or not fractions[0] < fractions[1] < fractions[2]):
            raise ValueError("three increasing support fractions in (0, .2) required")
        if not self.turn_start_degrees < self.turn_full_degrees < 180:
            raise ValueError("turn thresholds must increase below 180 degrees")
        if not self.persistence_start < self.persistence_full <= 1:
            raise ValueError("persistence thresholds must increase within (0, 1]")
        if self.smoothing_perimeter_fraction > .02:
            raise ValueError("smoothing must not erase large outline features")
        object.__setattr__(self, "support_perimeter_fractions", fractions)


@dataclass(frozen=True)
class LocalConvexContactFeatures:
    score: float
    peak_pixel_xy: tuple[float, float]
    signed_turns_degrees: tuple[float, float, float]
    support_pixels: tuple[float, float, float]
    perimeter_pixels: float
    smoothing_pixels: float


def local_convex_contact_features(contour, config=None):
    """Return a bounded corner cue from one accepted, closed outer contour.

    Interpolation is only numerical resampling, not additional image evidence.
    Pixel-scale smoothing/support floors suppress staircase noise. Concave turns
    never score; there is no global aspect-ratio/circularity/texture risk term.
    A low score describes this visible boundary only, not hidden interior edges.
    """
    settings = LocalContactGeometryConfig() if config is None else config
    if not isinstance(settings, LocalContactGeometryConfig):
        raise TypeError("invalid local contact geometry configuration")
    points = np.asarray(contour, dtype=float)
    if points.ndim == 3 and points.shape[1:] == (1, 2):
        points = points[:, 0]
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError("contour must contain at least three finite XY points")
    oriented_area = cv2.contourArea(points.astype(np.float32), oriented=True)
    if abs(oriented_area) <= 0:
        raise ValueError("contour must enclose positive area")
    if oriented_area < 0:
        points = points[::-1]
    closed = np.vstack((points, points[:1]))
    arc = np.r_[0., np.linalg.norm(np.diff(closed, axis=0), axis=1).cumsum()]
    perimeter = float(arc[-1])
    count = settings.sample_count
    position = np.linspace(0, perimeter, count, endpoint=False)
    curve = np.column_stack([np.interp(position, arc, closed[:, axis]) for axis in (0, 1)])
    smoothing = max(settings.minimum_smoothing_pixels, settings.smoothing_perimeter_fraction * perimeter)
    curve = gaussian_filter1d(curve, smoothing * count / perimeter, axis=0, mode="wrap")
    supports = tuple(max(settings.minimum_support_pixels, f * perimeter)
                     for f in settings.support_perimeter_fractions)
    # A support cannot traverse half of a tiny contour and masquerade as local.
    if max(supports) >= perimeter / 4:
        raise ValueError("contour is too small for local support")
    angles, ambiguous_cusps = [], []
    for support in supports:
        step = max(1, round(support * count / perimeter))
        before, after = curve - np.roll(curve, step, axis=0), np.roll(curve, -step, axis=0) - curve
        cross = before[:, 0] * after[:, 1] - before[:, 1] * after[:, 0]
        dot = (before * after).sum(axis=1)
        angles.append(np.arctan2(cross, dot))
        # A one-pixel-wide tip can backtrack along the same line. atan2(-0,
        # negative) returns -pi even at an outward tip. Resolve that numerical
        # sign ONLY from the next nondegenerate larger support; an inward slit
        # therefore stays inward. No resolved direction means zero strength.
        ambiguous_cusps.append((np.abs(cross) <= 1e-12 * np.maximum(np.abs(dot), 1.)) & (dot < 0))
    angles = np.asarray(angles)
    angles[-1, ambiguous_cusps[-1]] = 0.
    for level in (1, 0):
        chosen = ambiguous_cusps[level]
        angles[level, chosen] = np.pi * np.sign(angles[level+1, chosen])
    small, middle, large = angles
    # Large support may include a nearby notch and cancel part of a real local
    # outward turn. The two smaller supports must BOTH be convex; a negative
    # large turn cannot invent strength, which depends only on those two.
    ratio = small / np.maximum(large, 1e-6)
    persistence = np.clip((ratio - settings.persistence_start) /
                          (settings.persistence_full - settings.persistence_start), 0, 1)
    strength = np.clip((np.minimum(small, middle) - math.radians(settings.turn_start_degrees)) /
                       math.radians(settings.turn_full_degrees - settings.turn_start_degrees), 0, 1)
    scores = persistence * strength
    index = int(np.argmax(scores))
    return LocalConvexContactFeatures(float(scores[index]), tuple(float(v) for v in curve[index]),
        tuple(float(v) for v in np.degrees(angles[:, index])), supports, perimeter, smoothing)

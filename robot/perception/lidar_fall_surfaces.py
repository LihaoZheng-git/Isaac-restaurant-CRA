"""Infer support-to-landing fall evidence from local LiDAR surface bands.

The estimator never substitutes object height or center height. A positive drop
requires an object-adjacent support band, a lower landing band, and a sampled
direction in which the support-height surface no longer continues. Ambiguous or
unsupported geometry remains unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from robot.perception.fall_height import FallSurfaceEvidence
from robot.perception.physical_attributes import EstimateStatus, ScalarEstimate


LIDAR_SURFACE_SOURCE = "lidar_local_support_landing_bands"


@dataclass(frozen=True)
class LidarFallSurfaceConfig:
    support_xy_margin_m: float = 0.12
    support_maximum_gap_m: float = 0.06
    minimum_support_overhang_m: float = 0.10
    minimum_support_planarity_ratio: float = 0.05
    surface_band_tolerance_m: float = 0.04
    minimum_surface_points: int = 6
    landing_minimum_offset_m: float = 0.12
    landing_maximum_offset_m: float = 0.90
    landing_half_width_m: float = 0.22
    minimum_positive_drop_m: float = 0.08
    maximum_support_continuation_fraction: float = 0.35
    direction_count: int = 16
    height_uncertainty_m: float = 0.04
    confidence: float = 0.55

    def __post_init__(self) -> None:
        positive = (
            "support_xy_margin_m",
            "support_maximum_gap_m",
            "minimum_support_overhang_m",
            "surface_band_tolerance_m",
            "landing_minimum_offset_m",
            "landing_maximum_offset_m",
            "landing_half_width_m",
            "minimum_positive_drop_m",
            "height_uncertainty_m",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        if self.landing_maximum_offset_m <= self.landing_minimum_offset_m:
            raise ValueError("landing maximum offset must exceed minimum offset")
        for name in ("minimum_surface_points", "direction_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 3:
                raise ValueError(f"{name} must be an integer of at least 3")
        for name in ("maximum_support_continuation_fraction", "minimum_support_planarity_ratio", "confidence"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class LidarFallSurfaceResult:
    evidence: FallSurfaceEvidence | None
    fall_direction_robot_xy: tuple[float, float] | None
    rejection_reasons: tuple[str, ...]
    support_point_count: int = 0
    landing_point_count: int = 0

    @property
    def is_available(self) -> bool:
        return self.evidence is not None


def estimate_lidar_fall_surfaces(
    scene_points_robot_xyz: np.ndarray,
    object_points_robot_xyz: np.ndarray,
    *,
    timestamp_seconds: float,
    config: LidarFallSurfaceConfig | None = None,
) -> LidarFallSurfaceResult:
    """Infer local support and landing heights from calibrated LiDAR points."""
    settings = config or LidarFallSurfaceConfig()
    if not isinstance(settings, LidarFallSurfaceConfig):
        raise TypeError("config must be LidarFallSurfaceConfig")
    scene = _points(scene_points_robot_xyz, "scene_points_robot_xyz")
    object_points = _points(object_points_robot_xyz, "object_points_robot_xyz")
    timestamp = float(timestamp_seconds)
    if not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError("timestamp_seconds must be finite and nonnegative")
    if object_points.shape[0] < settings.minimum_surface_points:
        return _unavailable("insufficient object points")
    scene = _exclude_object_returns(scene, object_points)

    lower_xyz = np.quantile(object_points, 0.10, axis=0)
    upper_xyz = np.quantile(object_points, 0.90, axis=0)
    center_xy = 0.5 * (lower_xyz[:2] + upper_xyz[:2])
    margin = settings.support_xy_margin_m
    support_mask = (
        (scene[:, 0] >= lower_xyz[0] - margin)
        & (scene[:, 0] <= upper_xyz[0] + margin)
        & (scene[:, 1] >= lower_xyz[1] - margin)
        & (scene[:, 1] <= upper_xyz[1] + margin)
        & (scene[:, 2] <= lower_xyz[2] + settings.support_maximum_gap_m)
    )
    support_level, support_count = _support_surface_level(
        scene[support_mask], lower_xyz, upper_xyz, settings,
    )
    if support_level is None:
        return _unavailable(
            "object-adjacent planar support was not localized within the allowed gap",
            support_point_count=support_count,
        )

    half_extent = 0.5 * float(np.linalg.norm(upper_xyz[:2] - lower_xyz[:2]))
    direction_results = []
    contiguous_directions = 0
    for index in range(settings.direction_count):
        angle = 2.0 * math.pi * index / settings.direction_count
        direction = np.asarray((math.cos(angle), math.sin(angle)))
        offset = scene[:, :2] - center_xy
        along = offset @ direction
        across = np.abs(offset[:, 0] * direction[1] - offset[:, 1] * direction[0])
        corridor = (
            (along >= half_extent + settings.landing_minimum_offset_m)
            & (along <= half_extent + settings.landing_maximum_offset_m)
            & (across <= settings.landing_half_width_m)
            & (scene[:, 2] <= support_level + settings.surface_band_tolerance_m)
        )
        heights = scene[corridor, 2]
        same_count = int(
            np.count_nonzero(
                np.abs(heights - support_level)
                <= settings.surface_band_tolerance_m
            )
        )
        if same_count >= settings.minimum_surface_points:
            contiguous_directions += 1
        lower_heights = heights[
            heights <= support_level - settings.minimum_positive_drop_m
        ]
        landing_level, landing_count = _surface_level(
            lower_heights,
            tolerance=settings.surface_band_tolerance_m,
            minimum_points=settings.minimum_surface_points,
            target=support_level - settings.minimum_positive_drop_m,
        )
        if landing_level is None:
            continue
        continuation_fraction = same_count / max(1, same_count + landing_count)
        if continuation_fraction > settings.maximum_support_continuation_fraction:
            continue
        direction_results.append(
            (landing_count, -continuation_fraction, landing_level, direction)
        )

    if not direction_results:
        # A few visible directions cannot rule out an unseen nearby edge.
        # Zero is a bounded local-drop observation only with full sampled
        # support coverage; otherwise a missing landing remains unknown.
        if contiguous_directions == settings.direction_count:
            evidence = _surface_evidence(
                support_level,
                support_level,
                timestamp=timestamp,
                settings=settings,
                support_count=support_count,
                landing_count=contiguous_directions,
                detail="support-height surface continues around the object",
            )
            return LidarFallSurfaceResult(
                evidence, None, (), support_count, contiguous_directions
            )
        return _unavailable(
            "no exposed direction with a localized lower landing surface",
            support_point_count=support_count,
        )

    direction_results.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _, _, landing_level, direction = direction_results[0]
    compatible = tuple(
        item
        for item in direction_results
        if abs(item[2] - landing_level) <= settings.surface_band_tolerance_m
    )
    strongest = compatible[0]
    if any(
        float(strongest[3] @ item[3]) < -0.90
        and item[0] >= 0.5 * strongest[0]
        for item in compatible[1:]
    ):
        return _unavailable(
            "lower landing directions are geometrically ambiguous",
            support_point_count=support_count,
        )
    landing_count = sum(item[0] for item in compatible)
    mean_direction = np.sum([item[3] * item[0] for item in compatible], axis=0)
    direction_norm = float(np.linalg.norm(mean_direction))
    if direction_norm <= 1e-9:
        return _unavailable(
            "lower landing directions are geometrically ambiguous",
            support_point_count=support_count,
        )
    mean_direction /= direction_norm
    evidence = _surface_evidence(
        support_level,
        landing_level,
        timestamp=timestamp,
        settings=settings,
        support_count=support_count,
        landing_count=landing_count,
        detail=(
            "support-height returns cease before the lower landing band in "
            f"direction=({mean_direction[0]:.3f}, {mean_direction[1]:.3f})"
        ),
    )
    return LidarFallSurfaceResult(
        evidence,
        (float(mean_direction[0]), float(mean_direction[1])),
        (),
        support_count,
        landing_count,
    )


def _points(value: np.ndarray, name: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3)")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} must contain finite values")
    return points


def _exclude_object_returns(scene: np.ndarray, object_points: np.ndarray) -> np.ndarray:
    """Remove aligned track returns so an object bottom cannot become support."""
    object_keys = {
        tuple(row) for row in np.round(object_points, decimals=6).tolist()
    }
    keep = np.asarray(
        [tuple(row) not in object_keys for row in np.round(scene, decimals=6).tolist()],
        dtype=np.bool_,
    )
    return scene[keep]


def _surface_level(
    values: np.ndarray,
    *,
    tolerance: float,
    minimum_points: int,
    target: float,
) -> tuple[float | None, int]:
    if values.size < minimum_points:
        return None, 0
    ordered = np.sort(values)
    candidates = []
    for value in ordered:
        selected = ordered[np.abs(ordered - value) <= tolerance]
        if selected.size >= minimum_points:
            level = float(np.median(selected))
            candidates.append((int(selected.size), -abs(level - target), level))
    if not candidates:
        return None, 0
    count, _, level = max(candidates)
    return level, count


def _support_surface_level(points, lower_xyz, upper_xyz, settings):
    """Select only geometrically supported bands before ranking point counts.

    Dense vertical object remnants must not outrank a sparser supporting plane.
    A rotated line can span both coordinate axes, so require two-dimensional
    coverage as well as the existing footprint-overhang test.
    """
    if len(points) < settings.minimum_surface_points:
        return None, 0
    candidates = []
    seen_levels = set()
    for height in np.unique(points[:, 2]):
        selected = points[np.abs(points[:, 2] - height) <= settings.surface_band_tolerance_m]
        if len(selected) < settings.minimum_surface_points:
            continue
        level = float(np.median(selected[:, 2]))
        key = round(level, 6)
        if key in seen_levels:
            continue
        seen_levels.add(key)
        if abs(float(lower_xyz[2]) - level) > settings.support_maximum_gap_m:
            continue
        span = np.ptp(selected[:, :2], axis=0)
        if np.any(span < upper_xyz[:2] - lower_xyz[:2] + settings.minimum_support_overhang_m):
            continue
        eigenvalues = np.linalg.eigvalsh(np.cov(selected[:, :2], rowvar=False, bias=True))
        if eigenvalues[-1] <= 1e-12 or eigenvalues[0] / eigenvalues[-1] < settings.minimum_support_planarity_ratio:
            continue
        candidates.append((len(selected), -abs(level - float(lower_xyz[2])), level))
    if not candidates:
        return None, 0
    count, _, level = max(candidates)
    return level, count


def _surface_evidence(
    support_height: float,
    landing_height: float,
    *,
    timestamp: float,
    settings: LidarFallSurfaceConfig,
    support_count: int,
    landing_count: int,
    detail: str,
) -> FallSurfaceEvidence:
    def estimate(value: float, role: str, count: int) -> ScalarEstimate:
        uncertainty = settings.height_uncertainty_m
        return ScalarEstimate(
            value,
            value - uncertainty,
            value + uncertainty,
            settings.confidence,
            "m",
            LIDAR_SURFACE_SOURCE,
            timestamp,
            EstimateStatus.INFERRED,
            (
                f"{role} horizontal band from {count} local LiDAR returns",
                detail,
                "surface-band inference is not object height or center height",
            ),
        )

    return FallSurfaceEvidence(
        estimate(support_height, "support", support_count),
        estimate(landing_height, "landing", landing_count),
    )


def _unavailable(
    reason: str, *, support_point_count: int = 0
) -> LidarFallSurfaceResult:
    return LidarFallSurfaceResult(
        None, None, (reason,), support_point_count=support_point_count
    )

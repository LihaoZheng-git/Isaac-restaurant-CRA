"""Convert tracked LiDAR returns into CRA geometry attributes.

The obstacle tracker owns point-to-track membership.  This module consumes
that membership and measures a robust, chassis-frame envelope for one track.
The envelope describes the returns visible in the current scan; it is not
presented as a complete CAD model of the object.

Camera--LiDAR association evidence may be attached for provenance, but YOLO
confidence never changes geometric confidence.  Geometry confidence depends
only on LiDAR point support and the configured measurement model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real

import numpy as np

from robot.perception.camera_lidar_fusion import TrackSemanticAssociation
from robot.perception.physical_attributes import (
    EstimateStatus,
    GeometryAttributes,
    ScalarEstimate,
)
from robot.safety.obstacle_tracking import ObstacleTrackingResult


GEOMETRY_SOURCE = "lidar_track_visible_envelope"


def _finite_real(value: object, *, name: str) -> float:
    """Return one finite real value while rejecting booleans."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _positive_integer(value: object, *, name: str) -> int:
    """Return one positive integer while rejecting booleans."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    return result


@dataclass(frozen=True)
class LidarGeometryConfig:
    """Uncertainty and support settings for a visible LiDAR envelope."""

    minimum_track_points: int = 4
    full_confidence_point_count: int = 20
    lower_quantile: float = 0.05
    upper_quantile: float = 0.95
    range_uncertainty_m: float = 0.02
    unobserved_extent_margin_m: float = 0.10
    minimum_confidence: float = 0.35
    maximum_confidence: float = 0.90

    def __post_init__(self) -> None:
        minimum_points = _positive_integer(
            self.minimum_track_points,
            name="minimum_track_points",
        )
        full_points = _positive_integer(
            self.full_confidence_point_count,
            name="full_confidence_point_count",
        )
        if full_points < minimum_points:
            raise ValueError(
                "full_confidence_point_count cannot be smaller than "
                "minimum_track_points."
            )

        lower = _finite_real(self.lower_quantile, name="lower_quantile")
        upper = _finite_real(self.upper_quantile, name="upper_quantile")
        if not 0.0 <= lower < upper <= 1.0:
            raise ValueError(
                "lower_quantile and upper_quantile must satisfy "
                "0 <= lower < upper <= 1."
            )

        for name in ("range_uncertainty_m", "unobserved_extent_margin_m"):
            value = _finite_real(getattr(self, name), name=name)
            if value < 0.0:
                raise ValueError(f"{name} cannot be negative.")

        minimum_confidence = _finite_real(
            self.minimum_confidence,
            name="minimum_confidence",
        )
        maximum_confidence = _finite_real(
            self.maximum_confidence,
            name="maximum_confidence",
        )
        if not 0.0 <= minimum_confidence <= maximum_confidence <= 1.0:
            raise ValueError(
                "Geometry confidences must satisfy "
                "0 <= minimum <= maximum <= 1."
            )


def estimate_track_geometry(
    obstacle_points_robot_xyz: np.ndarray,
    *,
    tracking_result: ObstacleTrackingResult,
    track_id: int,
    floor_height_robot_m: float,
    association: TrackSemanticAssociation | None = None,
    config: LidarGeometryConfig | None = None,
) -> GeometryAttributes:
    """Estimate one track's visible metric envelope in the chassis frame.

    ``obstacle_points_robot_xyz`` must be the 3D array from which the tracker
    received its aligned XY coordinates.  Width is the robust Y span, depth
    is the robust X span, and height is the robust Z span.  Center height is
    reported relative to the supplied floor plane.  Fall height remains
    unavailable because it requires support and stability reasoning.
    """

    if not isinstance(tracking_result, ObstacleTrackingResult):
        raise TypeError("tracking_result must be an ObstacleTrackingResult.")
    selected_track_id = _positive_track_id(track_id)
    floor_height = _finite_real(
        floor_height_robot_m,
        name="floor_height_robot_m",
    )
    settings = config or LidarGeometryConfig()
    if not isinstance(settings, LidarGeometryConfig):
        raise TypeError("config must be a LidarGeometryConfig.")
    if association is not None:
        if not isinstance(association, TrackSemanticAssociation):
            raise TypeError(
                "association must be a TrackSemanticAssociation or None."
            )
        if association.track_id != selected_track_id:
            raise ValueError("association.track_id must match track_id.")

    points = np.asarray(obstacle_points_robot_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            "obstacle_points_robot_xyz must have shape (N, 3)."
        )
    if not np.isfinite(points).all():
        raise ValueError(
            "obstacle_points_robot_xyz must contain only finite values."
        )
    point_track_ids = np.asarray(
        tracking_result.point_track_ids,
        dtype=np.int64,
    )
    if point_track_ids.shape != (points.shape[0],):
        raise ValueError(
            "tracking_result.point_track_ids must align with obstacle points."
        )
    timestamp = _finite_real(
        tracking_result.timestamp_seconds,
        name="tracking_result.timestamp_seconds",
    )
    if timestamp < 0.0:
        raise ValueError("tracking_result.timestamp_seconds cannot be negative.")

    track_points = points[point_track_ids == selected_track_id]
    point_count = int(track_points.shape[0])
    association_evidence = _association_evidence(association)
    if point_count < settings.minimum_track_points:
        evidence = (
            f"track {selected_track_id} has {point_count} aligned LiDAR returns",
            f"at least {settings.minimum_track_points} returns are required",
            *association_evidence,
        )
        return _unavailable_geometry(timestamp=timestamp, evidence=evidence)

    lower = np.quantile(track_points, settings.lower_quantile, axis=0)
    upper = np.quantile(track_points, settings.upper_quantile, axis=0)
    depth = max(0.0, float(upper[0] - lower[0]))
    width = max(0.0, float(upper[1] - lower[1]))
    height = max(0.0, float(upper[2] - lower[2]))
    center_height = max(
        0.0,
        float(0.5 * (upper[2] + lower[2]) - floor_height),
    )

    confidence = _point_support_confidence(point_count, settings)
    evidence = (
        f"robust envelope from {point_count} tracked LiDAR returns",
        (
            "chassis-frame quantiles "
            f"[{settings.lower_quantile:.3f}, {settings.upper_quantile:.3f}]"
        ),
        "bounds include sensor uncertainty and unobserved-surface allowance",
        "visible envelope is not assumed to be a complete object model",
        *association_evidence,
    )

    return GeometryAttributes(
        width_m=_geometry_estimate(
            width,
            confidence=confidence,
            timestamp=timestamp,
            settings=settings,
            evidence=evidence,
        ),
        depth_m=_geometry_estimate(
            depth,
            confidence=0.80 * confidence,
            timestamp=timestamp,
            settings=settings,
            evidence=evidence,
        ),
        height_m=_geometry_estimate(
            height,
            confidence=0.85 * confidence,
            timestamp=timestamp,
            settings=settings,
            evidence=evidence,
        ),
        center_height_m=_geometry_estimate(
            center_height,
            confidence=0.80 * confidence,
            timestamp=timestamp,
            settings=settings,
            evidence=(
                *evidence,
                f"floor reference z={floor_height:.4f} m in chassis frame",
            ),
        ),
        fall_height_m=ScalarEstimate.unavailable(
            unit="m",
            source="support_and_stability_estimator_required",
            timestamp_seconds=timestamp,
            evidence=(
                "visible geometry alone cannot determine available fall height",
            ),
        ),
    )


def _positive_track_id(value: object) -> int:
    """Return a nonnegative integer track identifier."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("track_id must be an integer.")
    result = int(value)
    if result < 0:
        raise ValueError("track_id cannot be negative.")
    return result


def _point_support_confidence(
    point_count: int,
    settings: LidarGeometryConfig,
) -> float:
    """Map LiDAR support monotonically into the configured confidence range."""

    support_span = (
        settings.full_confidence_point_count - settings.minimum_track_points
    )
    if support_span == 0:
        fraction = 1.0
    else:
        fraction = min(
            1.0,
            (point_count - settings.minimum_track_points) / support_span,
        )
    return settings.minimum_confidence + fraction * (
        settings.maximum_confidence - settings.minimum_confidence
    )


def _geometry_estimate(
    value: float,
    *,
    confidence: float,
    timestamp: float,
    settings: LidarGeometryConfig,
    evidence: tuple[str, ...],
) -> ScalarEstimate:
    """Construct one nonnegative metric estimate with conservative bounds."""

    lower = max(0.0, value - settings.range_uncertainty_m)
    upper = (
        value
        + settings.range_uncertainty_m
        + settings.unobserved_extent_margin_m
    )
    return ScalarEstimate(
        value=value,
        lower_bound=lower,
        upper_bound=upper,
        confidence=min(1.0, max(0.0, confidence)),
        unit="m",
        source=GEOMETRY_SOURCE,
        timestamp_seconds=timestamp,
        status=EstimateStatus.INFERRED,
        evidence=evidence,
    )


def _association_evidence(
    association: TrackSemanticAssociation | None,
) -> tuple[str, ...]:
    """Describe optional semantic association without changing confidence."""

    if association is None:
        return ("no camera semantic association supplied",)
    return (
        (
            "camera-LiDAR association "
            f"status={association.status}, visible_points="
            f"{association.visible_point_count}, overlap="
            f"{association.overlap_fraction:.3f}"
        ),
        "semantic confidence is kept separate from geometry confidence",
    )


def _unavailable_geometry(
    *,
    timestamp: float,
    evidence: tuple[str, ...],
) -> GeometryAttributes:
    """Return an explicitly unavailable geometry group."""

    def unavailable(source: str = GEOMETRY_SOURCE) -> ScalarEstimate:
        return ScalarEstimate.unavailable(
            unit="m",
            source=source,
            timestamp_seconds=timestamp,
            evidence=evidence,
        )

    return GeometryAttributes(
        width_m=unavailable(),
        depth_m=unavailable(),
        height_m=unavailable(),
        center_height_m=unavailable(),
        fall_height_m=unavailable(
            "support_and_stability_estimator_required"
        ),
    )

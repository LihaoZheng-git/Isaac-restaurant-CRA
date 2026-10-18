"""Class-free visible elevation and support hypotheses from aligned LiDAR.

The local floor is fitted to real returns near an explicitly declared ground
reference. That reference selects a floor band; it never supplies a fitted
height or a hidden object bottom. Support classification and strict local drop
are separate: neither visible center height nor the lowest visible return is
automatically a fall height. No person/carry relationship is inferred here.
"""

from dataclasses import dataclass, field
import math

import numpy as np

from robot.perception.fall_height import estimate_fall_height
from robot.perception.lidar_fall_surfaces import (
    LidarFallSurfaceConfig,
    estimate_lidar_fall_surfaces,
    estimate_lidar_support_surface,
)
from robot.perception.physical_attributes import (
    EstimateStatus,
    ScalarEstimate,
    combine_estimate_statuses,
)


HEIGHT_SOURCE = "lidar_visible_height_and_adjacent_support"


@dataclass(frozen=True)
class ObjectHeightConfig:
    minimum_object_points: int = 6
    minimum_floor_points: int = 20
    search_radius_m: float = 2.5
    maximum_floor_extrapolation_m: float = 1.5
    floor_reference_band_m: float = 0.12
    maximum_plane_residual_m: float = 0.02
    maximum_plane_slope: float = 0.12
    minimum_floor_span_m: float = 0.4
    minimum_xy_eigenvalue_ratio: float = 0.05
    measurement_allowance_m: float = 0.03
    extrapolation_allowance_per_m: float = 0.02
    ground_support_tolerance_m: float = 0.06
    minimum_elevated_support_m: float = 0.15
    maximum_evidence_age_seconds: float = 0.35
    confidence_cap: float = 0.55
    surface_config: LidarFallSurfaceConfig = field(
        default_factory=LidarFallSurfaceConfig
    )

    def __post_init__(self):
        for name in ("minimum_object_points", "minimum_floor_points"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 6:
                raise ValueError(f"{name} must be an integer >= 6")
        for name in self.__dataclass_fields__:
            if name in {
                "minimum_object_points",
                "minimum_floor_points",
                "surface_config",
            }:
                continue
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.confidence_cap > 1 or self.minimum_xy_eigenvalue_ratio > 1:
            raise ValueError("confidence/planarity ratio must be <= 1")
        if self.maximum_floor_extrapolation_m > self.search_radius_m:
            raise ValueError("floor extrapolation must fit within search radius")
        if self.minimum_elevated_support_m <= self.ground_support_tolerance_m:
            raise ValueError("elevated support must exceed ground tolerance")
        if not isinstance(self.surface_config, LidarFallSurfaceConfig):
            raise TypeError("surface_config must be LidarFallSurfaceConfig")


@dataclass(frozen=True)
class ObjectHeightEvidence:
    ground_height_m: ScalarEstimate
    visible_lower_height_m: ScalarEstimate
    visible_center_height_m: ScalarEstimate
    support_height_m: ScalarEstimate
    support_height_above_ground_m: ScalarEstimate
    support_state: str
    fall_height_m: ScalarEstimate
    floor_point_count: int
    support_point_count: int
    support_candidate_heights_m: tuple[float, ...]
    nearest_floor_distance_m: float | None
    floor_plane_rms_m: float | None
    reference_scope: str
    rejection_reasons: tuple[str, ...]
    fall_rejection_reasons: tuple[str, ...]


def estimate_object_height(
    scene_points_xyz,
    object_points_xyz,
    *,
    timestamp_seconds: float,
    scene_timestamp_seconds: float,
    floor_reference_m: float,
    reference_scope: str,
    config: ObjectHeightConfig | None = None,
) -> ObjectHeightEvidence:
    """Estimate one instance's visible elevation and adjacent support.

    Inputs must share a metric gravity-aligned Z-up frame and source epoch.
    ``reference_scope`` must name any oracle/prior frame and floor-band selection
    dependency. Scalar heights are inferred from returns, in that declared
    reference frame; they are not an oracle-free ground identification system.
    A cluster without validated instance ownership is diagnostic only.
    """
    settings = config or ObjectHeightConfig()
    if not isinstance(settings, ObjectHeightConfig):
        raise TypeError("config must be ObjectHeightConfig")
    scene, item = (_points(value) for value in (scene_points_xyz, object_points_xyz))
    timestamp, source_time = float(timestamp_seconds), float(scene_timestamp_seconds)
    if not all(math.isfinite(t) and t >= 0 for t in (timestamp, source_time)):
        raise ValueError("timestamps must be finite and nonnegative")
    if isinstance(floor_reference_m, bool) or not math.isfinite(floor_reference_m):
        raise ValueError("floor_reference_m must be finite")
    if not isinstance(reference_scope, str) or not reference_scope.strip():
        raise ValueError("explicit reference_scope is required")

    def missing(reason):
        return ScalarEstimate.unavailable(
            unit="m",
            source=HEIGHT_SOURCE,
            timestamp_seconds=source_time,
            evidence=(reason, reference_scope),
        )

    absent = missing("usable local geometry not yet established")
    ground = lower = center = support = support_above = fall = absent
    floor_count = support_count = 0
    candidate_heights = ()
    nearest = rms = None
    state, reasons, fall_reasons = "uncertain", [], []

    def result():
        return ObjectHeightEvidence(
            ground,
            lower,
            center,
            support,
            support_above,
            state,
            fall,
            floor_count,
            support_count,
            candidate_heights,
            nearest,
            rms,
            reference_scope,
            tuple(reasons),
            tuple(fall_reasons),
        )

    if not 0 <= timestamp - source_time <= settings.maximum_evidence_age_seconds:
        reasons.append("scene/object evidence is stale or future")
        return result()
    if len(item) < settings.minimum_object_points:
        reasons.append("insufficient object returns")
        return result()

    # Keep the original strict support/drop gates. A floor fit alone never
    # supplies unobserved support/landing continuity beneath the object.
    supported = estimate_lidar_support_surface(
        scene, item, timestamp_seconds=source_time, config=settings.surface_config
    )
    support, support_count = supported.height_m, supported.point_count
    candidate_heights = supported.candidate_heights_m
    strict = estimate_lidar_fall_surfaces(
        scene, item, timestamp_seconds=source_time, config=settings.surface_config
    )
    fall = estimate_fall_height(strict.evidence, timestamp_seconds=source_time)
    fall_reasons.extend(strict.rejection_reasons)
    reasons.extend(supported.rejection_reasons)
    if not support.is_available:
        fall = missing("general support gate rejected; no drop inferred")
        fall_reasons.extend(supported.rejection_reasons)

    xy = np.median(item[:, :2], axis=0)
    local = scene[np.linalg.norm(scene[:, :2] - xy, axis=1) <= settings.search_radius_m]
    # The object cannot be evidence for its own floor. Exact same-source return
    # exclusion also works when an instance has no semantic category.
    keys = {tuple(row) for row in np.round(item, 6)}
    local = local[
        np.array([tuple(row) not in keys for row in np.round(local, 6)], dtype=bool)
    ]
    band = local[
        np.abs(local[:, 2] - floor_reference_m) <= settings.floor_reference_band_m
    ]
    floor_count = len(band)
    if floor_count < settings.minimum_floor_points:
        reasons.append("insufficient independent floor-band returns")
        return result()
    relative_xy = band[:, :2] - xy
    eigenvalues = np.linalg.eigvalsh(np.cov(relative_xy, rowvar=False, bias=True))
    if (
        np.any(np.ptp(relative_xy, axis=0) < settings.minimum_floor_span_m)
        or eigenvalues[-1] <= 1e-12
        or eigenvalues[0] / eigenvalues[-1] < settings.minimum_xy_eigenvalue_ratio
    ):
        reasons.append("floor band lacks two-dimensional spatial coverage")
        return result()
    design = np.column_stack((relative_xy, np.ones(floor_count)))
    plane, _, rank, _ = np.linalg.lstsq(design, band[:, 2], rcond=None)
    rms = float(np.sqrt(np.mean((design @ plane - band[:, 2]) ** 2)))
    if rank < 3 or rms > settings.maximum_plane_residual_m:
        reasons.append("floor band is not sufficiently planar")
        return result()
    if np.linalg.norm(plane[:2]) > settings.maximum_plane_slope:
        reasons.append("floor slope exceeds configured scope")
        return result()
    nearest = float(np.linalg.norm(relative_xy, axis=1).min())
    if nearest > settings.maximum_floor_extrapolation_m:
        reasons.append("observed floor is too far from object")
        return result()
    allowance = (
        settings.measurement_allowance_m
        + 2 * rms
        + nearest * settings.extrapolation_allowance_per_m
    )
    ground = ScalarEstimate(
        float(plane[2]),
        float(plane[2] - allowance),
        float(plane[2] + allowance),
        settings.confidence_cap,
        "m",
        "lidar_local_floor_plane_in_declared_reference",
        source_time,
        EstimateStatus.INFERRED,
        (
            reference_scope,
            f"floor_returns={floor_count}; plane_rms_m={rms}; nearest_floor_m={nearest}",
            "plane evaluated at object XY; unseen support/landing continuity is not established",
        ),
    )

    low, high = np.quantile(item[:, 2], (0.05, 0.95))

    def elevation(value, source):
        delta = float(value - ground.value)
        error = allowance + settings.measurement_allowance_m
        return ScalarEstimate(
            max(0.0, delta),
            max(0.0, delta - error),
            max(0.0, delta + error),
            settings.confidence_cap,
            "m",
            source,
            source_time,
            EstimateStatus.INFERRED,
            ground.evidence
            + (
                "visible envelope only; hidden base and physical center of mass are not measured",
            ),
        )

    lower = elevation(low, "lidar_visible_lower_boundary_above_floor")
    center = elevation((low + high) / 2, "lidar_visible_envelope_center_above_floor")
    if float(high) < ground.lower_bound - settings.measurement_allowance_m:
        reasons.append(
            "object lies below fitted floor; inconsistent association/reference"
        )
        lower = center = missing(reasons[-1])
        support = support_above = fall = missing(reasons[-1])
        return result()
    if support.is_available:
        if support.upper_bound < ground.lower_bound:
            reasons.append(
                "support lies below fitted local floor; association/reference inconsistent"
            )
            support_above = missing(reasons[-1])
            fall = missing(reasons[-1])
            fall_reasons.append(reasons[-1])
            return result()
        support_above = _difference(support, ground)
        if (
            abs(support_above.value) <= settings.ground_support_tolerance_m
            and support_above.lower_bound <= settings.ground_support_tolerance_m
            and support_above.upper_bound <= settings.minimum_elevated_support_m
        ):
            state = "ground_supported"
        elif support_above.lower_bound >= settings.minimum_elevated_support_m:
            state = "elevated_supported"
        else:
            reasons.append("support height cannot distinguish ground from elevation")
    # In particular, a floating torso/upper-object fragment remains uncertain.
    return result()


def _difference(support, ground):
    return ScalarEstimate(
        max(0.0, support.value - ground.value),
        max(0.0, support.lower_bound - ground.upper_bound),
        max(0.0, support.upper_bound - ground.lower_bound),
        min(support.confidence, ground.confidence),
        "m",
        "lidar_support_elevation_above_floor",
        min(support.timestamp_seconds, ground.timestamp_seconds),
        combine_estimate_statuses((support.status, ground.status)),
        support.evidence + ground.evidence,
    )


def _points(value):
    result = np.asarray(value, dtype=float)
    if result.ndim != 2 or result.shape[1] != 3 or not np.isfinite(result).all():
        raise ValueError("point clouds must be finite Nx3")
    return result

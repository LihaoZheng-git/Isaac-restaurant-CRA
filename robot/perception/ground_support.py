"""Prior-assisted local ground support, separate from observed edge/drop evidence.

A visible spherical surface can test consistency with an observed floor plane.
It cannot establish continuity through a LiDAR blind region. That assertion is
an explicit, bounded PRIOR; without it the result remains unavailable. Object
height/center height never becomes fall height, and hidden edges are not ruled
out by this method.
"""

from dataclasses import dataclass, replace
import math

import numpy as np

from robot.perception.fall_height import FallSurfaceEvidence, estimate_fall_height
from robot.perception.physical_attributes import EstimateStatus, ScalarEstimate
from robot.perception.spherical_mass import SphericalGeometryEvidence


GROUND_SOURCE = "lidar_floor_with_continuity_prior"


@dataclass(frozen=True)
class GroundSupportConfig:
    search_radius_m: float = 2.0
    maximum_extrapolation_m: float = 1.5
    minimum_floor_points: int = 20
    floor_band_half_width_m: float = 0.04
    maximum_plane_residual_m: float = 0.015
    maximum_plane_slope: float = 0.10
    minimum_plane_span_m: float = 0.40
    object_exclusion_margin_m: float = 0.10
    minimum_xy_eigenvalue_ratio: float = 0.05
    maximum_nominal_bottom_clearance_m: float = 0.06
    maximum_bottom_uncertainty_m: float = 0.15
    height_allowance_m: float = 0.02
    extrapolation_allowance_per_m: float = 0.015
    maximum_evidence_age_seconds: float = 0.5
    floor_confidence_cap: float = 0.55

    def __post_init__(self):
        count = self.minimum_floor_points
        if isinstance(count, bool) or not isinstance(count, int) or count < 6:
            raise ValueError("minimum_floor_points must be an integer >= 6")
        for name in self.__dataclass_fields__:
            if name == "minimum_floor_points":
                continue
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.maximum_extrapolation_m > self.search_radius_m:
            raise ValueError("extrapolation must fit inside the floor search radius")
        if self.floor_confidence_cap > 1 or self.minimum_xy_eigenvalue_ratio > 1:
            raise ValueError("confidence and eigenvalue ratio cannot exceed 1")


@dataclass(frozen=True)
class GroundSupportEvidence:
    fall_height_m: ScalarEstimate
    potential_interaction_likelihood: ScalarEstimate
    floor_height_m: ScalarEstimate
    fitted_bottom_height_m: ScalarEstimate
    surfaces: FallSurfaceEvidence | None
    floor_point_count: int
    nearest_floor_distance_m: float | None
    rejection_reasons: tuple[str, ...]
    prior_scope: str | None


def estimate_spherical_ground_support(
    scene_points_xyz: np.ndarray,
    spherical: SphericalGeometryEvidence,
    *,
    timestamp_seconds: float,
    scene_timestamp_seconds: float,
    ground_continuity_prior: ScalarEstimate | None = None,
    prior_scope: str | None = None,
    config: GroundSupportConfig | None = None,
) -> GroundSupportEvidence:
    """Test a local flat-ground hypothesis using calibrated Z-up LiDAR points.

    The fitted bottom is only a consistency gate. Support and landing come
    from the same observed plane plus the explicit continuity assumption;
    their difference supplies a bounded local drop. This does not model
    tipping, stairs, unseen voids or the object's center-of-mass energy.
    """
    settings = config or GroundSupportConfig()
    if not isinstance(settings, GroundSupportConfig):
        raise TypeError("config must be GroundSupportConfig")
    if not isinstance(spherical, SphericalGeometryEvidence):
        raise TypeError("spherical must be SphericalGeometryEvidence")
    points = np.asarray(scene_points_xyz, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("scene_points_xyz must be finite Nx3")
    timestamp = float(timestamp_seconds)
    scene_time = float(scene_timestamp_seconds)
    if not all(math.isfinite(t) and t >= 0 for t in (timestamp, scene_time)):
        raise ValueError("timestamp_seconds must be finite and nonnegative")
    if ground_continuity_prior is not None:
        if not isinstance(ground_continuity_prior, ScalarEstimate):
            raise TypeError("ground_continuity_prior must be ScalarEstimate or None")
        if ground_continuity_prior.unit != "1":
            raise ValueError("continuity prior must be dimensionless")
        if ground_continuity_prior.is_available and (
            ground_continuity_prior.status is not EstimateStatus.PRIOR
            or ground_continuity_prior.lower_bound != 1
            or ground_continuity_prior.upper_bound != 1
        ):
            raise ValueError("continuity assertion must be an explicit PRIOR of 1")
    if prior_scope is not None and (
        not isinstance(prior_scope, str) or not prior_scope.strip()
    ):
        raise ValueError("prior_scope must be nonempty text or None")

    def missing(unit, reason):
        return ScalarEstimate.unavailable(
            unit=unit,
            source=GROUND_SOURCE,
            timestamp_seconds=timestamp,
            evidence=(reason,),
        )

    bottom = missing("m", "accepted spherical geometry required")
    floor = missing("m", "local floor plane not observed")
    count, nearest = 0, None

    def reject(reason):
        return GroundSupportEvidence(
            missing("m", reason),
            missing("1", reason),
            floor,
            bottom,
            None,
            count,
            nearest,
            (reason,),
            prior_scope,
        )

    radius = spherical.radius_m
    if (
        not radius.is_available
        or radius.confidence <= 0
        or spherical.center_xyz is None
    ):
        return reject("accepted spherical geometry required")
    if radius.status not in {EstimateStatus.MEASURED, EstimateStatus.INFERRED}:
        return reject(
            "sphere geometry must be measured/inferred, not authored or prior"
        )
    if (
        not 0
        <= timestamp - radius.timestamp_seconds
        <= settings.maximum_evidence_age_seconds
    ):
        return reject("sphere geometry is future or stale")
    if not 0 <= timestamp - scene_time <= settings.maximum_evidence_age_seconds:
        return reject("scene point cloud is future or stale")
    center = np.asarray(spherical.center_xyz, dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("sphere center must be finite XYZ")
    allowance = max(
        radius.value - radius.lower_bound, radius.upper_bound - radius.value
    )
    # Allow center and radius errors together; this is an engineering bound.
    bottom_value = float(center[2] - radius.value)
    bottom = ScalarEstimate(
        bottom_value,
        bottom_value - 2 * allowance - settings.height_allowance_m,
        bottom_value + 2 * allowance + settings.height_allowance_m,
        radius.confidence,
        "m",
        "fitted_sphere_bottom_consistency_gate",
        radius.timestamp_seconds,
        radius.status,
        radius.evidence
        + ("fitted bottom is a contact-consistency gate, never fall height",),
    )
    distance_xy = np.linalg.norm(points[:, :2] - center[:2], axis=1)
    # A spherical surface/remnant must never establish its own floor plane.
    local = points[
        (distance_xy <= settings.search_radius_m)
        & (distance_xy > radius.upper_bound + settings.object_exclusion_margin_m)
    ]
    if len(local) < settings.minimum_floor_points:
        return reject("insufficient local scene returns")
    low_level = np.quantile(local[:, 2], 0.02)
    band = local[np.abs(local[:, 2] - low_level) <= settings.floor_band_half_width_m]
    count = len(band)
    if count < settings.minimum_floor_points:
        return reject("insufficient low horizontal-band returns")
    xy = band[:, :2] - center[:2]
    eigenvalues = np.linalg.eigvalsh(np.cov(xy, rowvar=False, bias=True))
    if (
        np.any(np.ptp(xy, axis=0) < settings.minimum_plane_span_m)
        or eigenvalues[-1] <= 1e-12
        or eigenvalues[0] / eigenvalues[-1] < settings.minimum_xy_eigenvalue_ratio
    ):
        return reject("floor band lacks two-dimensional spatial support")
    design = np.column_stack((xy, np.ones(count)))
    plane, _, rank, _ = np.linalg.lstsq(design, band[:, 2], rcond=None)
    residual = float(np.sqrt(np.mean((design @ plane - band[:, 2]) ** 2)))
    if rank < 3 or residual > settings.maximum_plane_residual_m:
        return reject("floor band is not sufficiently planar")
    if np.linalg.norm(plane[:2]) > settings.maximum_plane_slope:
        return reject("floor plane slope exceeds bounded flat-ground scope")
    nearest = float(np.linalg.norm(xy, axis=1).min())
    plane_allowance = (
        settings.height_allowance_m
        + 2 * residual
        + nearest * settings.extrapolation_allowance_per_m
    )
    floor = ScalarEstimate(
        float(plane[2]),
        float(plane[2] - plane_allowance),
        float(plane[2] + plane_allowance),
        settings.floor_confidence_cap,
        "m",
        "lidar_observed_local_floor_plane",
        scene_time,
        EstimateStatus.INFERRED,
        (
            f"floor_returns={count}; plane_rms_m={residual}; nearest_observed_floor_m={nearest}",
            "plane height at object XY is extrapolated; actual underlying support is not observed",
        ),
    )
    if nearest > settings.maximum_extrapolation_m:
        return reject("floor extrapolation exceeds configured prior scope")
    nominal_clearance = bottom.value - floor.value
    if abs(nominal_clearance) > settings.maximum_nominal_bottom_clearance_m:
        return reject("fitted bottom is inconsistent with local ground contact")
    if (
        bottom.lower_bound > floor.upper_bound
        or bottom.upper_bound < floor.lower_bound
        or bottom.upper_bound - floor.lower_bound
        > settings.maximum_bottom_uncertainty_m
    ):
        return reject("ground-contact intervals are inconsistent or too uncertain")
    prior = ground_continuity_prior
    if (
        prior is None
        or not prior.is_available
        or prior.confidence <= 0
        or prior_scope is None
    ):
        return reject(
            "explicit scoped ground-continuity prior required for unseen support"
        )
    if (
        not 0
        <= timestamp - prior.timestamp_seconds
        <= settings.maximum_evidence_age_seconds
    ):
        return reject("ground-continuity prior is future or stale")
    evidence_time = min(radius.timestamp_seconds, scene_time, prior.timestamp_seconds)
    detail = (
        f"continuity source={prior.source}; status=prior; scope={prior_scope}",
        "unseen support and local landing are assumed to continue the observed floor plane",
        "local support-to-landing drop only; not object height, center height or tipping energy",
    )
    support = replace(
        floor,
        source=GROUND_SOURCE,
        status=EstimateStatus.PRIOR,
        timestamp_seconds=evidence_time,
        confidence=min(floor.confidence, bottom.confidence, prior.confidence),
        evidence=floor.evidence + bottom.evidence + detail,
    )
    landing = replace(
        floor,
        source=GROUND_SOURCE,
        status=EstimateStatus.PRIOR,
        timestamp_seconds=evidence_time,
        confidence=support.confidence,
        evidence=floor.evidence + detail,
    )
    surfaces = FallSurfaceEvidence(support, landing)
    interaction = ScalarEstimate(
        1.0,
        0.0,
        1.0,
        min(0.5, prior.confidence),
        "1",
        "ground_support_worst_case_interaction_prior",
        evidence_time,
        EstimateStatus.PRIOR,
        (
            "harmful-fall opportunity is unmeasured; explicit worst-case upper bound 1",
            "not route-crossing probability, instability measurement or evidence of absence",
        ),
    )
    return GroundSupportEvidence(
        estimate_fall_height(surfaces, timestamp_seconds=timestamp),
        interaction,
        floor,
        bottom,
        surfaces,
        count,
        nearest,
        (),
        prior_scope,
    )

"""Bounded LiDAR sphere fitting and explicit effective-density mass priors.

Extrapolation from a visible surface assumes a single spherical object. It is
not complete measured geometry or a material detector. Never use the authored
simulation radius, center, density or mass as estimator inputs.
"""

from dataclasses import dataclass
import math

import numpy as np

from robot.perception.mass_estimation import estimate_mass_from_volume_density
from robot.perception.physical_attributes import (
    EstimateStatus,
    MassAttributes,
    ScalarEstimate,
)


SPHERE_VOLUME_SOURCE = "lidar_fitted_sphere_volume"


@dataclass(frozen=True)
class SphereFitConfig:
    minimum_points: int = 40
    maximum_condition_number: float = 10.0
    maximum_relative_residual: float = 0.05
    minimum_normal_span: float = 0.5
    minimum_radius_m: float = 0.03
    maximum_radius_m: float = 1.0
    sensor_allowance_m: float = 0.005
    relative_radius_allowance: float = 0.05
    confidence_cap: float = 0.6

    def __post_init__(self):
        if (
            isinstance(self.minimum_points, bool)
            or not isinstance(self.minimum_points, int)
            or self.minimum_points < 4
        ):
            raise ValueError("minimum_points must be an integer >= 4")
        for name in self.__dataclass_fields__:
            if name == "minimum_points":
                continue
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.minimum_radius_m >= self.maximum_radius_m:
            raise ValueError("minimum radius must be below maximum radius")
        if (
            self.confidence_cap > 1
            or self.maximum_relative_residual >= 1
            or self.relative_radius_allowance >= 1
            or self.minimum_normal_span > 2
        ):
            raise ValueError("invalid bounded sphere-fit configuration")


@dataclass(frozen=True)
class SphericalGeometryEvidence:
    radius_m: ScalarEstimate
    volume_m3: ScalarEstimate
    center_xyz: tuple[float, float, float] | None
    relative_residual: float | None
    condition_number: float | None
    rejection_reasons: tuple[str, ...]


def estimate_spherical_geometry(
    object_points_xyz: np.ndarray,
    *,
    timestamp_seconds: float,
    config: SphereFitConfig | None = None,
) -> SphericalGeometryEvidence:
    """Fit one well-conditioned curved surface; reject flat/narrow/noisy fits.

    Algebraic least squares is centered for numerical stability. All supplied
    returns must already belong to the same sensor-associated object. The 95th
    percentile residual and minimum normal spans gate extrapolation. Engineering
    interval allowances are not calibrated statistical confidence intervals.
    """
    settings = config or SphereFitConfig()
    if not isinstance(settings, SphereFitConfig):
        raise TypeError("config must be SphereFitConfig")
    timestamp = float(timestamp_seconds)
    if not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError("timestamp_seconds must be finite and nonnegative")
    points = np.asarray(object_points_xyz, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("object_points_xyz must be finite Nx3")

    def rejected(reason, residual=None, condition=None):
        return SphericalGeometryEvidence(
            ScalarEstimate.unavailable(
                unit="m",
                source=SPHERE_VOLUME_SOURCE,
                timestamp_seconds=timestamp,
                evidence=(reason,),
            ),
            ScalarEstimate.unavailable(
                unit="m^3",
                source=SPHERE_VOLUME_SOURCE,
                timestamp_seconds=timestamp,
                evidence=(reason,),
            ),
            None,
            residual,
            condition,
            (reason,),
        )

    if len(points) < settings.minimum_points:
        return rejected("insufficient associated LiDAR returns")
    origin = points.mean(axis=0)
    centered = points - origin
    condition = float(np.linalg.cond(centered))
    if not math.isfinite(condition) or condition > settings.maximum_condition_number:
        return rejected("surface is flat/narrow or sphere fit is ill-conditioned")
    solution, _, rank, _ = np.linalg.lstsq(
        np.column_stack((2 * centered, np.ones(len(points)))),
        np.sum(centered**2, axis=1),
        rcond=None,
    )
    radius_squared = solution[3] + solution[:3] @ solution[:3]
    if rank < 4 or radius_squared <= 0:
        return rejected("degenerate sphere fit", condition=condition)
    radius = math.sqrt(radius_squared)
    if not settings.minimum_radius_m <= radius <= settings.maximum_radius_m:
        return rejected("fitted radius outside configured scope", condition=condition)
    vectors = centered - solution[:3]
    residual_m = float(np.quantile(abs(np.linalg.norm(vectors, axis=1) - radius), 0.95))
    relative_residual = residual_m / radius
    if relative_residual > settings.maximum_relative_residual:
        return rejected(
            "surface does not support spherical geometry", relative_residual, condition
        )
    spans = np.ptp(vectors / radius, axis=0)
    if spans.min() < settings.minimum_normal_span:
        return rejected(
            "insufficient curved-surface coverage", relative_residual, condition
        )
    allowance = max(
        settings.sensor_allowance_m + settings.relative_radius_allowance * radius,
        2 * residual_m * condition,
    )
    lower, upper = max(0.0, radius - allowance), radius + allowance
    confidence = settings.confidence_cap * max(
        0.1, 1 - relative_residual / settings.maximum_relative_residual
    )
    evidence = (
        f"associated_returns={len(points)}; condition={condition:.6g}; p95_relative_residual={relative_residual:.6g}",
        f"surface_normal_spans={spans.tolist()}",
        "single-sphere shape hypothesis extrapolates unobserved back surface; not complete measured geometry",
        "engineering radius allowance includes residual, conditioning and configured sensor/model allowances",
        "occupied material fraction and internal contents are not observed",
    )
    radius_estimate = ScalarEstimate(
        radius,
        lower,
        upper,
        confidence,
        "m",
        SPHERE_VOLUME_SOURCE,
        timestamp,
        EstimateStatus.INFERRED,
        evidence,
    )
    volumes = [4 * math.pi * value**3 / 3 for value in (radius, lower, upper)]
    volume = ScalarEstimate(
        *volumes,
        confidence,
        "m^3",
        SPHERE_VOLUME_SOURCE,
        timestamp,
        EstimateStatus.INFERRED,
        evidence,
    )
    return SphericalGeometryEvidence(
        radius_estimate,
        volume,
        tuple(float(value) for value in origin + solution[:3]),
        relative_residual,
        condition,
        (),
    )


def estimate_spherical_mass(
    geometry: SphericalGeometryEvidence,
    *,
    effective_density_kg_m3: ScalarEstimate | None,
    timestamp_seconds: float,
) -> MassAttributes:
    """Mass = fitted envelope volume * explicit effective bulk density.

    Effective density includes hollow/solid construction and contents. There is
    deliberately no default material or density. Prior density keeps mass PRIOR.
    """
    if not isinstance(geometry, SphericalGeometryEvidence):
        raise TypeError("geometry must be SphericalGeometryEvidence")
    density = effective_density_kg_m3
    if density is None:
        density = ScalarEstimate.unavailable(
            unit="kg/m^3",
            source="effective_density_unavailable",
            timestamp_seconds=timestamp_seconds,
            evidence=("size alone cannot establish mass or material density",),
        )
    return estimate_mass_from_volume_density(
        geometry.volume_m3, density, timestamp_seconds=timestamp_seconds
    )

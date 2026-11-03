"""Offline thin-surface candidates, never runtime occupancy or factor evidence.

The 2 mm residual is a diagnostic search threshold, NOT a sensor accuracy
claim. Original height uncertainty is reported separately. No fixture pose,
class, mask, retained points or authored dimensions enter this extraction.
"""

from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial import cKDTree

from robot.perception.object_height import ObjectHeightConfig


@dataclass(frozen=True)
class NearFloorDiagnosticConfig:
    minimum_range_m: float = .5
    maximum_range_m: float = 4.
    candidate_residual_min_m: float = .002
    candidate_residual_max_m: float = .10
    cluster_radius_m: float = .12
    minimum_candidate_points: int = 3
    robust_fit_clip_floor_m: float = .005

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.maximum_range_m <= self.minimum_range_m:
            raise ValueError("range bounds must increase")
        if self.candidate_residual_max_m <= self.candidate_residual_min_m:
            raise ValueError("residual bounds must increase")
        if not isinstance(self.minimum_candidate_points, int) or self.minimum_candidate_points < 3:
            raise ValueError("minimum candidate support must be an integer >= 3")


def diagnose_near_floor(scene_xyz, *, floor_reference_m, reference_scope,
                        source_time, observation_time, config=None):
    """Return indices into ONE original gravity-aligned scan, with explicit scope.

    A declared floor reference selects a band only; the fitted plane comes from
    actual returns. Robust trimming is for proposal discovery, not a validated
    uncertainty reduction. Even a height-resolved candidate is not an accepted
    object: RGB ownership, dimensions, temporal support and runtime integration
    are outside this function.
    """
    settings = config or NearFloorDiagnosticConfig()
    if not isinstance(settings, NearFloorDiagnosticConfig):
        raise TypeError("config must be NearFloorDiagnosticConfig")
    height = ObjectHeightConfig()
    points = np.asarray(scene_xyz, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all():
        raise ValueError("scene must be finite Nx3 current returns")
    for value in (floor_reference_m, source_time, observation_time):
        if isinstance(value, bool) or not math.isfinite(value):
            raise ValueError("reference and source times must be finite")
    if min(source_time, observation_time) < 0:
        raise ValueError("source times must be nonnegative")
    if not isinstance(reference_scope, str) or not reference_scope.strip():
        raise ValueError("explicit reference scope required")
    report = dict(source_time=float(source_time), observation_time=float(observation_time),
                  reference_scope=reference_scope, floor_reference_m=float(floor_reference_m),
                  scope="offline candidates only; no runtime occupancy, mass, CRA or fall-height evidence",
                  runtime_admissible=False, floor_fit_available=False, rejections=[], candidates=[])
    if not 0 <= observation_time - source_time <= height.maximum_evidence_age_seconds:
        report["rejections"].append("stale or future scan")
        return report
    radial = np.linalg.norm(points[:, :2], axis=1)
    radial_mask = (radial >= settings.minimum_range_m) & (radial <= settings.maximum_range_m)
    band = points[radial_mask & (abs(points[:, 2] - floor_reference_m) <= height.floor_reference_band_m)]
    if len(band) < height.minimum_floor_points:
        report["rejections"].append("insufficient floor-band returns")
        return report
    design = np.column_stack((band[:, :2], np.ones(len(band))))
    keep = abs(band[:, 2] - np.median(band[:, 2])) <= height.measurement_allowance_m
    for _ in range(5):
        if keep.sum() < height.minimum_floor_points:
            report["rejections"].append("insufficient robust floor support")
            return report
        plane, _, rank, _ = np.linalg.lstsq(design[keep], band[keep, 2], rcond=None)
        residual = band[:, 2] - design @ plane
        median = np.median(residual[keep])
        mad = np.median(abs(residual[keep] - median))
        keep = abs(residual - median) <= max(settings.robust_fit_clip_floor_m, 3 * 1.4826 * mad)
    if keep.sum() < height.minimum_floor_points:
        report["rejections"].append("insufficient robust floor support")
        return report
    plane, _, rank, _ = np.linalg.lstsq(design[keep], band[keep, 2], rcond=None)
    residual = band[:, 2] - design @ plane
    floor_points = band[keep]
    rms = float(np.sqrt(np.mean(residual[keep] ** 2)))
    eigenvalues = np.linalg.eigvalsh(np.cov(floor_points[:, :2], rowvar=False, bias=True))
    if (rank < 3 or np.any(np.ptp(floor_points[:, :2], axis=0) < height.minimum_floor_span_m)
        or eigenvalues[-1] <= 1e-12
        or eigenvalues[0] / eigenvalues[-1] < height.minimum_xy_eigenvalue_ratio
        or rms > height.maximum_plane_residual_m
        or np.linalg.norm(plane[:2]) > height.maximum_plane_slope):
        report["rejections"].append("floor fit lacks planar two-dimensional support")
        return report
    report.update(floor_fit_available=True, floor_points=int(keep.sum()), plane=plane.tolist(),
                  floor_rms_m=rms, measurement_allowance_m=height.measurement_allowance_m)
    residual = points[:, 2] - np.column_stack((points[:, :2], np.ones(len(points)))) @ plane
    indices = np.flatnonzero(radial_mask & (residual > settings.candidate_residual_min_m)
                            & (residual <= settings.candidate_residual_max_m))
    tree = cKDTree(points[indices])
    unseen = set(range(len(indices)))
    groups = []
    while unseen:
        seed = min(unseen)
        unseen.remove(seed)
        group, pending = [seed], [seed]
        while pending:
            neighbors = unseen.intersection(tree.query_ball_point(points[indices[pending.pop()]], settings.cluster_radius_m))
            unseen.difference_update(neighbors)
            group.extend(sorted(neighbors))
            pending.extend(sorted(neighbors))
        if len(group) >= settings.minimum_candidate_points:
            groups.append(np.sort(indices[group]))
    floor_tree = cKDTree(floor_points[:, :2])
    for group in sorted(groups, key=lambda g: (-len(g), int(g[0]))):
        nearest = float(floor_tree.query(points[group, :2])[0].max())
        allowance = 2 * height.measurement_allowance_m + 2 * rms + nearest * height.extrapolation_allowance_per_m
        lower = float(residual[group].min() - allowance)
        report["candidates"].append(dict(indices=group.tolist(), point_count=len(group),
            residual_range_m=[float(residual[group].min()), float(residual[group].max())],
            centroid_xyz=points[group].mean(axis=0).tolist(), visible_extent_xyz_m=np.ptp(points[group], axis=0).tolist(),
            nearest_floor_max_m=nearest, original_height_allowance_m=allowance,
            residual_lower_bound_m=lower,
            height_resolved_under_original_allowance=bool(lower > 0 and nearest <= height.maximum_floor_extrapolation_m),
            runtime_admissible=False))
    return report

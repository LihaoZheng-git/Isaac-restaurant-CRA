"""Evaluate LiDAR track velocity against a separate motion oracle."""

from dataclasses import dataclass
import math

import numpy as np


LIDAR_TRACK_VELOCITY_SOURCE = "lidar_centroid_tracking"


@dataclass(frozen=True)
class MotionDiagnosticReport:
    """Offline score for one controlled moving-object LiDAR experiment."""

    expected_speed_mps: float
    moving_estimate_count: int
    median_estimated_speed_mps: float
    median_velocity_error_mps: float
    first_confirmed_motion_seconds: float | None
    final_stationary_speed_mps: float
    proxy_track_id_switches: int
    minimum_association_error_m: float
    evidence_source: str
    oracle_role: str
    blockers: tuple[str, ...]
    passed: bool


def evaluate_motion_diagnostic(
    estimated_velocities_world_xy: np.ndarray,
    *,
    expected_velocity_world_xy: tuple[float, float],
    first_confirmed_motion_seconds: float | None,
    final_stationary_speed_mps: float,
    proxy_track_id_switches: int,
    minimum_association_error_m: float,
    minimum_estimates: int = 10,
    maximum_median_velocity_error_mps: float = 0.25,
    maximum_confirmation_latency_seconds: float = 1.0,
    maximum_final_stationary_speed_mps: float = 0.15,
    maximum_track_id_switches: int = 2,
) -> MotionDiagnosticReport:
    """Score inferred track motion without feeding oracle motion to tracking."""

    estimates = np.asarray(estimated_velocities_world_xy, dtype=np.float64)
    expected = np.asarray(expected_velocity_world_xy, dtype=np.float64)
    if estimates.ndim != 2 or estimates.shape[1:] != (2,):
        raise ValueError("estimated_velocities_world_xy must have shape (N, 2)")
    if expected.shape != (2,) or not np.isfinite(expected).all():
        raise ValueError("expected_velocity_world_xy must contain two finite values")
    if not np.isfinite(estimates).all():
        raise ValueError("estimated velocities must be finite")
    if minimum_estimates <= 0:
        raise ValueError("minimum_estimates must be positive")
    scalar_values = {
        "final_stationary_speed_mps": final_stationary_speed_mps,
        "minimum_association_error_m": minimum_association_error_m,
        "maximum_median_velocity_error_mps": maximum_median_velocity_error_mps,
        "maximum_confirmation_latency_seconds": maximum_confirmation_latency_seconds,
        "maximum_final_stationary_speed_mps": maximum_final_stationary_speed_mps,
    }
    if any(
        not math.isfinite(float(value)) or value < 0
        for value in scalar_values.values()
    ):
        raise ValueError("diagnostic scalar values must be finite and nonnegative")
    if first_confirmed_motion_seconds is not None and (
        not math.isfinite(first_confirmed_motion_seconds)
        or first_confirmed_motion_seconds < 0
    ):
        raise ValueError("first_confirmed_motion_seconds must be nonnegative")
    if proxy_track_id_switches < 0 or maximum_track_id_switches < 0:
        raise ValueError("track switch counts must be nonnegative")

    expected_speed = float(np.linalg.norm(expected))
    if estimates.size:
        estimated_speeds = np.linalg.norm(estimates, axis=1)
        velocity_errors = np.linalg.norm(estimates - expected, axis=1)
        median_speed = float(np.median(estimated_speeds))
        median_error = float(np.median(velocity_errors))
    else:
        median_speed = math.inf
        median_error = math.inf

    blockers: list[str] = []
    if estimates.shape[0] < minimum_estimates:
        blockers.append("insufficient moving-object velocity estimates")
    if median_error > maximum_median_velocity_error_mps:
        blockers.append("median moving-object velocity error exceeds tolerance")
    if first_confirmed_motion_seconds is None:
        blockers.append("motion was never confirmed")
    elif first_confirmed_motion_seconds > maximum_confirmation_latency_seconds:
        blockers.append("motion confirmation latency exceeds tolerance")
    if final_stationary_speed_mps > maximum_final_stationary_speed_mps:
        blockers.append("tracker did not return to stationary after motion stopped")
    if proxy_track_id_switches > maximum_track_id_switches:
        blockers.append("proxy track identity was unstable")

    return MotionDiagnosticReport(
        expected_speed_mps=expected_speed,
        moving_estimate_count=int(estimates.shape[0]),
        median_estimated_speed_mps=median_speed,
        median_velocity_error_mps=median_error,
        first_confirmed_motion_seconds=first_confirmed_motion_seconds,
        final_stationary_speed_mps=float(final_stationary_speed_mps),
        proxy_track_id_switches=int(proxy_track_id_switches),
        minimum_association_error_m=float(minimum_association_error_m),
        evidence_source=LIDAR_TRACK_VELOCITY_SOURCE,
        oracle_role="offline_scoring_and_track_association_only",
        blockers=tuple(blockers),
        passed=not blockers,
    )

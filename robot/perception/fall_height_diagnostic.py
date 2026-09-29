"""Audit one stationary sensor-derived fall-height diagnostic."""

from __future__ import annotations

from dataclasses import dataclass
import math

from robot.perception.fall_height import FALL_HEIGHT_SOURCE
from robot.perception.lidar_fall_surfaces import LIDAR_SURFACE_SOURCE
from robot.perception.physical_attributes import EstimateStatus, ScalarEstimate


@dataclass(frozen=True)
class FallHeightDiagnosticReport:
    estimated_fall_height_m: float | None
    oracle_fall_height_m: float
    absolute_error_m: float | None
    interval_covers_oracle: bool | None
    blockers: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.blockers


def evaluate_fall_height_diagnostic(
    estimate: ScalarEstimate,
    *,
    authored_support_height_m: float,
    authored_landing_height_m: float,
    maximum_absolute_error_m: float = 0.10,
) -> FallHeightDiagnosticReport:
    """Compare sensor inference with oracle geometry used only for scoring."""

    if not isinstance(estimate, ScalarEstimate):
        raise TypeError("estimate must be ScalarEstimate")
    support = float(authored_support_height_m)
    landing = float(authored_landing_height_m)
    tolerance = float(maximum_absolute_error_m)
    if not all(math.isfinite(value) for value in (support, landing, tolerance)):
        raise ValueError("oracle heights and tolerance must be finite")
    if tolerance <= 0:
        raise ValueError("maximum_absolute_error_m must be positive")
    oracle = max(0.0, support - landing)
    blockers = []
    if oracle <= 0:
        blockers.append("authored fixture does not contain a positive fall")
    if not estimate.is_available:
        blockers.append("sensor-derived fall height is unavailable")
        return FallHeightDiagnosticReport(None, oracle, None, None, tuple(blockers))
    if estimate.status is not EstimateStatus.INFERRED:
        blockers.append("fall height is not identified as inferred evidence")
    if estimate.source != FALL_HEIGHT_SOURCE:
        blockers.append("fall height does not use the support-to-landing derivation")
    expected_surface_source = f"source={LIDAR_SURFACE_SOURCE}"
    if sum(expected_surface_source in item for item in estimate.evidence) < 2:
        blockers.append("support and landing provenance is not LiDAR-derived")
    error = abs(estimate.value - oracle)
    if error > tolerance:
        blockers.append(
            f"absolute fall-height error exceeds tolerance: {error:.4f} > {tolerance:.4f} m"
        )
    coverage = estimate.lower_bound <= oracle <= estimate.upper_bound
    if not coverage:
        blockers.append("fall-height interval does not cover authored oracle")
    return FallHeightDiagnosticReport(
        estimate.value,
        oracle,
        error,
        coverage,
        tuple(blockers),
    )

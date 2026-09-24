"""Summarize stationary tracker speeds into a configurable noise-floor gate."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class StationaryMotionCalibrationConfig:
    quantile: float = 0.95
    safety_margin_mps: float = 0.02
    minimum_samples: int = 20

    def __post_init__(self) -> None:
        quantile = float(self.quantile)
        margin = float(self.safety_margin_mps)
        if not math.isfinite(quantile) or not 0.5 <= quantile < 1:
            raise ValueError("quantile must lie in [0.5, 1)")
        if not math.isfinite(margin) or margin < 0:
            raise ValueError("safety_margin_mps must be finite and nonnegative")
        if (
            isinstance(self.minimum_samples, bool)
            or not isinstance(self.minimum_samples, int)
            or self.minimum_samples <= 0
        ):
            raise ValueError("minimum_samples must be a positive integer")
        object.__setattr__(self, "quantile", quantile)
        object.__setattr__(self, "safety_margin_mps", margin)


@dataclass(frozen=True)
class StationaryMotionCalibration:
    sample_count: int
    median_speed_mps: float
    p95_speed_mps: float
    configured_quantile_speed_mps: float
    maximum_speed_mps: float
    recommended_stationary_threshold_mps: float
    config: StationaryMotionCalibrationConfig


def calibrate_stationary_motion_threshold(
    speeds_mps: np.ndarray,
    *,
    config: StationaryMotionCalibrationConfig | None = None,
) -> StationaryMotionCalibration:
    """Return a robust stationary threshold from raw matched-track speeds."""
    settings = config or StationaryMotionCalibrationConfig()
    if not isinstance(settings, StationaryMotionCalibrationConfig):
        raise TypeError("config must be StationaryMotionCalibrationConfig")
    speeds = np.asarray(speeds_mps, dtype=np.float64)
    if speeds.ndim != 1:
        raise ValueError("speeds_mps must have shape (N,)")
    if not np.isfinite(speeds).all() or np.any(speeds < 0):
        raise ValueError("speeds_mps must contain finite nonnegative values")
    if speeds.size < settings.minimum_samples:
        raise ValueError(
            f"at least {settings.minimum_samples} stationary samples are required"
        )
    configured_quantile = float(np.quantile(speeds, settings.quantile))
    return StationaryMotionCalibration(
        sample_count=int(speeds.size),
        median_speed_mps=float(np.median(speeds)),
        p95_speed_mps=float(np.quantile(speeds, 0.95)),
        configured_quantile_speed_mps=configured_quantile,
        maximum_speed_mps=float(np.max(speeds)),
        recommended_stationary_threshold_mps=(
            configured_quantile + settings.safety_margin_mps
        ),
        config=settings,
    )

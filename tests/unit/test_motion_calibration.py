import numpy as np
import pytest

from robot.perception.motion_calibration import (
    StationaryMotionCalibrationConfig,
    calibrate_stationary_motion_threshold,
)


def test_stationary_quantile_and_margin_are_reported():
    speeds = np.linspace(0, 0.10, 101)
    result = calibrate_stationary_motion_threshold(
        speeds,
        config=StationaryMotionCalibrationConfig(
            quantile=0.95, safety_margin_mps=0.02, minimum_samples=20
        ),
    )
    assert result.sample_count == 101
    assert result.median_speed_mps == pytest.approx(0.05)
    assert result.p95_speed_mps == pytest.approx(0.095)
    assert result.configured_quantile_speed_mps == pytest.approx(0.095)
    assert result.recommended_stationary_threshold_mps == pytest.approx(0.115)


def test_insufficient_invalid_and_negative_samples_are_rejected():
    with pytest.raises(ValueError, match="at least"):
        calibrate_stationary_motion_threshold(np.zeros(3))
    with pytest.raises(ValueError, match="shape"):
        calibrate_stationary_motion_threshold(np.zeros((20, 1)))
    with pytest.raises(ValueError, match="nonnegative"):
        calibrate_stationary_motion_threshold(np.full(20, -0.1))


@pytest.mark.parametrize(
    "changes",
    ({"quantile": 1}, {"safety_margin_mps": -1}, {"minimum_samples": 0}),
)
def test_configuration_rejects_invalid_values(changes):
    with pytest.raises(ValueError):
        StationaryMotionCalibrationConfig(**changes)

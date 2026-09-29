import math

import numpy as np
import pytest

from robot.perception.motion_diagnostic import evaluate_motion_diagnostic


def valid_estimates() -> np.ndarray:
    return np.tile(np.array([[0.0, -0.58]]), (12, 1))


def test_motion_diagnostic_accepts_accurate_sensor_estimates() -> None:
    report = evaluate_motion_diagnostic(
        valid_estimates(),
        expected_velocity_world_xy=(0.0, -0.6),
        first_confirmed_motion_seconds=0.35,
        final_stationary_speed_mps=0.02,
        proxy_track_id_switches=0,
        minimum_association_error_m=0.04,
    )

    assert report.passed
    assert report.blockers == ()
    assert report.evidence_source == "lidar_centroid_tracking"
    assert report.oracle_role == "offline_scoring_and_track_association_only"
    assert report.median_velocity_error_mps == pytest.approx(0.02)


@pytest.mark.parametrize(
    ("changes", "blocker"),
    [
        ({"estimates": np.zeros((2, 2))}, "insufficient moving-object"),
        ({"estimates": np.tile([[0.6, 0.0]], (12, 1))}, "velocity error"),
        ({"latency": None}, "never confirmed"),
        ({"latency": 1.2}, "latency"),
        ({"stationary": 0.2}, "return to stationary"),
        ({"switches": 3}, "identity"),
    ],
)
def test_motion_diagnostic_reports_each_failed_gate(changes, blocker) -> None:
    report = evaluate_motion_diagnostic(
        changes.get("estimates", valid_estimates()),
        expected_velocity_world_xy=(0.0, -0.6),
        first_confirmed_motion_seconds=changes.get("latency", 0.35),
        final_stationary_speed_mps=changes.get("stationary", 0.02),
        proxy_track_id_switches=changes.get("switches", 0),
        minimum_association_error_m=0.04,
    )

    assert not report.passed
    assert any(blocker in item for item in report.blockers)


def test_motion_diagnostic_rejects_invalid_shapes_and_values() -> None:
    kwargs = dict(
        expected_velocity_world_xy=(0.0, -0.6),
        first_confirmed_motion_seconds=0.2,
        final_stationary_speed_mps=0.0,
        proxy_track_id_switches=0,
        minimum_association_error_m=0.0,
    )
    with pytest.raises(ValueError, match="shape"):
        evaluate_motion_diagnostic(np.zeros(3), **kwargs)
    with pytest.raises(ValueError, match="finite"):
        evaluate_motion_diagnostic(np.array([[math.nan, 0.0]]), **kwargs)

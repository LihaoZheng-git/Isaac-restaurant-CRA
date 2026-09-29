import pytest

from robot.perception.fall_height_diagnostic import evaluate_fall_height_diagnostic
from robot.perception.fall_height import FALL_HEIGHT_SOURCE
from robot.perception.lidar_fall_surfaces import LIDAR_SURFACE_SOURCE
from robot.perception.physical_attributes import EstimateStatus, ScalarEstimate


def estimate(value=0.8, *, source=FALL_HEIGHT_SOURCE, status=EstimateStatus.INFERRED):
    return ScalarEstimate(
        value,
        value - 0.05,
        value + 0.05,
        0.55,
        "m",
        source,
        2,
        status,
        (
            f"support source={LIDAR_SURFACE_SOURCE}; status=inferred",
            f"landing source={LIDAR_SURFACE_SOURCE}; status=inferred",
        ),
    )


def test_sensor_inference_passes_against_separate_offline_oracle():
    report = evaluate_fall_height_diagnostic(
        estimate(), authored_support_height_m=0.8, authored_landing_height_m=0.0
    )

    assert report.passed
    assert report.absolute_error_m == pytest.approx(0.0)
    assert report.interval_covers_oracle


def test_unavailable_estimate_remains_blocked():
    report = evaluate_fall_height_diagnostic(
        ScalarEstimate.unavailable(unit="m", source=FALL_HEIGHT_SOURCE, timestamp_seconds=2),
        authored_support_height_m=0.8,
        authored_landing_height_m=0.0,
    )

    assert not report.passed
    assert report.estimated_fall_height_m is None


@pytest.mark.parametrize(
    "candidate",
    [
        estimate(source="object_height"),
        estimate(status=EstimateStatus.ORACLE),
        estimate(0.5),
    ],
)
def test_wrong_source_status_or_large_error_blocks_claim(candidate):
    report = evaluate_fall_height_diagnostic(
        candidate, authored_support_height_m=0.8, authored_landing_height_m=0.0
    )

    assert not report.passed


def test_nonpositive_oracle_fixture_cannot_validate_positive_fall():
    report = evaluate_fall_height_diagnostic(
        estimate(0.0), authored_support_height_m=0.0, authored_landing_height_m=0.0
    )

    assert not report.passed
    assert "positive fall" in report.blockers[0]

import pytest

from robot.perception.fall_height import FallSurfaceEvidence, estimate_fall_height
from robot.perception.physical_attributes import EstimateStatus, ScalarEstimate


def scalar(
    value,
    lower,
    upper,
    *,
    status=EstimateStatus.MEASURED,
    source="depth_plane",
    timestamp=2,
):
    return ScalarEstimate(value, lower, upper, 0.8, "m", source, timestamp, status)


def test_surface_drop_propagates_interval_and_provenance():
    result = estimate_fall_height(
        FallSurfaceEvidence(scalar(1.0, 0.9, 1.1), scalar(0.2, 0.1, 0.3)),
        timestamp_seconds=2,
    )
    assert result.value == pytest.approx(0.8)
    assert result.lower_bound == pytest.approx(0.6)
    assert result.upper_bound == pytest.approx(1.0)
    assert result.status is EstimateStatus.MEASURED
    assert "not object height" in result.evidence[-1]


def test_overlap_clips_drop_at_zero():
    result = estimate_fall_height(
        FallSurfaceEvidence(scalar(0.1, 0, 0.2), scalar(0.2, 0.1, 0.3)),
        timestamp_seconds=2,
    )
    assert result.value == result.lower_bound == 0
    assert result.upper_bound == pytest.approx(0.1)


def test_missing_surface_is_unavailable():
    assert not estimate_fall_height(None, timestamp_seconds=2).is_available


def test_inferred_surface_marks_result_inferred():
    result = estimate_fall_height(
        FallSurfaceEvidence(
            scalar(1, 1, 1), scalar(0, 0, 0, status=EstimateStatus.INFERRED)
        ),
        timestamp_seconds=2,
    )
    assert result.status is EstimateStatus.INFERRED


def test_prior_surfaces_stay_prior():
    result = estimate_fall_height(
        FallSurfaceEvidence(
            scalar(1, 1, 1, status=EstimateStatus.PRIOR),
            scalar(0, 0, 0, status=EstimateStatus.PRIOR),
        ),
        timestamp_seconds=2,
    )
    assert result.status is EstimateStatus.PRIOR


def test_oracle_surface_dependency_remains_oracle():
    result = estimate_fall_height(
        FallSurfaceEvidence(
            scalar(1, 1, 1, status=EstimateStatus.ORACLE),
            scalar(0, 0, 0, status=EstimateStatus.MEASURED),
        ),
        timestamp_seconds=2,
    )
    assert result.status is EstimateStatus.ORACLE


def test_units_and_future_timestamp_rejected():
    wrong = ScalarEstimate(1, 1, 1, 1, "kg", "bad", 2)
    with pytest.raises(ValueError):
        FallSurfaceEvidence(wrong, scalar(0, 0, 0))
    with pytest.raises(ValueError):
        estimate_fall_height(
            FallSurfaceEvidence(scalar(1, 1, 1), scalar(0, 0, 0)), timestamp_seconds=1
        )


def test_fall_height_keeps_oldest_contributing_timestamp():
    result = estimate_fall_height(
        FallSurfaceEvidence(
            scalar(1, 1, 1, timestamp=1.5),
            scalar(0, 0, 0, timestamp=1.0),
        ),
        timestamp_seconds=2,
    )

    assert result.timestamp_seconds == 1.0

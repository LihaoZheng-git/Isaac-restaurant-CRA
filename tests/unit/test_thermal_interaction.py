from dataclasses import replace

import pytest

from robot.perception.physical_attributes import (
    EstimateStatus,
    PhysicalAttributes,
    ScalarEstimate,
)
from robot.perception.thermal_interaction import (
    THERMAL_ROUTE_INTERACTION_SOURCE,
    estimate_thermal_route_interaction,
)


def cue(value, *, timestamp=2.0, status=EstimateStatus.INFERRED):
    return ScalarEstimate(
        value,
        max(0.0, value - 0.1),
        min(1.0, value + 0.1),
        0.8,
        "1",
        "fixture",
        timestamp,
        status,
    )


def thermal(value=0.7, *, timestamp=2.0):
    unknown = PhysicalAttributes.unknown(track_id=4, timestamp_seconds=timestamp)
    return replace(unknown.thermal, heat_source_likelihood=cue(value, timestamp=timestamp))


def test_same_track_route_score_becomes_continuous_thermal_interaction():
    result = estimate_thermal_route_interaction(
        thermal(), cue(0.35), timestamp_seconds=2.1
    )
    assert result.value == pytest.approx(0.35)
    assert result.lower_bound == pytest.approx(0.25)
    assert result.source == THERMAL_ROUTE_INTERACTION_SOURCE
    assert result.status is EstimateStatus.INFERRED
    assert "no carried-state" in result.evidence[-2]


def test_fallback_route_provenance_is_preserved():
    result = estimate_thermal_route_interaction(
        thermal(),
        cue(0.2, status=EstimateStatus.FALLBACK),
        timestamp_seconds=2.1,
    )
    assert result.status is EstimateStatus.FALLBACK


def test_missing_or_stale_evidence_stays_unavailable():
    unknown = PhysicalAttributes.unknown(track_id=4, timestamp_seconds=2.0)
    assert not estimate_thermal_route_interaction(
        unknown.thermal, cue(0.4), timestamp_seconds=2.1
    ).is_available
    assert not estimate_thermal_route_interaction(
        thermal(timestamp=1.0), cue(0.4, timestamp=1.0), timestamp_seconds=2.0
    ).is_available


def test_dimension_and_age_validation():
    with pytest.raises(ValueError, match="dimensionless"):
        estimate_thermal_route_interaction(
            thermal(),
            ScalarEstimate(1, 1, 1, 1, "m", "fixture", 2),
            timestamp_seconds=2,
        )
    with pytest.raises(ValueError, match="positive"):
        estimate_thermal_route_interaction(
            thermal(), cue(0.4), timestamp_seconds=2, maximum_evidence_age_seconds=0
        )

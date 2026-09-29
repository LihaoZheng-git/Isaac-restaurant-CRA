"""Derive route interaction for an already associated visual thermal source."""

from __future__ import annotations

import math

from robot.perception.physical_attributes import (
    ScalarEstimate,
    ThermalAttributes,
    combine_estimate_statuses,
)


THERMAL_ROUTE_INTERACTION_SOURCE = "thermal_track_waypoint_interaction"


def estimate_thermal_route_interaction(
    thermal: ThermalAttributes,
    route_crossing_likelihood: ScalarEstimate,
    *,
    timestamp_seconds: float,
    maximum_evidence_age_seconds: float = 0.5,
) -> ScalarEstimate:
    """Reuse same-track route geometry as continuous thermal accessibility.

    The heat cue supplies severity elsewhere. This function only states how
    strongly the associated track overlaps the current waypoint corridor. It
    does not infer carriage, spill, temperature, or burn severity.
    """

    if not isinstance(thermal, ThermalAttributes):
        raise TypeError("thermal must be ThermalAttributes")
    if not isinstance(route_crossing_likelihood, ScalarEstimate):
        raise TypeError("route_crossing_likelihood must be ScalarEstimate")
    if route_crossing_likelihood.unit != "1":
        raise ValueError("route_crossing_likelihood must be dimensionless")
    timestamp = float(timestamp_seconds)
    maximum_age = float(maximum_evidence_age_seconds)
    if not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError("timestamp_seconds must be finite and nonnegative")
    if not math.isfinite(maximum_age) or maximum_age <= 0:
        raise ValueError("maximum_evidence_age_seconds must be positive")

    heat = thermal.heat_source_likelihood
    required = (
        ("heat_source_likelihood", heat),
        ("route_crossing_likelihood", route_crossing_likelihood),
    )
    reasons = []
    for name, item in required:
        if not item.is_available:
            reasons.append(f"{name}: unavailable")
        elif item.timestamp_seconds > timestamp:
            reasons.append(f"{name}: future evidence")
        elif timestamp - item.timestamp_seconds > maximum_age:
            reasons.append(f"{name}: stale evidence")
    provenance = tuple(
        f"{name}: status={item.status.value}, source={item.source}, "
        f"timestamp={item.timestamp_seconds}, confidence={item.confidence}"
        for name, item in required
    )
    limits = (
        "same-track waypoint-corridor score is used as thermal interaction",
        "no carried-state or spill-state inference is made",
        "thermal interaction is not temperature or burn severity",
    )
    if reasons:
        return ScalarEstimate.unavailable(
            unit="1",
            source=THERMAL_ROUTE_INTERACTION_SOURCE,
            timestamp_seconds=timestamp,
            evidence=provenance + tuple(reasons) + limits,
        )
    return ScalarEstimate(
        value=route_crossing_likelihood.value,
        lower_bound=route_crossing_likelihood.lower_bound,
        upper_bound=route_crossing_likelihood.upper_bound,
        confidence=min(heat.confidence, route_crossing_likelihood.confidence),
        unit="1",
        source=THERMAL_ROUTE_INTERACTION_SOURCE,
        timestamp_seconds=timestamp,
        status=combine_estimate_statuses(item.status for _, item in required),
        evidence=provenance + limits,
    )

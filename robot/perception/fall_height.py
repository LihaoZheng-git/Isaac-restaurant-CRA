"""Derive fall height only from explicit support and landing-surface evidence."""

from dataclasses import dataclass

from robot.perception.physical_attributes import (
    ScalarEstimate,
    combine_estimate_statuses,
)


FALL_HEIGHT_SOURCE = "support_to_landing_surface_geometry"


@dataclass(frozen=True)
class FallSurfaceEvidence:
    """Vertical support and landing heights in one calibrated metric frame."""

    support_height_m: ScalarEstimate
    landing_height_m: ScalarEstimate

    def __post_init__(self) -> None:
        for name, item in (
            ("support_height_m", self.support_height_m),
            ("landing_height_m", self.landing_height_m),
        ):
            if not isinstance(item, ScalarEstimate):
                raise TypeError(f"{name} must be a ScalarEstimate")
            if item.unit != "m":
                raise ValueError(f"{name} must use unit 'm'")


def estimate_fall_height(
    evidence: FallSurfaceEvidence | None, *, timestamp_seconds: float
) -> ScalarEstimate:
    """Return support-to-landing drop; unavailable evidence stays unavailable."""
    timestamp = float(timestamp_seconds)
    if timestamp < 0:
        raise ValueError("timestamp_seconds cannot be negative")
    if evidence is None or not (
        evidence.support_height_m.is_available
        and evidence.landing_height_m.is_available
    ):
        return ScalarEstimate.unavailable(
            unit="m",
            source=FALL_HEIGHT_SOURCE,
            timestamp_seconds=timestamp,
            evidence=("support and landing surfaces must both be localized",),
        )
    support = evidence.support_height_m
    landing = evidence.landing_height_m
    if max(support.timestamp_seconds, landing.timestamp_seconds) > timestamp:
        raise ValueError("surface evidence cannot be newer than fall-height result")
    value = max(0.0, support.value - landing.value)
    lower = max(0.0, support.lower_bound - landing.upper_bound)
    upper = max(0.0, support.upper_bound - landing.lower_bound)
    status = combine_estimate_statuses((support.status, landing.status))
    return ScalarEstimate(
        value,
        lower,
        upper,
        min(support.confidence, landing.confidence),
        "m",
        FALL_HEIGHT_SOURCE,
        timestamp,
        status,
        (
            f"support source={support.source}; status={support.status.value}",
            f"landing source={landing.source}; status={landing.status.value}",
            "fall height is vertical support-to-landing drop, not object height",
        ),
    )

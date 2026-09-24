"""Contextual visual cue adapter for the bounded hot-soup thermal example.

The adapter does not estimate temperature.  It requires a visible plume, an
open-container cue, and evidence that the plume is above the container before
publishing an inferred heat-source proxy.  A generic smoke/plume detection by
itself therefore remains insufficient for the CRA thermal factor.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from robot.perception.physical_attributes import (
    CategoricalEstimate,
    ScalarEstimate,
    ThermalAttributes,
    combine_estimate_statuses,
)


HOT_SOUP_SOURCE = "contextual_visible_hot_soup_plume"
SUPPORTED_PLUME_LABELS = frozenset({"steam", "smoke_like", "visible_plume"})


@dataclass(frozen=True)
class HotSoupCueConfig:
    """Configurable evidence age and source-type publication threshold."""

    maximum_evidence_age_seconds: float = 1.0
    source_type_lower_bound_threshold: float = 0.5

    def __post_init__(self) -> None:
        age = float(self.maximum_evidence_age_seconds)
        threshold = float(self.source_type_lower_bound_threshold)
        if not math.isfinite(age) or age <= 0:
            raise ValueError("maximum_evidence_age_seconds must be positive")
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError(
                "source_type_lower_bound_threshold must lie in [0, 1]"
            )
        object.__setattr__(self, "maximum_evidence_age_seconds", age)
        object.__setattr__(self, "source_type_lower_bound_threshold", threshold)


@dataclass(frozen=True)
class HotSoupCueResult:
    """Thermal contract plus explicit acceptance/rejection diagnostics."""

    thermal: ThermalAttributes
    accepted_context: bool
    plume_label: str
    rejection_reasons: tuple[str, ...]


def _unavailable(timestamp: float, source: str, *evidence: str) -> ScalarEstimate:
    return ScalarEstimate.unavailable(
        unit="1",
        source=source,
        timestamp_seconds=timestamp,
        evidence=evidence,
    )


def _validate_cue(name: str, cue: ScalarEstimate) -> None:
    if not isinstance(cue, ScalarEstimate):
        raise TypeError(f"{name} must be a ScalarEstimate")
    if cue.unit != "1":
        raise ValueError(f"{name} must be dimensionless")
    if cue.is_available and not 0 <= cue.lower_bound <= cue.upper_bound <= 1:
        raise ValueError(f"{name} bounds must lie in [0, 1]")


def estimate_hot_soup_cues(
    *,
    plume_likelihood: ScalarEstimate,
    open_container_likelihood: ScalarEstimate,
    plume_above_container_likelihood: ScalarEstimate,
    timestamp_seconds: float,
    plume_label: str = "visible_plume",
    carried_likelihood: ScalarEstimate | None = None,
    spill_likelihood: ScalarEstimate | None = None,
    config: HotSoupCueConfig | None = None,
) -> HotSoupCueResult:
    """Infer a contextual hot-soup proxy from three visual cue channels.

    ``plume_label`` records the fixed detector vocabulary.  ``smoke_like`` is
    allowed because a visible synthetic plume can resemble smoke, but it is not
    relabeled as steam.  The inferred heat-source score is the minimum of the
    three required likelihoods so one weak contextual cue cannot be hidden by
    two strong cues.
    """

    timestamp = float(timestamp_seconds)
    if not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError("timestamp_seconds must be finite and nonnegative")
    settings = config or HotSoupCueConfig()
    label = "_".join(str(plume_label).casefold().replace("-", " ").split())
    if label not in SUPPORTED_PLUME_LABELS:
        raise ValueError(
            f"plume_label must be one of {sorted(SUPPORTED_PLUME_LABELS)}"
        )

    required = (
        ("plume_likelihood", plume_likelihood),
        ("open_container_likelihood", open_container_likelihood),
        ("plume_above_container_likelihood", plume_above_container_likelihood),
    )
    optional = (
        (
            "carried_likelihood",
            carried_likelihood
            or _unavailable(timestamp, "carried_state_unavailable"),
        ),
        (
            "spill_likelihood",
            spill_likelihood or _unavailable(timestamp, "spill_state_unavailable"),
        ),
    )
    for name, cue in required + optional:
        _validate_cue(name, cue)

    reasons: list[str] = []
    for name, cue in required:
        if not cue.is_available:
            reasons.append(f"{name}: unavailable")
        elif cue.timestamp_seconds > timestamp:
            reasons.append(f"{name}: future evidence")
        elif timestamp - cue.timestamp_seconds > settings.maximum_evidence_age_seconds:
            reasons.append(f"{name}: stale evidence")

    provenance = tuple(
        f"{name}: status={cue.status.value}, source={cue.source}, "
        f"timestamp={cue.timestamp_seconds}, confidence={cue.confidence}"
        for name, cue in required
    )
    ambiguity = (
        f"fixed visual plume label={label}",
        "contextual visual proxy is not a temperature measurement",
        "visible plume does not prove steam rather than smoke or another aerosol",
    )
    heat_source = _unavailable(
        timestamp,
        HOT_SOUP_SOURCE,
        *(provenance + tuple(reasons) + ambiguity),
    )
    source_type = CategoricalEstimate.unavailable(
        source=HOT_SOUP_SOURCE,
        timestamp_seconds=timestamp,
        evidence=provenance + tuple(reasons) + ambiguity,
    )
    if not reasons:
        cues = tuple(cue for _, cue in required)
        assert all(cue.value is not None for cue in cues)
        assert all(cue.lower_bound is not None for cue in cues)
        assert all(cue.upper_bound is not None for cue in cues)
        heat_source = ScalarEstimate(
            value=min(cue.value for cue in cues),
            lower_bound=min(cue.lower_bound for cue in cues),
            upper_bound=min(cue.upper_bound for cue in cues),
            confidence=min(cue.confidence for cue in cues),
            unit="1",
            source=HOT_SOUP_SOURCE,
            timestamp_seconds=timestamp,
            status=combine_estimate_statuses(cue.status for cue in cues),
            evidence=provenance
            + ambiguity
            + ("minimum of plume, open-container, and above-container cues",),
        )
        if heat_source.lower_bound >= settings.source_type_lower_bound_threshold:
            source_type = CategoricalEstimate(
                value="visible_hot_soup_plume_proxy",
                confidence=heat_source.confidence,
                source=HOT_SOUP_SOURCE,
                timestamp_seconds=timestamp,
                status=heat_source.status,
                evidence=heat_source.evidence,
            )

    steam = (
        plume_likelihood
        if label == "steam"
        else _unavailable(
            timestamp,
            HOT_SOUP_SOURCE,
            f"detector label {label!r} is not a steam label",
            "smoke-like or generic plume cues are not relabeled as steam",
        )
    )
    carried = optional[0][1]
    spill = optional[1][1]
    return HotSoupCueResult(
        thermal=ThermalAttributes(
            heat_source_likelihood=heat_source,
            steam_likelihood=steam,
            open_container_likelihood=open_container_likelihood,
            carried_likelihood=carried,
            spill_likelihood=spill,
            source_type=source_type,
        ),
        accepted_context=not reasons,
        plume_label=label,
        rejection_reasons=tuple(reasons),
    )

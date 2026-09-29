"""Continuous contextual visual evidence for possible thermal hazards.

The adapter estimates how strongly an image supports a heat-source hypothesis.
It does not classify an object as hot, identify soup, or estimate temperature.
Complete context requires a visible plume-like cue, a source/container cue, and
their spatial association. Missing inputs remain unavailable.
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


CONTEXTUAL_THERMAL_SOURCE = "contextual_visual_thermal_evidence"
SUPPORTED_PLUME_LABELS = frozenset({"steam", "smoke_like", "visible_plume"})


@dataclass(frozen=True)
class ContextualThermalCueConfig:
    """Configure evidence freshness."""

    maximum_evidence_age_seconds: float = 1.0

    def __post_init__(self) -> None:
        age = float(self.maximum_evidence_age_seconds)
        if not math.isfinite(age) or age <= 0:
            raise ValueError("maximum_evidence_age_seconds must be positive")
        object.__setattr__(self, "maximum_evidence_age_seconds", age)


@dataclass(frozen=True)
class ContextualThermalCueResult:
    """Continuous thermal evidence plus availability diagnostics."""

    thermal: ThermalAttributes
    context_available: bool
    plume_label: str
    rejection_reasons: tuple[str, ...]

    @property
    def accepted_context(self) -> bool:
        """Compatibility diagnostic; this is not a hot/not-hot decision."""

        return self.context_available


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


def estimate_contextual_thermal_cues(
    *,
    plume_likelihood: ScalarEstimate,
    source_context_likelihood: ScalarEstimate,
    plume_source_association_likelihood: ScalarEstimate,
    timestamp_seconds: float,
    plume_label: str = "visible_plume",
    open_container_likelihood: ScalarEstimate | None = None,
    carried_likelihood: ScalarEstimate | None = None,
    spill_likelihood: ScalarEstimate | None = None,
    config: ContextualThermalCueConfig | None = None,
) -> ContextualThermalCueResult:
    """Infer a continuous heat-source proxy from contextual visual cues.

    The conservative minimum is a fuzzy conjunction: all three required cues
    retain their numeric strength, while one weak cue cannot be hidden by two
    strong cues. The result remains visual evidence, not a calibrated
    probability, temperature measurement, or semantic object classification.
    """

    timestamp = float(timestamp_seconds)
    if not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError("timestamp_seconds must be finite and nonnegative")
    settings = config or ContextualThermalCueConfig()
    label = "_".join(str(plume_label).casefold().replace("-", " ").split())
    if label not in SUPPORTED_PLUME_LABELS:
        raise ValueError(
            f"plume_label must be one of {sorted(SUPPORTED_PLUME_LABELS)}"
        )

    required = (
        ("plume_likelihood", plume_likelihood),
        ("source_context_likelihood", source_context_likelihood),
        (
            "plume_source_association_likelihood",
            plume_source_association_likelihood,
        ),
    )
    optional = (
        (
            "open_container_likelihood",
            open_container_likelihood
            or _unavailable(timestamp, "open_container_state_unavailable"),
        ),
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
        "no hot/not-hot or soup/not-soup classification is produced",
    )
    heat_source = _unavailable(
        timestamp,
        CONTEXTUAL_THERMAL_SOURCE,
        *(provenance + tuple(reasons) + ambiguity),
    )
    source_type = CategoricalEstimate.unavailable(
        source=CONTEXTUAL_THERMAL_SOURCE,
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
            source=CONTEXTUAL_THERMAL_SOURCE,
            timestamp_seconds=timestamp,
            status=combine_estimate_statuses(cue.status for cue in cues),
            evidence=provenance
            + ambiguity
            + (
                "minimum of plume, source-context, and spatial-association cues",
            ),
        )
        source_type = CategoricalEstimate(
            value="contextual_visible_plume_proxy",
            confidence=heat_source.confidence,
            source=CONTEXTUAL_THERMAL_SOURCE,
            timestamp_seconds=timestamp,
            status=heat_source.status,
            evidence=heat_source.evidence
            + ("descriptive source metadata; not a risk threshold",),
        )

    steam = (
        plume_likelihood
        if label == "steam"
        else _unavailable(
            timestamp,
            CONTEXTUAL_THERMAL_SOURCE,
            f"detector label {label!r} is not a steam label",
            "smoke-like or generic plume cues are not relabeled as steam",
        )
    )
    return ContextualThermalCueResult(
        thermal=ThermalAttributes(
            heat_source_likelihood=heat_source,
            steam_likelihood=steam,
            open_container_likelihood=optional[0][1],
            carried_likelihood=optional[1][1],
            spill_likelihood=optional[2][1],
            source_type=source_type,
        ),
        context_available=not reasons,
        plume_label=label,
        rejection_reasons=tuple(reasons),
    )

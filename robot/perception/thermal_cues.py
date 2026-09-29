"""Temporal aggregation for continuous visual thermal evidence."""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import median

from robot.perception.contextual_thermal_cues import (
    ContextualThermalCueConfig,
    estimate_contextual_thermal_cues,
)
from robot.perception.physical_attributes import (
    CategoricalEstimate,
    EstimateStatus,
    ScalarEstimate,
    ThermalAttributes,
    combine_estimate_statuses,
)


THERMAL_CUE_SOURCE = "temporal_visual_thermal_evidence"


@dataclass(frozen=True)
class ThermalCueFrame:
    timestamp_seconds: float
    flame_likelihood: float | None
    steam_likelihood: float | None
    model_id: str
    smoke_likelihood: float | None = None

    def __post_init__(self) -> None:
        for name in ("flame_likelihood", "steam_likelihood", "smoke_likelihood"):
            raw = getattr(self, name)
            if raw is None:
                continue
            value = float(raw)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
            object.__setattr__(self, name, value)
        timestamp = float(self.timestamp_seconds)
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("timestamp_seconds must be finite and nonnegative")
        object.__setattr__(self, "timestamp_seconds", timestamp)
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must be nonempty")


@dataclass(frozen=True)
class ThermalCueConfig:
    minimum_frame_count: int = 3
    maximum_window_seconds: float = 1.0
    score_uncertainty: float = 0.15


def _unavailable(timestamp: float, source: str = THERMAL_CUE_SOURCE) -> ScalarEstimate:
    return ScalarEstimate.unavailable(
        unit="1", source=source, timestamp_seconds=timestamp
    )


def _combine_heat_evidence(
    flame: ScalarEstimate,
    contextual_plume: ScalarEstimate,
    *,
    timestamp: float,
) -> ScalarEstimate:
    available = tuple(
        item for item in (flame, contextual_plume) if item.is_available
    )
    if not available:
        return _unavailable(timestamp)
    if len(available) == 1:
        return available[0]
    return ScalarEstimate(
        value=max(item.value for item in available),
        lower_bound=max(item.lower_bound for item in available),
        upper_bound=max(item.upper_bound for item in available),
        confidence=min(item.confidence for item in available),
        unit="1",
        source=THERMAL_CUE_SOURCE,
        timestamp_seconds=timestamp,
        status=combine_estimate_statuses(item.status for item in available),
        evidence=tuple(
            f"candidate source={item.source}; value={item.value}; "
            f"confidence={item.confidence}"
            for item in available
        )
        + (
            "maximum of direct-flame and contextual-plume visual evidence",
            "visual evidence is not a temperature measurement",
        ),
    )


def estimate_thermal_cues(
    frames: tuple[ThermalCueFrame, ...],
    *,
    timestamp_seconds: float,
    config: ThermalCueConfig | None = None,
    open_container_likelihood: ScalarEstimate | None = None,
    plume_source_association_likelihood: ScalarEstimate | None = None,
    carried_likelihood: ScalarEstimate | None = None,
    spill_likelihood: ScalarEstimate | None = None,
) -> ThermalAttributes:
    """Aggregate temporal evidence without binary hot-object classification.

    A direct flame cue can support heat evidence on its own. A steam cue needs
    source context and spatial association before it can support heat evidence.
    Missing detector output stays unavailable rather than becoming zero.
    """

    settings = config or ThermalCueConfig()
    timestamp = float(timestamp_seconds)
    if settings.minimum_frame_count <= 0 or settings.maximum_window_seconds <= 0:
        raise ValueError("thermal temporal settings must be positive")
    if not 0 <= settings.score_uncertainty <= 1:
        raise ValueError("score_uncertainty must lie in [0, 1]")
    if any(frame.timestamp_seconds > timestamp for frame in frames):
        raise ValueError("thermal frame cannot be newer than result")
    recent = tuple(
        frame
        for frame in frames
        if timestamp - frame.timestamp_seconds <= settings.maximum_window_seconds
    )

    def temporal(name: str) -> ScalarEstimate:
        supported = tuple(frame for frame in recent if getattr(frame, name) is not None)
        if len(supported) < settings.minimum_frame_count:
            return _unavailable(timestamp)
        values = tuple(float(getattr(frame, name)) for frame in supported)
        value = median(values)
        confidence = min(1.0, len(supported) / (2 * settings.minimum_frame_count))
        uncertainty = settings.score_uncertainty + (1 - confidence) * 0.2
        return ScalarEstimate(
            value,
            max(0, value - uncertainty),
            min(1, value + uncertainty),
            confidence,
            "1",
            THERMAL_CUE_SOURCE,
            timestamp,
            EstimateStatus.INFERRED,
            (
                f"temporal median over {len(supported)} supported frames",
                f"model_ids={','.join(sorted({frame.model_id for frame in supported}))}",
                "visual cue is not a temperature measurement",
            ),
        )

    flame = temporal("flame_likelihood")
    steam = temporal("steam_likelihood")
    container = open_container_likelihood or _unavailable(
        timestamp, "container_state_unavailable"
    )
    association = plume_source_association_likelihood or _unavailable(
        timestamp, "plume_source_association_unavailable"
    )
    carried = carried_likelihood or _unavailable(
        timestamp, "carried_state_unavailable"
    )
    spill = spill_likelihood or _unavailable(
        timestamp, "spill_specialist_unavailable"
    )
    for name, item in (
        ("open_container", container),
        ("plume_source_association", association),
        ("carried", carried),
        ("spill", spill),
    ):
        if not isinstance(item, ScalarEstimate) or item.unit != "1":
            raise ValueError(f"{name} must be a dimensionless ScalarEstimate")

    contextual = estimate_contextual_thermal_cues(
        plume_likelihood=steam,
        source_context_likelihood=container,
        plume_source_association_likelihood=association,
        timestamp_seconds=timestamp,
        plume_label="steam",
        open_container_likelihood=container,
        carried_likelihood=carried,
        spill_likelihood=spill,
        config=ContextualThermalCueConfig(
            maximum_evidence_age_seconds=settings.maximum_window_seconds
        ),
    )
    contextual_heat = contextual.thermal.heat_source_likelihood
    heat_source = _combine_heat_evidence(
        flame, contextual_heat, timestamp=timestamp
    )

    source_type = CategoricalEstimate.unavailable(
        source=THERMAL_CUE_SOURCE, timestamp_seconds=timestamp
    )
    if heat_source.is_available:
        use_flame = flame.is_available and (
            not contextual_heat.is_available or flame.value >= contextual_heat.value
        )
        source_type = CategoricalEstimate(
            "visible_flame_proxy" if use_flame else "contextual_visible_plume_proxy",
            heat_source.confidence,
            THERMAL_CUE_SOURCE,
            timestamp,
            heat_source.status,
            heat_source.evidence
            + ("descriptive source metadata; not a risk threshold",),
        )
    return ThermalAttributes(
        heat_source_likelihood=heat_source,
        steam_likelihood=steam,
        open_container_likelihood=container,
        carried_likelihood=carried,
        spill_likelihood=spill,
        source_type=source_type,
    )

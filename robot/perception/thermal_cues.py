"""Temporal aggregation for fixed-label flame and steam visual cues."""

from dataclasses import dataclass
import math
from statistics import median

from robot.perception.physical_attributes import (
    CategoricalEstimate,
    EstimateStatus,
    ScalarEstimate,
    ThermalAttributes,
)


THERMAL_CUE_SOURCE = "temporal_fixed_label_thermal_cues"


@dataclass(frozen=True)
class ThermalCueFrame:
    timestamp_seconds: float
    flame_likelihood: float
    steam_likelihood: float
    model_id: str

    def __post_init__(self):
        for name in ("timestamp_seconds", "flame_likelihood", "steam_likelihood"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            if name != "timestamp_seconds" and value > 1:
                raise ValueError(f"{name} must lie in [0, 1]")
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must be nonempty")


@dataclass(frozen=True)
class ThermalCueConfig:
    minimum_frame_count: int = 3
    maximum_window_seconds: float = 1.0
    score_uncertainty: float = 0.15


def estimate_thermal_cues(
    frames: tuple[ThermalCueFrame, ...],
    *,
    timestamp_seconds: float,
    config: ThermalCueConfig | None = None,
    open_container_likelihood: ScalarEstimate | None = None,
    carried_likelihood: ScalarEstimate | None = None,
    spill_likelihood: ScalarEstimate | None = None,
) -> ThermalAttributes:
    """Aggregate temporal evidence without treating smoke as steam/temperature."""
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

    def unavailable(source):
        return ScalarEstimate.unavailable(
            unit="1", source=source, timestamp_seconds=timestamp
        )

    def temporal(name):
        if len(recent) < settings.minimum_frame_count:
            return unavailable(THERMAL_CUE_SOURCE)
        values = tuple(getattr(frame, name) for frame in recent)
        value = median(values)
        confidence = min(1.0, len(recent) / (2 * settings.minimum_frame_count))
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
                f"temporal median over {len(recent)} frames",
                f"model_ids={','.join(sorted({frame.model_id for frame in recent}))}",
                "visual cue is not a temperature measurement",
            ),
        )

    flame = temporal("flame_likelihood")
    steam = temporal("steam_likelihood")
    optional = {
        "open": open_container_likelihood or unavailable("container_state_unavailable"),
        "carried": carried_likelihood or unavailable("carried_state_unavailable"),
        "spill": spill_likelihood or unavailable("spill_specialist_unavailable"),
    }
    for name, item in optional.items():
        if not isinstance(item, ScalarEstimate) or item.unit != "1":
            raise ValueError(f"{name} must be a dimensionless ScalarEstimate")
    source_type = CategoricalEstimate.unavailable(
        source=THERMAL_CUE_SOURCE, timestamp_seconds=timestamp
    )
    if flame.is_available and flame.lower_bound > 0.5:
        source_type = CategoricalEstimate(
            "visible_flame",
            flame.confidence,
            THERMAL_CUE_SOURCE,
            timestamp,
            EstimateStatus.INFERRED,
            flame.evidence,
        )
    elif steam.is_available and steam.lower_bound > 0.5:
        source_type = CategoricalEstimate(
            "visible_steam",
            steam.confidence,
            THERMAL_CUE_SOURCE,
            timestamp,
            EstimateStatus.INFERRED,
            steam.evidence,
        )
    return ThermalAttributes(
        heat_source_likelihood=flame,
        steam_likelihood=steam,
        open_container_likelihood=optional["open"],
        carried_likelihood=optional["carried"],
        spill_likelihood=optional["spill"],
        source_type=source_type,
    )

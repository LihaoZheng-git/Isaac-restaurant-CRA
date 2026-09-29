"""Compatibility wrapper for the general contextual thermal-cue contract.

Hot soup is a restaurant evaluation scenario. Runtime perception estimates
continuous visual heat-source evidence and never classifies soup or hotness.
New code should use :mod:`robot.perception.contextual_thermal_cues` directly.
"""

from __future__ import annotations

from robot.perception.contextual_thermal_cues import (
    CONTEXTUAL_THERMAL_SOURCE,
    ContextualThermalCueConfig,
    ContextualThermalCueResult,
    estimate_contextual_thermal_cues,
)
from robot.perception.physical_attributes import ScalarEstimate


HOT_SOUP_SOURCE = CONTEXTUAL_THERMAL_SOURCE
HotSoupCueConfig = ContextualThermalCueConfig
HotSoupCueResult = ContextualThermalCueResult


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
    """Adapt legacy restaurant names to general continuous thermal evidence."""

    return estimate_contextual_thermal_cues(
        plume_likelihood=plume_likelihood,
        source_context_likelihood=open_container_likelihood,
        plume_source_association_likelihood=plume_above_container_likelihood,
        timestamp_seconds=timestamp_seconds,
        plume_label=plume_label,
        open_container_likelihood=open_container_likelihood,
        carried_likelihood=carried_likelihood,
        spill_likelihood=spill_likelihood,
        config=config,
    )

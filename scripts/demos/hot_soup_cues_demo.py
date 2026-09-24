"""Stationary synthetic hot-soup cue and CRA thermal-factor diagnostic."""

from dataclasses import replace
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot.perception.hot_soup_cues import estimate_hot_soup_cues
from robot.perception.physical_attributes import (
    EstimateStatus,
    PhysicalAttributes,
    ScalarEstimate,
)
from robot.risk.physical_factor_adapter import adapt_physical_factors


def cue(value: float, name: str) -> ScalarEstimate:
    return ScalarEstimate(
        value=value,
        lower_bound=max(0.0, value - 0.1),
        upper_bound=min(1.0, value + 0.1),
        confidence=0.8,
        unit="1",
        source=f"synthetic_{name}_fixture",
        timestamp_seconds=1.0,
        status=EstimateStatus.PRIOR,
        evidence=("synthetic diagnostic input; not sensor-derived",),
    )


def main() -> None:
    result = estimate_hot_soup_cues(
        plume_likelihood=cue(0.9, "visible_plume"),
        open_container_likelihood=cue(0.85, "open_container"),
        plume_above_container_likelihood=cue(0.8, "spatial_relation"),
        timestamp_seconds=1.1,
        plume_label="smoke_like",
    )
    unknown = PhysicalAttributes.unknown(track_id=1, timestamp_seconds=1.1)
    attributes = replace(unknown, thermal=result.thermal)
    no_interaction = adapt_physical_factors(attributes, timestamp_seconds=1.1)
    with_interaction = adapt_physical_factors(
        attributes,
        timestamp_seconds=1.1,
        thermal_interaction_likelihood=cue(0.6, "route_interaction"),
    )

    print("STATIONARY HOT-SOUP DIAGNOSTIC: synthetic priors, no camera or motion")
    print(
        "heat_proxy="
        f"{result.thermal.heat_source_likelihood.value:.2f}; "
        f"source_type={result.thermal.source_type.value}"
    )
    print("plume_label=smoke_like; steam=unavailable; temperature=unavailable")
    print(
        "CRA thermal without route interaction: "
        f"confidence={no_interaction.thermal.estimate.confidence:.2f} (unknown)"
    )
    print(
        "CRA thermal with synthetic route interaction: "
        f"severity={with_interaction.thermal.estimate.severity:.2f}, "
        f"interaction={with_interaction.thermal.estimate.interaction_likelihood:.2f}"
    )
    assert not result.thermal.steam_likelihood.is_available
    assert no_interaction.thermal.estimate.confidence == 0.0
    assert with_interaction.thermal.estimate.confidence > 0.0
    print("PASS: context was required and temperature stayed unavailable")


if __name__ == "__main__":
    main()

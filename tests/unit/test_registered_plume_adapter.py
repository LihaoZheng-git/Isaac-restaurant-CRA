from dataclasses import replace

import pytest

from robot.perception.physical_attributes import (
    PhysicalAttributes,
    ScalarEstimate,
    EstimateStatus,
)
from robot.risk.physical_factor_adapter import (
    adapt_physical_factors,
    PhysicalFactorAdapterConfig,
)


def test_registered_plume_keeps_proxy_basis_and_original_continuous_uncertainty():
    base = PhysicalAttributes.unknown(track_id=1, timestamp_seconds=1.0)
    cue = ScalarEstimate(
        0.4,
        0.3,
        0.5,
        0.5,
        "1",
        "registered_rgb_object_plume_proxy",
        1.0,
        EstimateStatus.INFERRED,
    )
    route = ScalarEstimate(
        0.3, 0.2, 0.4, 0.6, "1", "tracked_route", 1.0, EstimateStatus.INFERRED
    )
    attributes = replace(
        base, thermal=replace(base.thermal, heat_source_likelihood=cue)
    )
    result = adapt_physical_factors(
        attributes,
        timestamp_seconds=1.0,
        thermal_interaction_likelihood=route,
        thermal_basis="visual_object_plume_proxy",
        config=PhysicalFactorAdapterConfig(allow_visual_thermal_proxy=True),
    )
    assert result.thermal.basis == "visual_object_plume_proxy"
    assert result.thermal.estimate.severity == 0.5
    assert result.thermal.estimate.interaction_likelihood == 0.4
    assert result.thermal.inputs[0][1] is cue
    with pytest.raises(ValueError, match="opt-in"):
        adapt_physical_factors(
            attributes, timestamp_seconds=1.0, thermal_basis="visual_object_plume_proxy"
        )


def test_missing_registered_plume_is_unknown_not_a_negative_observation():
    base = PhysicalAttributes.unknown(track_id=1, timestamp_seconds=1.0)
    cue = ScalarEstimate.unavailable(
        unit="1", source="registered_rgb_object_plume_proxy", timestamp_seconds=1.0
    )
    result = adapt_physical_factors(
        replace(base, thermal=replace(base.thermal, heat_source_likelihood=cue)),
        timestamp_seconds=1.0,
        thermal_basis="visual_object_plume_proxy",
        config=PhysicalFactorAdapterConfig(allow_visual_thermal_proxy=True),
    )
    assert result.thermal.rejection_reasons
    assert result.thermal.estimate.confidence == 0

"""Expose adapted sensor-derived physical profiles to the existing CRA runtime."""

from types import MappingProxyType
from typing import Mapping

from robot.perception.physical_attributes import PhysicalAttributes, ScalarEstimate
from robot.risk.cra import PhysicalRiskProfile
from robot.risk.physical_factor_adapter import (
    PhysicalFactorAdapterConfig,
    PhysicalFactorAdapterResult,
    adapt_physical_factors,
)
from robot.safety.semantic_obstacle import SemanticObstacle


class PhysicalAttributeCRARiskProfileProvider:
    """Immutable per-frame provider keyed by existing LiDAR track IDs."""

    def __init__(
        self,
        attributes_by_track_id: Mapping[int, PhysicalAttributes],
        *,
        timestamp_seconds: float,
        potential_interactions: Mapping[int, ScalarEstimate] | None = None,
        thermal_interactions: Mapping[int, ScalarEstimate] | None = None,
        config: PhysicalFactorAdapterConfig | None = None,
    ):
        potential = potential_interactions or {}
        thermal = thermal_interactions or {}
        results = {}
        for track_id, attributes in attributes_by_track_id.items():
            if track_id != attributes.track_id:
                raise ValueError("attribute mapping key must match track_id")
            results[track_id] = adapt_physical_factors(
                attributes,
                timestamp_seconds=timestamp_seconds,
                potential_interaction_likelihood=potential.get(track_id),
                thermal_interaction_likelihood=thermal.get(track_id),
                config=config,
            )
        self._results = MappingProxyType(results)

    @property
    def diagnostics_by_track_id(self) -> Mapping[int, PhysicalFactorAdapterResult]:
        return self._results

    def profile_for(self, obstacle: SemanticObstacle) -> PhysicalRiskProfile | None:
        if not isinstance(obstacle, SemanticObstacle):
            raise TypeError("obstacle must be SemanticObstacle")
        result = self._results.get(obstacle.track_id)
        return None if result is None else result.profile

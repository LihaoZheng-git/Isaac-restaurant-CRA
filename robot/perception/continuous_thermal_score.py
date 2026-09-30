"""Interpretable continuous thermal evidence from visual plume features.

The score is a visual heat-source proxy, not temperature, hot/not-hot
classification, or burn severity.  Mask geometry and persistence determine
plume strength.  Segmentation confidence affects only epistemic confidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

from robot.perception.physical_attributes import (
    CategoricalEstimate,
    ScalarEstimate,
    ThermalAttributes,
    combine_estimate_statuses,
)
from robot.perception.plume_features import ContinuousPlumeFeatures


CONTINUOUS_THERMAL_SOURCE = "continuous_visual_plume_thermal_evidence"


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class ContinuousThermalScoreConfig:
    """Interpretable monotone mapping from plume descriptors to evidence."""

    relative_area_weight: float = 0.40
    vertical_extent_weight: float = 0.25
    persistence_weight: float = 0.35
    full_strength_relative_area: float = 1.0
    full_strength_vertical_extent: float = 2.0
    maximum_evidence_age_seconds: float = 0.5

    def __post_init__(self) -> None:
        weights = tuple(
            _finite(getattr(self, name), name)
            for name in (
                "relative_area_weight",
                "vertical_extent_weight",
                "persistence_weight",
            )
        )
        if any(value < 0 for value in weights) or not math.isclose(
            sum(weights), 1.0, abs_tol=1e-9
        ):
            raise ValueError("thermal feature weights must be nonnegative and sum to 1")
        for name in (
            "full_strength_relative_area",
            "full_strength_vertical_extent",
            "maximum_evidence_age_seconds",
        ):
            value = _finite(getattr(self, name), name)
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "relative_area_weight", weights[0])
        object.__setattr__(self, "vertical_extent_weight", weights[1])
        object.__setattr__(self, "persistence_weight", weights[2])


@dataclass(frozen=True)
class ContinuousThermalScoreResult:
    """Continuous plume strength and contextual heat-source evidence."""

    plume_strength: ScalarEstimate
    thermal: ThermalAttributes
    rejection_reasons: tuple[str, ...]


def _unavailable(timestamp: float, *evidence: str) -> ScalarEstimate:
    return ScalarEstimate.unavailable(
        unit="1",
        source=CONTINUOUS_THERMAL_SOURCE,
        timestamp_seconds=timestamp,
        evidence=evidence,
    )


def _normalized(value: float, scale: float) -> float:
    return min(1.0, max(0.0, value / scale))


def estimate_continuous_thermal_score(
    features: ContinuousPlumeFeatures,
    *,
    timestamp_seconds: float,
    config: ContinuousThermalScoreConfig | None = None,
) -> ContinuousThermalScoreResult:
    """Map visual plume descriptors to continuous contextual evidence.

    Plume strength is a weighted monotone mapping.  Source context and spatial
    association remain conservative fuzzy conjunctions through ``min``.  Raw
    model confidence caps result confidence but never changes score magnitude.
    """

    if not isinstance(features, ContinuousPlumeFeatures):
        raise TypeError("features must be ContinuousPlumeFeatures")
    timestamp = _finite(timestamp_seconds, "timestamp_seconds")
    if timestamp < 0:
        raise ValueError("timestamp_seconds cannot be negative")
    settings = config or ContinuousThermalScoreConfig()
    if not isinstance(settings, ContinuousThermalScoreConfig):
        raise TypeError("config must be a ContinuousThermalScoreConfig")

    required = (
        ("relative_mask_area", features.relative_mask_area),
        ("vertical_extent", features.vertical_extent),
        ("persistence", features.persistence),
        ("source_context_likelihood", features.source_context_likelihood),
        (
            "spatial_association_likelihood",
            features.spatial_association_likelihood,
        ),
        ("segmentation_confidence", features.segmentation_confidence),
    )
    reasons = []
    for name, item in required:
        if not item.is_available:
            reasons.append(f"{name}: unavailable")
        elif item.timestamp_seconds > timestamp:
            reasons.append(f"{name}: future evidence")
        elif (
            timestamp - item.timestamp_seconds
            > settings.maximum_evidence_age_seconds
        ):
            reasons.append(f"{name}: stale evidence")
    provenance = tuple(
        f"{name}: status={item.status.value}, source={item.source}, "
        f"confidence={item.confidence}"
        for name, item in required
    )
    limits = (
        "visual plume evidence is not a temperature measurement",
        "score is not hot/not-hot classification or burn severity",
        "segmentation confidence affects confidence only, never score magnitude",
    )
    if reasons:
        unavailable = _unavailable(timestamp, *(provenance + tuple(reasons) + limits))
        thermal = ThermalAttributes(
            heat_source_likelihood=unavailable,
            steam_likelihood=_unavailable(
                timestamp,
                "plume feature score does not prove steam",
            ),
            open_container_likelihood=_unavailable(
                timestamp,
                "source context is generic and is not relabeled as open-container state",
            ),
            carried_likelihood=_unavailable(timestamp, "carried state unavailable"),
            spill_likelihood=_unavailable(timestamp, "spill state unavailable"),
            source_type=CategoricalEstimate.unavailable(
                source=CONTINUOUS_THERMAL_SOURCE,
                timestamp_seconds=timestamp,
                evidence=provenance + tuple(reasons) + limits,
            ),
        )
        return ContinuousThermalScoreResult(unavailable, thermal, tuple(reasons))

    area = features.relative_mask_area
    extent = features.vertical_extent
    persistence = features.persistence
    context = features.source_context_likelihood
    association = features.spatial_association_likelihood
    model_confidence = features.segmentation_confidence
    assert all(
        item.value is not None and item.lower_bound is not None and item.upper_bound is not None
        for _, item in required
    )

    def plume_mapping(bound: str) -> float:
        return min(
            1.0,
            settings.relative_area_weight
            * _normalized(getattr(area, bound), settings.full_strength_relative_area)
            + settings.vertical_extent_weight
            * _normalized(getattr(extent, bound), settings.full_strength_vertical_extent)
            + settings.persistence_weight * getattr(persistence, bound),
        )

    plume_value = plume_mapping("value")
    plume_lower = plume_mapping("lower_bound")
    plume_upper = plume_mapping("upper_bound")
    status = combine_estimate_statuses(item.status for _, item in required)
    epistemic_confidence = min(
        *(item.confidence for _, item in required),
        model_confidence.value,
    )
    mapping_evidence = provenance + limits + (
        "plume strength is a configured weighted sum of normalized area, "
        "vertical extent, and persistence",
        f"weights=({settings.relative_area_weight}, "
        f"{settings.vertical_extent_weight}, {settings.persistence_weight})",
        f"full_strength_scales=({settings.full_strength_relative_area}, "
        f"{settings.full_strength_vertical_extent})",
    )
    plume_strength = ScalarEstimate(
        value=plume_value,
        lower_bound=plume_lower,
        upper_bound=plume_upper,
        confidence=epistemic_confidence,
        unit="1",
        source=CONTINUOUS_THERMAL_SOURCE,
        timestamp_seconds=timestamp,
        status=status,
        evidence=mapping_evidence,
    )
    heat = ScalarEstimate(
        value=min(plume_value, context.value, association.value),
        lower_bound=min(plume_lower, context.lower_bound, association.lower_bound),
        upper_bound=min(plume_upper, context.upper_bound, association.upper_bound),
        confidence=epistemic_confidence,
        unit="1",
        source=CONTINUOUS_THERMAL_SOURCE,
        timestamp_seconds=timestamp,
        status=status,
        evidence=mapping_evidence
        + ("heat-source evidence is min(plume strength, source context, association)",),
    )
    thermal = ThermalAttributes(
        heat_source_likelihood=heat,
        steam_likelihood=_unavailable(
            timestamp,
            "generic plume features do not prove steam",
        ),
        open_container_likelihood=_unavailable(
            timestamp,
            "generic source context does not prove an open container",
        ),
        carried_likelihood=_unavailable(timestamp, "carried state unavailable"),
        spill_likelihood=_unavailable(timestamp, "spill state unavailable"),
        source_type=CategoricalEstimate(
            value="continuous_visual_plume_proxy",
            confidence=epistemic_confidence,
            source=CONTINUOUS_THERMAL_SOURCE,
            timestamp_seconds=timestamp,
            status=status,
            evidence=heat.evidence + ("descriptive evidence type, not a threshold",),
        ),
    )
    return ContinuousThermalScoreResult(plume_strength, thermal, ())

"""Physical evidence to CRA; energy scales are uncalibrated hypotheses.

Potential energy uses explicit fall height, never object/centroid height.
The caller supplies harmful-fall interaction opportunity (including support
and accessibility where justified); instability alone is not that opportunity.
Kinetic energy uses the supplied relative-speed reference, which the current
motion estimator computes against waypoint-aligned nominal robot motion.
Neither energy nor route crossing is a collision probability or safety proof.
Contact and thermal remain unknown even if attribute placeholders are filled.
"""

from dataclasses import dataclass, fields
import math
from numbers import Real

from robot.perception.physical_attributes import (
    EstimateStatus,
    PhysicalAttributes,
    ScalarEstimate,
)
from robot.risk.cra import PhysicalRiskProfile, RiskFactorEstimate


ADAPTER_SOURCE = "physical_evidence_cra_adapter"


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class PhysicalFactorAdapterConfig:
    """Linear clipped severity = energy / scale; upper bounds by default.

    Confidence is the weakest accepted input, capped for priors/fallbacks.
    CRA applies its own uncertainty uplift afterwards. Interval bounds are
    engineering evidence bounds, not statistical confidence intervals.
    """

    potential_energy_scale_j: float = 100.0
    kinetic_energy_scale_j: float = 100.0
    gravity_mps2: float = 9.81
    maximum_evidence_age_seconds: float = 1.0
    prior_confidence_cap: float = 0.5
    fallback_confidence_cap: float = 0.0
    use_upper_bounds: bool = True
    allow_priors: bool = True

    def __post_init__(self) -> None:
        for name in (
            "potential_energy_scale_j",
            "kinetic_energy_scale_j",
            "gravity_mps2",
            "maximum_evidence_age_seconds",
            "prior_confidence_cap",
            "fallback_confidence_cap",
        ):
            value = _number(getattr(self, name), name)
            if name.endswith("cap"):
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"{name} must lie in [0, 1]")
            elif value <= 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        for name in ("use_upper_bounds", "allow_priors"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")


@dataclass(frozen=True)
class PhysicalFactorDiagnostic:
    """Immutable inputs retain exact statuses, sources, bounds and timestamps."""

    factor: str
    inputs: tuple[tuple[str, ScalarEstimate], ...]
    energy_j: ScalarEstimate
    estimate: RiskFactorEstimate
    rejection_reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        def scalar(item: ScalarEstimate) -> dict[str, object]:
            return {
                field.name: (
                    item.status.value
                    if field.name == "status"
                    else getattr(item, field.name)
                )
                for field in fields(item)
            }

        return {
            "factor": self.factor,
            "inputs": {name: scalar(item) for name, item in self.inputs},
            "energy_j": scalar(self.energy_j),
            "severity": self.estimate.severity,
            "interaction_likelihood": self.estimate.interaction_likelihood,
            "confidence": self.estimate.confidence,
            "rejection_reasons": self.rejection_reasons,
        }


@dataclass(frozen=True)
class PhysicalFactorAdapterResult:
    track_id: int
    timestamp_seconds: float
    config: PhysicalFactorAdapterConfig
    profile: PhysicalRiskProfile
    potential: PhysicalFactorDiagnostic
    kinetic: PhysicalFactorDiagnostic


def _factor(
    factor: str,
    inputs: tuple[tuple[str, ScalarEstimate], ...],
    *,
    timestamp: float,
    config: PhysicalFactorAdapterConfig,
) -> PhysicalFactorDiagnostic:
    reasons = []
    for name, item in inputs:
        if item.timestamp_seconds > timestamp:
            reasons.append(f"{name}: future evidence")
        elif timestamp - item.timestamp_seconds > config.maximum_evidence_age_seconds:
            reasons.append(f"{name}: stale evidence")
        if not item.is_available:
            reasons.append(f"{name}: unavailable")
        if item.status is EstimateStatus.PRIOR and not config.allow_priors:
            reasons.append(f"{name}: priors disabled")
    provenance = tuple(
        f"{name}: status={item.status.value}, source={item.source}, "
        f"timestamp={item.timestamp_seconds}, confidence={item.confidence}"
        for name, item in inputs
    )
    energy = ScalarEstimate.unavailable(
        unit="J",
        source=ADAPTER_SOURCE,
        timestamp_seconds=timestamp,
        evidence=provenance,
    )
    if not reasons:
        mass, quantity, interaction = (item for _, item in inputs)
        energies = []
        for bound in ("value", "lower_bound", "upper_bound"):
            m = getattr(mass, bound)
            q = getattr(quantity, bound)
            assert m is not None and q is not None
            value = (
                m * config.gravity_mps2 * q
                if factor == "potential"
                else 0.5 * m * q * q
            )
            energies.append(value)
        if not all(math.isfinite(value) for value in energies):
            reasons.append("energy: non-finite arithmetic")
        else:

            def confidence(items: tuple[ScalarEstimate, ...]) -> float:
                return min(
                    min(item.confidence, config.prior_confidence_cap)
                    if item.status is EstimateStatus.PRIOR
                    else min(item.confidence, config.fallback_confidence_cap)
                    if item.status is EstimateStatus.FALLBACK
                    else item.confidence
                    for item in items
                )

            energy = ScalarEstimate(
                value=energies[0],
                lower_bound=energies[1],
                upper_bound=energies[2],
                confidence=confidence((mass, quantity)),
                unit="J",
                source=ADAPTER_SOURCE,
                timestamp_seconds=timestamp,
                status=EstimateStatus.INFERRED,
                evidence=provenance,
            )
            bound = "upper_bound" if config.use_upper_bounds else "value"
            scale = (
                config.potential_energy_scale_j
                if factor == "potential"
                else config.kinetic_energy_scale_j
            )
            estimate = RiskFactorEstimate(
                severity=min(1.0, getattr(energy, bound) / scale),
                interaction_likelihood=getattr(interaction, bound),
                confidence=confidence((mass, quantity, interaction)),
                source=ADAPTER_SOURCE,
                evidence=provenance
                + (
                    f"linear clipped energy severity; scale={scale} J; bound={bound}",
                    "uncalibrated consequence and interaction proxies",
                ),
            )
            return PhysicalFactorDiagnostic(factor, inputs, energy, estimate, ())
    estimate = RiskFactorEstimate.unknown(
        source=ADAPTER_SOURCE,
        evidence=provenance + tuple(reasons),
    )
    return PhysicalFactorDiagnostic(factor, inputs, energy, estimate, tuple(reasons))


def adapt_physical_factors(
    attributes: PhysicalAttributes,
    *,
    timestamp_seconds: float,
    potential_interaction_likelihood: ScalarEstimate | None = None,
    config: PhysicalFactorAdapterConfig | None = None,
) -> PhysicalFactorAdapterResult:
    """Adapt one track without semantic-class or object-height substitutions.

    Missing fall interaction evidence leaves potential unknown. No automatic
    instability, distance, camera exposure, or route-score substitution is made.
    Future/stale contributing evidence leaves the corresponding factor unknown.
    """
    if not isinstance(attributes, PhysicalAttributes):
        raise TypeError("attributes must be PhysicalAttributes")
    timestamp = _number(timestamp_seconds, "timestamp_seconds")
    if timestamp < 0.0:
        raise ValueError("timestamp_seconds cannot be negative")
    settings = PhysicalFactorAdapterConfig() if config is None else config
    if not isinstance(settings, PhysicalFactorAdapterConfig):
        raise TypeError("config must be PhysicalFactorAdapterConfig")
    interaction = potential_interaction_likelihood
    if interaction is None:
        interaction = ScalarEstimate.unavailable(
            unit="1",
            source="harmful_fall_interaction_unavailable",
            timestamp_seconds=timestamp,
        )
    if not isinstance(interaction, ScalarEstimate):
        raise TypeError("potential interaction must be ScalarEstimate")
    if interaction.unit != "1":
        raise ValueError("potential interaction must use unit '1'")
    if interaction.is_available and not (
        0.0 <= interaction.lower_bound <= interaction.upper_bound <= 1.0
    ):
        raise ValueError("potential interaction bounds must lie in [0, 1]")
    potential = _factor(
        "potential",
        (
            ("mass_kg", attributes.mass.mass_kg),
            ("fall_height_m", attributes.geometry.fall_height_m),
            ("harmful_fall_interaction", interaction),
        ),
        timestamp=timestamp,
        config=settings,
    )
    kinetic = _factor(
        "kinetic",
        (
            ("mass_kg", attributes.mass.mass_kg),
            ("relative_speed_mps", attributes.motion.relative_speed_mps),
            ("route_crossing_likelihood", attributes.motion.route_crossing_likelihood),
        ),
        timestamp=timestamp,
        config=settings,
    )
    unsupported = RiskFactorEstimate.unknown(
        source="physical_adapter_unsupported",
        evidence=("contact/thermal specialist adaptation is not implemented",),
    )
    return PhysicalFactorAdapterResult(
        attributes.track_id,
        timestamp,
        settings,
        PhysicalRiskProfile(
            unsupported, unsupported, potential.estimate, kinetic.estimate
        ),
        potential,
        kinetic,
    )

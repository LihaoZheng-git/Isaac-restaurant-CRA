"""Dependency-free mathematical core for contextual risk assessment (CRA).

CRA represents four physical hazard families: thermal, contact, potential,
and kinetic.  Each factor separates consequence severity from contextual
interaction likelihood and records the confidence of that estimate.  Exact
robot--obstacle distance is deliberately excluded; geometry remains the
responsibility of the safety function and controller.

The assessor first constructs a conservative factor risk

``r_i = severity_i * interaction_likelihood_i``

and then moves low-confidence estimates toward a configured fallback without
ever reducing their nominal risk.  A replaceable monotone aggregation produces
the normalized danger index ``phi``.  This module has no Isaac Sim, perception,
or Safe-SAGE dependencies so the mathematical assumptions can be tested and
revised independently.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from numbers import Real
from typing import Iterable


CRA_SCHEMA_VERSION = 1


def _unit_interval(value: object, *, name: str) -> float:
    """Return one finite value in ``[0, 1]`` while rejecting booleans."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1].")
    return result


def _nonempty_text(value: object, *, name: str) -> str:
    """Return stripped nonempty text."""

    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    result = value.strip()
    if not result:
        raise ValueError(f"{name} cannot be empty.")
    return result


def _evidence_items(value: object) -> tuple[str, ...]:
    """Return immutable, nonempty evidence statements."""

    if isinstance(value, str):
        raise TypeError("evidence must be an iterable of strings, not a string.")
    try:
        items = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError("evidence must be an iterable of strings.") from error
    return tuple(_nonempty_text(item, name="evidence item") for item in items)


class PhysicalRiskFactor(str, Enum):
    """Physical hazard families used by CRA."""

    THERMAL = "thermal"
    CONTACT = "contact"
    POTENTIAL = "potential"
    KINETIC = "kinetic"


class CRAAggregation(str, Enum):
    """Supported monotone mappings from factor risks to ``phi``."""

    NOISY_OR = "noisy_or"
    WEIGHTED_SUM = "weighted_sum"
    MAXIMUM = "maximum"


@dataclass(frozen=True)
class RiskFactorEstimate:
    """One explainable physical-factor estimate.

    ``interaction_likelihood`` describes contextual accessibility or the
    opportunity for harmful interaction.  It is not camera exposure and must
    not simply duplicate instantaneous obstacle distance.
    """

    severity: float
    interaction_likelihood: float
    confidence: float
    source: str = "manual"
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "severity",
            _unit_interval(self.severity, name="severity"),
        )
        object.__setattr__(
            self,
            "interaction_likelihood",
            _unit_interval(
                self.interaction_likelihood,
                name="interaction_likelihood",
            ),
        )
        object.__setattr__(
            self,
            "confidence",
            _unit_interval(self.confidence, name="confidence"),
        )
        object.__setattr__(
            self,
            "source",
            _nonempty_text(self.source, name="source"),
        )
        object.__setattr__(self, "evidence", _evidence_items(self.evidence))

    @property
    def nominal_risk(self) -> float:
        """Return ``severity * interaction_likelihood``."""

        return self.severity * self.interaction_likelihood

    @classmethod
    def known_absent(
        cls,
        *,
        source: str = "manual",
        evidence: Iterable[str] = (),
    ) -> "RiskFactorEstimate":
        """Return a high-confidence observation that this hazard is absent."""

        return cls(
            severity=0.0,
            interaction_likelihood=0.0,
            confidence=1.0,
            source=source,
            evidence=tuple(evidence),
        )

    @classmethod
    def unknown(
        cls,
        *,
        source: str = "unavailable",
        evidence: Iterable[str] = (),
    ) -> "RiskFactorEstimate":
        """Return an unknown factor that will resolve toward its fallback."""

        return cls(
            severity=0.0,
            interaction_likelihood=0.0,
            confidence=0.0,
            source=source,
            evidence=tuple(evidence),
        )


@dataclass(frozen=True)
class PhysicalRiskProfile:
    """The four factor estimates for one contextual subject."""

    thermal: RiskFactorEstimate
    contact: RiskFactorEstimate
    potential: RiskFactorEstimate
    kinetic: RiskFactorEstimate

    def __post_init__(self) -> None:
        for factor, estimate in self.items():
            if not isinstance(estimate, RiskFactorEstimate):
                raise TypeError(f"{factor.value} must be a RiskFactorEstimate.")

    def estimate_for(self, factor: PhysicalRiskFactor) -> RiskFactorEstimate:
        """Return the estimate associated with one physical factor."""

        resolved = _physical_factor(factor)
        return getattr(self, resolved.value)

    def items(
        self,
    ) -> tuple[tuple[PhysicalRiskFactor, RiskFactorEstimate], ...]:
        """Return estimates in stable scientific-report order."""

        return tuple(
            (factor, getattr(self, factor.value)) for factor in PhysicalRiskFactor
        )

    @classmethod
    def known_safe(cls, *, source: str = "manual") -> "PhysicalRiskProfile":
        """Return a profile whose four hazards are confidently absent."""

        return cls(
            **{
                factor.value: RiskFactorEstimate.known_absent(source=source)
                for factor in PhysicalRiskFactor
            }
        )

    @classmethod
    def unknown(cls, *, source: str = "unavailable") -> "PhysicalRiskProfile":
        """Return a profile whose four hazards are unknown."""

        return cls(
            **{
                factor.value: RiskFactorEstimate.unknown(source=source)
                for factor in PhysicalRiskFactor
            }
        )


@dataclass(frozen=True)
class PhysicalRiskVector:
    """One validated scalar value for each physical factor."""

    thermal: float
    contact: float
    potential: float
    kinetic: float

    def __post_init__(self) -> None:
        for factor in PhysicalRiskFactor:
            object.__setattr__(
                self,
                factor.value,
                _unit_interval(
                    getattr(self, factor.value),
                    name=f"{factor.value} value",
                ),
            )

    def value_for(self, factor: PhysicalRiskFactor) -> float:
        """Return the scalar associated with one physical factor."""

        resolved = _physical_factor(factor)
        return float(getattr(self, resolved.value))

    def items(self) -> tuple[tuple[PhysicalRiskFactor, float], ...]:
        """Return values in stable scientific-report order."""

        return tuple((factor, self.value_for(factor)) for factor in PhysicalRiskFactor)

    def as_dict(self) -> dict[str, float]:
        """Return a JSON-compatible factor mapping."""

        return {factor.value: value for factor, value in self.items()}

    @classmethod
    def zeros(cls) -> "PhysicalRiskVector":
        """Return four zero values."""

        return cls(thermal=0.0, contact=0.0, potential=0.0, kinetic=0.0)

    @classmethod
    def ones(cls) -> "PhysicalRiskVector":
        """Return four unit values."""

        return cls(thermal=1.0, contact=1.0, potential=1.0, kinetic=1.0)


@dataclass(frozen=True)
class CRAAssessmentConfig:
    """Aggregation, weighting, and fail-closed uncertainty assumptions."""

    aggregation: CRAAggregation = CRAAggregation.NOISY_OR
    weights: PhysicalRiskVector = field(default_factory=PhysicalRiskVector.ones)
    fallback_risks: PhysicalRiskVector = field(default_factory=PhysicalRiskVector.ones)

    def __post_init__(self) -> None:
        try:
            aggregation = CRAAggregation(self.aggregation)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "aggregation must be a supported CRAAggregation."
            ) from error
        if not isinstance(self.weights, PhysicalRiskVector):
            raise TypeError("weights must be a PhysicalRiskVector.")
        if not isinstance(self.fallback_risks, PhysicalRiskVector):
            raise TypeError("fallback_risks must be a PhysicalRiskVector.")
        if not any(value > 0.0 for _, value in self.weights.items()):
            raise ValueError("At least one physical-factor weight must be positive.")
        object.__setattr__(self, "aggregation", aggregation)


@dataclass(frozen=True)
class CRAFactorAssessment:
    """Nominal and uncertainty-adjusted diagnostics for one factor."""

    factor: PhysicalRiskFactor
    estimate: RiskFactorEstimate
    fallback_risk: float
    weight: float
    nominal_risk: float
    conservative_risk: float
    weighted_conservative_risk: float
    uncertainty_uplift: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "factor", _physical_factor(self.factor))
        if not isinstance(self.estimate, RiskFactorEstimate):
            raise TypeError("estimate must be a RiskFactorEstimate.")
        for name in (
            "fallback_risk",
            "weight",
            "nominal_risk",
            "conservative_risk",
            "weighted_conservative_risk",
            "uncertainty_uplift",
        ):
            object.__setattr__(
                self,
                name,
                _unit_interval(getattr(self, name), name=name),
            )
        if self.conservative_risk + 1e-12 < self.nominal_risk:
            raise ValueError("conservative_risk cannot be below nominal_risk.")

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible diagnostic mapping."""

        return {
            "severity": self.estimate.severity,
            "interaction_likelihood": self.estimate.interaction_likelihood,
            "confidence": self.estimate.confidence,
            "source": self.estimate.source,
            "evidence": list(self.estimate.evidence),
            "nominal_risk": self.nominal_risk,
            "fallback_risk": self.fallback_risk,
            "conservative_risk": self.conservative_risk,
            "weight": self.weight,
            "weighted_conservative_risk": self.weighted_conservative_risk,
            "uncertainty_uplift": self.uncertainty_uplift,
        }


@dataclass(frozen=True)
class CRAAssessment:
    """Immutable, explainable output of one CRA evaluation."""

    aggregation: CRAAggregation
    factors: tuple[CRAFactorAssessment, ...]
    nominal_phi: float
    phi: float
    overall_confidence: float
    uncertainty_uplift: float
    dominant_factor: PhysicalRiskFactor | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "aggregation", CRAAggregation(self.aggregation))
        factors = tuple(self.factors)
        if len(factors) != len(PhysicalRiskFactor):
            raise ValueError("factors must contain all four physical factors.")
        if any(not isinstance(item, CRAFactorAssessment) for item in factors):
            raise TypeError("Every factors item must be a CRAFactorAssessment.")
        factor_names = tuple(item.factor for item in factors)
        if factor_names != tuple(PhysicalRiskFactor):
            raise ValueError("factors must use stable physical-factor order.")
        object.__setattr__(self, "factors", factors)
        for name in (
            "nominal_phi",
            "phi",
            "overall_confidence",
            "uncertainty_uplift",
        ):
            object.__setattr__(
                self,
                name,
                _unit_interval(getattr(self, name), name=name),
            )
        if self.phi + 1e-12 < self.nominal_phi:
            raise ValueError("phi cannot be below nominal_phi.")
        if self.dominant_factor is not None:
            object.__setattr__(
                self,
                "dominant_factor",
                _physical_factor(self.dominant_factor),
            )

    def factor(self, factor: PhysicalRiskFactor) -> CRAFactorAssessment:
        """Return diagnostics for one physical factor."""

        resolved = _physical_factor(factor)
        return self.factors[list(PhysicalRiskFactor).index(resolved)]

    def as_dict(self) -> dict[str, object]:
        """Return a versioned JSON-compatible assessment."""

        return {
            "schema_version": CRA_SCHEMA_VERSION,
            "aggregation": self.aggregation.value,
            "nominal_phi": self.nominal_phi,
            "phi": self.phi,
            "overall_confidence": self.overall_confidence,
            "uncertainty_uplift": self.uncertainty_uplift,
            "dominant_factor": (
                None if self.dominant_factor is None else self.dominant_factor.value
            ),
            "factors": {item.factor.value: item.as_dict() for item in self.factors},
        }


class CRARiskAssessor:
    """Evaluate physical-factor profiles under one explicit CRA policy."""

    def __init__(self, config: CRAAssessmentConfig | None = None) -> None:
        if config is not None and not isinstance(config, CRAAssessmentConfig):
            raise TypeError("config must be a CRAAssessmentConfig.")
        self.config = config or CRAAssessmentConfig()

    def assess(self, profile: PhysicalRiskProfile) -> CRAAssessment:
        """Return one conservative, explainable CRA danger assessment."""

        if not isinstance(profile, PhysicalRiskProfile):
            raise TypeError("profile must be a PhysicalRiskProfile.")
        factors = tuple(
            self._assess_factor(factor, estimate)
            for factor, estimate in profile.items()
        )
        nominal_phi = _aggregate(
            tuple(item.nominal_risk for item in factors),
            weights=tuple(item.weight for item in factors),
            strategy=self.config.aggregation,
        )
        phi = _aggregate(
            tuple(item.conservative_risk for item in factors),
            weights=tuple(item.weight for item in factors),
            strategy=self.config.aggregation,
        )
        weight_sum = sum(item.weight for item in factors)
        overall_confidence = (
            sum(item.weight * item.estimate.confidence for item in factors) / weight_sum
        )
        dominant_item = max(
            factors,
            key=lambda item: item.weighted_conservative_risk,
        )
        dominant = (
            None
            if dominant_item.weighted_conservative_risk <= 0.0
            else dominant_item.factor
        )
        return CRAAssessment(
            aggregation=self.config.aggregation,
            factors=factors,
            nominal_phi=nominal_phi,
            phi=phi,
            overall_confidence=overall_confidence,
            uncertainty_uplift=max(0.0, phi - nominal_phi),
            dominant_factor=dominant,
        )

    def _assess_factor(
        self,
        factor: PhysicalRiskFactor,
        estimate: RiskFactorEstimate,
    ) -> CRAFactorAssessment:
        """Return uncertainty-adjusted diagnostics for one factor."""

        nominal_risk = estimate.nominal_risk
        fallback_risk = self.config.fallback_risks.value_for(factor)
        uncertainty_increase = (1.0 - estimate.confidence) * max(
            0.0, fallback_risk - nominal_risk
        )
        conservative_risk = min(1.0, nominal_risk + uncertainty_increase)
        weight = self.config.weights.value_for(factor)
        return CRAFactorAssessment(
            factor=factor,
            estimate=estimate,
            fallback_risk=fallback_risk,
            weight=weight,
            nominal_risk=nominal_risk,
            conservative_risk=conservative_risk,
            weighted_conservative_risk=weight * conservative_risk,
            uncertainty_uplift=conservative_risk - nominal_risk,
        )


def _aggregate(
    risks: tuple[float, ...],
    *,
    weights: tuple[float, ...],
    strategy: CRAAggregation,
) -> float:
    """Apply one bounded monotone aggregation strategy."""

    weighted = tuple(weight * risk for weight, risk in zip(weights, risks, strict=True))
    if strategy is CRAAggregation.NOISY_OR:
        return min(1.0, max(0.0, 1.0 - math.prod(1.0 - value for value in weighted)))
    if strategy is CRAAggregation.WEIGHTED_SUM:
        return min(1.0, max(0.0, sum(weighted) / sum(weights)))
    if strategy is CRAAggregation.MAXIMUM:
        return min(1.0, max(0.0, max(weighted)))
    raise RuntimeError(f"Unsupported CRA aggregation strategy: {strategy!r}")


def _physical_factor(value: object) -> PhysicalRiskFactor:
    """Return one supported physical factor."""

    try:
        return PhysicalRiskFactor(value)
    except (TypeError, ValueError) as error:
        raise ValueError("factor must be a supported PhysicalRiskFactor.") from error

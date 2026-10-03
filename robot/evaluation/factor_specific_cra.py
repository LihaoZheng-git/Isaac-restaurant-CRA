"""Report factor evidence and aggregate CRA saturation without hiding unknowns."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

from robot.risk.cra import CRAAssessment, CRARiskAssessor, PhysicalRiskFactor
from robot.risk.physical_factor_adapter import PhysicalFactorAdapterResult


@dataclass(frozen=True)
class FactorInputProvenance:
    name: str
    status: str
    source: str
    timestamp_seconds: float
    confidence: float


@dataclass(frozen=True)
class FactorSpecificCRAReport:
    factor: PhysicalRiskFactor
    available: bool
    nominal_risk: float
    conservative_risk: float
    uncertainty_uplift: float
    confidence: float
    fallback_risk: float
    rejection_reasons: tuple[str, ...]
    inputs: tuple[FactorInputProvenance, ...]
    aggregate_phi_reportable: bool = False
    basis: str = "physical"

    def __post_init__(self) -> None:
        if not isinstance(self.factor, PhysicalRiskFactor):
            raise TypeError("factor must be PhysicalRiskFactor")
        for name in (
            "nominal_risk",
            "conservative_risk",
            "uncertainty_uplift",
            "confidence",
            "fallback_risk",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
            object.__setattr__(self, name, value)
        if self.aggregate_phi_reportable:
            raise ValueError("factor-specific reports cannot claim aggregate phi")


@dataclass(frozen=True)
class FourFactorSaturationReport:
    track_id: int
    timestamp_seconds: float
    nominal_phi: float
    conservative_phi: float
    saturation_threshold: float
    aggregate_saturated: bool
    unknown_fallback_factors: tuple[PhysicalRiskFactor, ...]
    saturation_involves_unknown_fallback: bool
    factors: tuple[FactorSpecificCRAReport, ...]
    assessment: CRAAssessment

    def __post_init__(self) -> None:
        if isinstance(self.track_id, bool) or not isinstance(self.track_id, int):
            raise TypeError("track_id must be an integer")
        if not isinstance(self.assessment, CRAAssessment):
            raise TypeError("assessment must be CRAAssessment")
        if tuple(item.factor for item in self.factors) != tuple(PhysicalRiskFactor):
            raise ValueError("factors must use stable four-factor order")

    def factor(self, factor: PhysicalRiskFactor) -> FactorSpecificCRAReport:
        resolved = PhysicalRiskFactor(factor)
        return self.factors[list(PhysicalRiskFactor).index(resolved)]

    def to_dict(self) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "timestamp_seconds": self.timestamp_seconds,
            "aggregate": {
                "nominal_phi": self.nominal_phi,
                "conservative_phi": self.conservative_phi,
                "saturation_threshold": self.saturation_threshold,
                "saturated": self.aggregate_saturated,
                "unknown_fallback_factors": [
                    item.value for item in self.unknown_fallback_factors
                ],
                "saturation_involves_unknown_fallback": (
                    self.saturation_involves_unknown_fallback
                ),
            },
            "factor_specific": {
                item.factor.value: {
                    **asdict(item),
                    "factor": item.factor.value,
                }
                for item in self.factors
            },
            "scope_note": (
                "factor-specific entries do not report aggregate phi; unknown inputs "
                "remain unknown and contribute configured conservative fallback only "
                "to the separately labeled four-factor aggregate"
            ),
        }

    def write_json(self, path: Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def diagnose_factor_specific_cra(
    result: PhysicalFactorAdapterResult,
    *,
    assessor: CRARiskAssessor | None = None,
    saturation_threshold: float = 0.95,
) -> FourFactorSaturationReport:
    """Separate factor evidence from the conservative four-factor aggregate."""

    if not isinstance(result, PhysicalFactorAdapterResult):
        raise TypeError("result must be PhysicalFactorAdapterResult")
    threshold = float(saturation_threshold)
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("saturation_threshold must lie in (0, 1]")
    risk_assessor = assessor or CRARiskAssessor()
    if not isinstance(risk_assessor, CRARiskAssessor):
        raise TypeError("assessor must be CRARiskAssessor")
    assessment = risk_assessor.assess(result.profile)
    diagnostics = {
        PhysicalRiskFactor.THERMAL: result.thermal,
        PhysicalRiskFactor.CONTACT: result.contact,
        PhysicalRiskFactor.POTENTIAL: result.potential,
        PhysicalRiskFactor.KINETIC: result.kinetic,
    }
    reports = []
    unknown = []
    for factor in PhysicalRiskFactor:
        diagnostic = diagnostics[factor]
        assessed = assessment.factor(factor)
        available = not diagnostic.rejection_reasons and (
            diagnostic.estimate.confidence > 0
        )
        if not available:
            unknown.append(factor)
        reports.append(
            FactorSpecificCRAReport(
                factor=factor,
                available=available,
                nominal_risk=assessed.nominal_risk,
                conservative_risk=assessed.conservative_risk,
                uncertainty_uplift=assessed.uncertainty_uplift,
                confidence=assessed.estimate.confidence,
                fallback_risk=assessed.fallback_risk,
                rejection_reasons=diagnostic.rejection_reasons,
                basis=diagnostic.basis,
                inputs=tuple(
                    FactorInputProvenance(
                        name=name,
                        status=item.status.value,
                        source=item.source,
                        timestamp_seconds=item.timestamp_seconds,
                        confidence=item.confidence,
                    )
                    for name, item in diagnostic.inputs
                ),
            )
        )
    saturated = assessment.phi >= threshold
    return FourFactorSaturationReport(
        track_id=result.track_id,
        timestamp_seconds=result.timestamp_seconds,
        nominal_phi=assessment.nominal_phi,
        conservative_phi=assessment.phi,
        saturation_threshold=threshold,
        aggregate_saturated=saturated,
        unknown_fallback_factors=tuple(unknown),
        saturation_involves_unknown_fallback=bool(saturated and unknown),
        factors=tuple(reports),
        assessment=assessment,
    )

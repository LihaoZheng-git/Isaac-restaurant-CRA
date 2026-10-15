"""Explicit engineering factor priors for the persistent object-map experiment.

Missing physical inputs remain unavailable in adapter diagnostics. The control
policy assigns each such factor a configurable numeric prior, rather than
claiming a measured zero or forcing the entire aggregate to unit risk.
"""

from dataclasses import dataclass, field, replace

from robot.risk.cra import CRAAssessmentConfig, CRARiskAssessor, PhysicalRiskVector
from robot.safety.cra_safe_sage_policy import CRASafeSAGEPolicy


@dataclass(frozen=True)
class ObjectRiskPriorConfig:
    factor_risks: PhysicalRiskVector = field(
        default_factory=lambda: PhysicalRiskVector(
            thermal=0.25, contact=0.25, potential=0.25, kinetic=0.25
        )
    )
    scope: str = (
        "bounded separated restaurant fixtures; uncalibrated normalized factor priors"
    )

    def __post_init__(self):
        if not isinstance(self.factor_risks, PhysicalRiskVector):
            raise TypeError("factor_risks must be PhysicalRiskVector")
        if any(value >= 1 for _, value in self.factor_risks.items()):
            raise ValueError("object-map factor priors must be below one")
        if not isinstance(self.scope, str) or not self.scope.strip():
            raise ValueError("factor priors require an explicit scope")


def evaluate_object_risk(adapted, config):
    if not isinstance(config, ObjectRiskPriorConfig):
        raise TypeError("config must be ObjectRiskPriorConfig")
    policy = CRASafeSAGEPolicy(
        risk_assessor=CRARiskAssessor(
            CRAAssessmentConfig(fallback_risks=config.factor_risks)
        )
    )
    estimates, details = {}, {}
    for factor, estimate in adapted.profile.items():
        diagnostic = getattr(adapted, factor.value)
        missing = bool(diagnostic.rejection_reasons) or estimate.confidence <= 0
        prior = config.factor_risks.value_for(factor)
        evidence = (
            (
                f"object_factor_prior={prior}; scope={config.scope}; engineering assumption, not measured absence",
            )
            if missing
            else (f"uncertainty prior={prior}; scope={config.scope}",)
        )
        estimates[factor.value] = replace(
            estimate,
            source="configured_object_factor_prior" if missing else estimate.source,
            evidence=(*estimate.evidence, *evidence),
        )
        details[factor.value] = {
            "mode": "configured_prior" if missing else "evidence_supported",
            "input_rejections": diagnostic.rejection_reasons,
            "prior_risk": prior,
            "basis": diagnostic.basis,
        }
    result = policy.evaluate(replace(adapted.profile, **estimates))
    for factor, detail in details.items():
        assessed = next(
            item for item in result.assessment.factors if item.factor.value == factor
        )
        detail.update(
            score=assessed.conservative_risk,
            nominal_score=assessed.nominal_risk,
            confidence=assessed.estimate.confidence,
        )
    return result, details

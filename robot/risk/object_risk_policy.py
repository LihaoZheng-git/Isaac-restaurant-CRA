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


def evaluate_object_risk(
    adapted,
    config,
    *,
    height_evidence=None,
    human_body_contact_prior=None,
    human_part_contact=None,
):
    if not isinstance(config, ObjectRiskPriorConfig):
        raise TypeError("config must be ObjectRiskPriorConfig")
    if height_evidence is not None:
        from robot.perception.object_height import ObjectHeightEvidence

        if not isinstance(height_evidence, ObjectHeightEvidence):
            raise TypeError("height_evidence must be ObjectHeightEvidence")
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
        if factor.value == "potential" and height_evidence is not None:
            details[factor.value]["height_context"] = {
                "support_state": height_evidence.support_state,
                "score_basis": "ground_supported_baseline_prior"
                if missing and height_evidence.support_state == "ground_supported"
                else "uncertainty_prior"
                if missing
                else "physical_drop_estimate",
                "support_height_above_ground_m": height_evidence.support_height_above_ground_m.value,
                "reference_scope": height_evidence.reference_scope,
            }
    result = policy.evaluate(replace(adapted.profile, **estimates))
    if human_body_contact_prior is not None:
        result = _apply_human_body_contact(
            adapted,
            config,
            estimates,
            details,
            result,
            human_body_contact_prior,
            human_part_contact,
        )
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


def _apply_human_body_contact(
    adapted, config, estimates, details, baseline, prior, part
):
    from dataclasses import asdict

    from robot.perception.human_body_contact import (
        HUMAN_BODY_CONTACT_SOURCE,
        HUMAN_BODY_CONTACT_BASIS,
    )
    from robot.perception.physical_attributes import (
        EstimateStatus,
        ScalarEstimate,
        combine_estimate_statuses,
    )
    from robot.risk.cra import RiskFactorEstimate
    from robot.risk.physical_factor_adapter import PhysicalFactorDiagnostic

    if not isinstance(prior, ScalarEstimate):
        raise TypeError("human_body_contact_prior must be ScalarEstimate")
    if not prior.is_available:
        details["contact"]["human_body_prior_rejections"] = prior.evidence
        return baseline
    if (
        prior.status is not EstimateStatus.PRIOR
        or prior.source != HUMAN_BODY_CONTACT_SOURCE
        or prior.unit != "1"
        or prior.confidence != 0
        or prior.timestamp_seconds != adapted.timestamp_seconds
        or not 0 < prior.value < 1
        or prior.lower_bound != prior.value
        or prior.upper_bound != prior.value
        or f"track_id={adapted.track_id}" not in prior.evidence
    ):
        details["contact"]["human_body_prior_rejections"] = (
            "valid same-track, same-source-time human policy prior required",
        )
        return baseline
    diagnostic = adapted.contact
    # Complete physical evidence and directly supplied part models retain their
    # legacy policy. Only whole-body silhouette or missing inputs are replaced.
    if diagnostic.basis not in {"physical", "visual_silhouette_geometry_proxy"} or (
        diagnostic.basis == "physical"
        and not diagnostic.rejection_reasons
        and diagnostic.estimate.confidence > 0
    ):
        details["contact"]["human_body_prior_rejections"] = (
            "accepted physical/specialist contact model preserved",
        )
        return baseline
    raw_factor = next(
        f for f in baseline.assessment.factors if f.factor.value == "contact"
    )
    details["contact"] = {
        "mode": "configured_human_body_contact_prior",
        "basis": HUMAN_BODY_CONTACT_BASIS,
        "status": "prior",
        "prior_risk": prior.value,
        "input_rejections": diagnostic.rejection_reasons,
        "human_body_prior": asdict(prior),
        "raw_outline_or_missing_contact": diagnostic.as_dict(),
        "raw_contact_policy_score": raw_factor.conservative_risk,
        "raw_aggregate_phi": baseline.risk,
    }
    fallback = replace(config.factor_risks, contact=prior.value)
    contact = RiskFactorEstimate.unknown(source=prior.source, evidence=prior.evidence)
    selected = CRASafeSAGEPolicy(
        risk_assessor=CRARiskAssessor(CRAAssessmentConfig(fallback_risks=fallback))
    ).evaluate(replace(adapted.profile, **{**estimates, "contact": contact}))
    if part is not None:
        if (
            not isinstance(part, PhysicalFactorDiagnostic)
            or part.factor != "contact"
            or part.basis != "visual_exposed_edge_proxy"
        ):
            raise ValueError("separate accepted part contact diagnostic required")
        part_result = CRASafeSAGEPolicy(
            risk_assessor=CRARiskAssessor(
                CRAAssessmentConfig(fallback_risks=config.factor_risks)
            )
        ).evaluate(replace(adapted.profile, **{**estimates, "contact": part.estimate}))
        part_factor = next(
            f for f in part_result.assessment.factors if f.factor.value == "contact"
        )
        details["contact"]["separate_part_contact"] = part.as_dict()
        details["contact"]["separate_part_policy_score"] = part_factor.conservative_risk
        if part_factor.conservative_risk > prior.value:
            selected = part_result
            details["contact"].update(
                mode="configured_prior"
                if part.rejection_reasons or part.estimate.confidence <= 0
                else "evidence_supported",
                basis=part.basis,
                status="prior"
                if part.rejection_reasons or part.estimate.confidence <= 0
                else combine_estimate_statuses(
                    item.status for _, item in part.inputs
                ).value,
                prior_risk=config.factor_risks.contact,
                input_rejections=part.rejection_reasons,
            )
    return selected

"""Stationary evidence gate for restaurant CRA experiments."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from robot.evaluation.cra_experiments import EvidenceMode, RestaurantScenario
from robot.perception.physical_attributes import EstimateStatus
from robot.risk.physical_factor_adapter import (
    PhysicalFactorAdapterResult,
    PhysicalFactorDiagnostic,
)


FACTOR_NAMES = ("thermal", "contact", "potential", "kinetic")


class EvaluationScope(str, Enum):
    """Whether a trial claims aggregate CRA or one isolated factor."""

    FOUR_FACTOR = "four_factor"
    FACTOR_SPECIFIC = "factor_specific"


@dataclass(frozen=True)
class FactorReadiness:
    factor: str
    accepted: bool
    reasons: tuple[str, ...]
    basis: str = "physical"


@dataclass(frozen=True)
class PerceptionReadinessResult:
    scenario_id: str
    evidence_mode: EvidenceMode
    scope: EvaluationScope
    ready: bool
    aggregate_phi_reportable: bool
    factors: tuple[FactorReadiness, ...]

    @property
    def rejected_factors(self) -> tuple[str, ...]:
        return tuple(item.factor for item in self.factors if not item.accepted)


def _allowed_statuses(mode: EvidenceMode) -> frozenset[EstimateStatus]:
    if mode is EvidenceMode.SENSOR:
        return frozenset({EstimateStatus.MEASURED, EstimateStatus.INFERRED})
    if mode is EvidenceMode.PRIOR:
        return frozenset(
            {
                EstimateStatus.MEASURED,
                EstimateStatus.INFERRED,
                EstimateStatus.PRIOR,
            }
        )
    return frozenset({EstimateStatus.ORACLE})


def _factor_readiness(
    diagnostic: PhysicalFactorDiagnostic,
    *,
    evidence_mode: EvidenceMode,
) -> FactorReadiness:
    reasons = list(diagnostic.rejection_reasons)
    if diagnostic.basis not in {"physical", "visual_exposed_edge_proxy"}:
        reasons.append("unsupported factor basis")
    if diagnostic.basis == "visual_exposed_edge_proxy" and diagnostic.factor != "contact":
        reasons.append("visual exposed-edge proxy is only valid for contact")
    if diagnostic.estimate.confidence <= 0:
        reasons.append("factor estimate has zero confidence")
    allowed = _allowed_statuses(evidence_mode)
    for name, estimate in diagnostic.inputs:
        if estimate.status not in allowed:
            reasons.append(
                f"{name}: status={estimate.status.value} is not valid for "
                f"evidence_mode={evidence_mode.value}"
            )
    return FactorReadiness(
        diagnostic.factor,
        not reasons,
        tuple(dict.fromkeys(reasons)),
        diagnostic.basis,
    )


def evaluate_perception_readiness(
    scenario: RestaurantScenario,
    result: PhysicalFactorAdapterResult,
    *,
    scope: EvaluationScope = EvaluationScope.FOUR_FACTOR,
) -> PerceptionReadinessResult:
    """Validate evidence provenance before any restaurant motion trial.

    Four-factor scope requires every factor to be available with provenance
    matching the scenario evidence mode. Factor-specific scope checks only the
    scenario's named hazard family and explicitly forbids reporting aggregate
    ``phi`` from that acceptance result.
    """

    if not isinstance(scenario, RestaurantScenario):
        raise TypeError("scenario must be a RestaurantScenario")
    if not isinstance(result, PhysicalFactorAdapterResult):
        raise TypeError("result must be a PhysicalFactorAdapterResult")
    try:
        evaluation_scope = EvaluationScope(scope)
    except (TypeError, ValueError) as error:
        raise ValueError("scope must be a supported EvaluationScope") from error
    if scenario.hazard_family not in FACTOR_NAMES:
        raise ValueError(
            f"unsupported scenario hazard_family={scenario.hazard_family!r}"
        )

    diagnostics = {name: getattr(result, name) for name in FACTOR_NAMES}
    if (result.contact.basis == "visual_exposed_edge_proxy"
            and not result.config.allow_visual_contact_proxy):
        raise ValueError("contact proxy readiness requires adapter configuration opt-in")
    checked = (
        FACTOR_NAMES
        if evaluation_scope is EvaluationScope.FOUR_FACTOR
        else (scenario.hazard_family,)
    )
    factors = tuple(
        _factor_readiness(
            diagnostics[name], evidence_mode=scenario.evidence_mode
        )
        for name in checked
    )
    ready = all(item.accepted for item in factors)
    return PerceptionReadinessResult(
        scenario_id=scenario.scenario_id,
        evidence_mode=scenario.evidence_mode,
        scope=evaluation_scope,
        ready=ready,
        aggregate_phi_reportable=(
            ready and evaluation_scope is EvaluationScope.FOUR_FACTOR
        ),
        factors=factors,
    )

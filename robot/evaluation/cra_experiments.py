"""Matched restaurant experiment plans and result aggregation."""

from dataclasses import dataclass
from enum import Enum
import math
from statistics import mean

from robot.risk.cra import PhysicalRiskFactor
from robot.scene.restaurant_risk_scenarios import (
    primary_restaurant_risk_scenarios,
)


class CRABaseline(str, Enum):
    GEOMETRIC_VELOCITY_CBF = "geometric_velocity_cbf"
    FIXED_CLASS_SAFE_SAGE = "fixed_class_safe_sage"
    CONTINUOUS_CRA_SAFE_SAGE = "continuous_cra_safe_sage"
    CRA_RISK_DEPENDENT_LAMBDA_ABLATION = "cra_risk_dependent_lambda_ablation"


class EvidenceMode(str, Enum):
    ORACLE = "oracle"
    PRIOR = "prior"
    SENSOR = "sensor"


PRIMARY_BASELINES = (
    CRABaseline.GEOMETRIC_VELOCITY_CBF,
    CRABaseline.FIXED_CLASS_SAFE_SAGE,
    CRABaseline.CONTINUOUS_CRA_SAFE_SAGE,
)


@dataclass(frozen=True)
class RestaurantScenario:
    scenario_id: str
    hazard_family: str
    description: str
    evidence_mode: EvidenceMode
    seed: int


@dataclass(frozen=True)
class RestaurantTrialResult:
    scenario_id: str
    seed: int
    baseline: CRABaseline
    evidence_mode: EvidenceMode
    completed: bool
    fail_closed: bool
    collision: bool
    minimum_clearance_m: float
    intervention_count: int
    fallback_count: int
    elapsed_seconds: float
    minimum_cbf_margin: float
    lambda_value: float

    def __post_init__(self):
        for name in (
            "minimum_clearance_m",
            "elapsed_seconds",
            "minimum_cbf_margin",
            "lambda_value",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.minimum_clearance_m < 0 or self.elapsed_seconds < 0:
            raise ValueError("clearance and elapsed time cannot be negative")
        if self.intervention_count < 0 or self.fallback_count < 0:
            raise ValueError("counts cannot be negative")
        if self.baseline in PRIMARY_BASELINES[1:] and self.lambda_value != 1.25:
            raise ValueError("primary Safe-SAGE comparisons require lambda=1.25")
        if self.collision and self.completed:
            raise ValueError("a collided trial cannot be marked completed")


@dataclass(frozen=True)
class BaselineSummary:
    evidence_mode: EvidenceMode
    baseline: CRABaseline
    trial_count: int
    completion_rate: float
    collision_rate: float
    fail_closed_rate: float
    mean_minimum_clearance_m: float
    mean_intervention_count: float
    mean_fallback_count: float


@dataclass(frozen=True)
class FactorSensorReadiness:
    """Evidence required before a factor enters wheel-motion experiments."""

    factor: PhysicalRiskFactor
    stationary_pipeline_validated: bool
    runtime_provenance_validated: bool
    sensor_domain_accuracy_validated: bool
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.factor, PhysicalRiskFactor):
            raise TypeError("factor must be PhysicalRiskFactor")
        for name in (
            "stationary_pipeline_validated",
            "runtime_provenance_validated",
            "sensor_domain_accuracy_validated",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        limitations = tuple(self.limitations)
        if any(not isinstance(item, str) or not item.strip() for item in limitations):
            raise ValueError("limitations must contain nonempty text")
        object.__setattr__(self, "limitations", limitations)

    @property
    def ready_for_motion(self) -> bool:
        return (
            self.stationary_pipeline_validated
            and self.runtime_provenance_validated
            and self.sensor_domain_accuracy_validated
        )


@dataclass(frozen=True)
class RestaurantMotionReadiness:
    """Factor-specific gate for sensor-derived restaurant motion trials."""

    factors: tuple[FactorSensorReadiness, ...]
    blocking_reasons: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return not self.blocking_reasons


def assess_restaurant_motion_readiness(
    factors: tuple[FactorSensorReadiness, ...],
) -> RestaurantMotionReadiness:
    """Require all four factor pipelines without converting gaps to zero."""

    items = tuple(factors)
    by_factor = {item.factor: item for item in items}
    if len(by_factor) != len(items):
        raise ValueError("factor readiness contains duplicates")
    missing = tuple(factor for factor in PhysicalRiskFactor if factor not in by_factor)
    reasons = [f"{factor.value}: readiness record unavailable" for factor in missing]
    for factor in PhysicalRiskFactor:
        item = by_factor.get(factor)
        if item is None or item.ready_for_motion:
            continue
        if not item.stationary_pipeline_validated:
            reasons.append(f"{factor.value}: stationary pipeline not validated")
        if not item.runtime_provenance_validated:
            reasons.append(f"{factor.value}: runtime provenance not validated")
        if not item.sensor_domain_accuracy_validated:
            reasons.append(f"{factor.value}: sensor-domain accuracy not validated")
        reasons.extend(f"{factor.value}: {text}" for text in item.limitations)
    return RestaurantMotionReadiness(items, tuple(reasons))


def require_sensor_motion_readiness(
    report: RestaurantMotionReadiness,
) -> None:
    """Fail before wheel motion when any physical factor is unsupported."""

    if not isinstance(report, RestaurantMotionReadiness):
        raise TypeError("report must be RestaurantMotionReadiness")
    if not report.ready:
        raise RuntimeError(
            "sensor-derived restaurant motion is not ready: "
            + "; ".join(report.blocking_reasons)
        )


def validate_matched_trials(results: tuple[RestaurantTrialResult, ...]) -> None:
    """Require every scenario/seed/mode cell for every primary baseline."""
    cells: dict[tuple[str, int, EvidenceMode], set[CRABaseline]] = {}
    for item in results:
        key = (item.scenario_id, item.seed, item.evidence_mode)
        baselines = cells.setdefault(key, set())
        if item.baseline in baselines:
            raise ValueError(f"duplicate trial cell: {key} {item.baseline.value}")
        baselines.add(item.baseline)
    required = set(PRIMARY_BASELINES)
    incomplete = tuple(
        key for key, baselines in cells.items() if not required <= baselines
    )
    if incomplete:
        raise ValueError(f"unmatched primary baseline cells: {incomplete}")


def summarize_trials(
    results: tuple[RestaurantTrialResult, ...],
) -> tuple[BaselineSummary, ...]:
    """Summarize matched trials without pooling evidence modes."""

    validate_matched_trials(results)
    summaries = []
    for evidence_mode in EvidenceMode:
        for baseline in CRABaseline:
            items = tuple(
                item
                for item in results
                if item.evidence_mode is evidence_mode
                and item.baseline is baseline
            )
            if not items:
                continue
            count = len(items)
            summaries.append(
                BaselineSummary(
                    evidence_mode,
                    baseline,
                    count,
                    mean(item.completed for item in items),
                    mean(item.collision for item in items),
                    mean(item.fail_closed for item in items),
                    mean(item.minimum_clearance_m for item in items),
                    mean(item.intervention_count for item in items),
                    mean(item.fallback_count for item in items),
                )
            )
    return tuple(summaries)


def primary_restaurant_scenarios(
    *, evidence_mode: EvidenceMode = EvidenceMode.SENSOR, seed: int = 0
) -> tuple[RestaurantScenario, ...]:
    return tuple(
        RestaurantScenario(
            item.scenario_id,
            item.factor.value,
            item.description,
            evidence_mode,
            seed,
        )
        for item in primary_restaurant_risk_scenarios()
    )

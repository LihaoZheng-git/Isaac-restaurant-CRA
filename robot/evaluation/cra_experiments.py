"""Matched restaurant experiment plans and result aggregation."""

from dataclasses import dataclass
from enum import Enum
import math
from statistics import mean


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
    baseline: CRABaseline
    trial_count: int
    completion_rate: float
    collision_rate: float
    fail_closed_rate: float
    mean_minimum_clearance_m: float
    mean_intervention_count: float
    mean_fallback_count: float


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
    validate_matched_trials(results)
    summaries = []
    for baseline in CRABaseline:
        items = tuple(item for item in results if item.baseline is baseline)
        if not items:
            continue
        count = len(items)
        summaries.append(
            BaselineSummary(
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
        RestaurantScenario(identifier, family, description, evidence_mode, seed)
        for identifier, family, description in (
            ("chair", "potential", "ordinary chair on floor"),
            ("crossing_person", "kinetic", "person crossing nominal route"),
            (
                "elevated_object",
                "potential",
                "object above a localized landing surface",
            ),
            ("exposed_utensil", "contact", "fixed-label exposed blade or tines"),
            ("carried_hot_liquid", "thermal", "open carried container with steam cue"),
            ("stove_flame", "thermal", "visible flame source"),
        )
    )

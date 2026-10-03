"""Offline relative-heaviness diagnostics; catalog labels are never sensors."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class HeavinessPrediction:
    object_id: str
    semantic_class: str
    reference_mass_kg: float
    predicted_mass_kg: float

    def __post_init__(self) -> None:
        if not self.object_id or not self.semantic_class:
            raise ValueError("object and class identifiers are required")
        for value in (self.reference_mass_kg, self.predicted_mass_kg):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("mass values must be positive and finite")


def evaluate_relative_heaviness(
    predictions: tuple[HeavinessPrediction, ...], *, minimum_mass_ratio: float = 3.0,
) -> dict[str, object]:
    """Evaluate every eligible pair, including failures and prediction ties.

    The ratio is chosen before inference. All-object and within-class results
    are separate so successful class-scale ordering cannot hide weak ordering
    of two objects from the same class. Pairs share objects and are not
    independent statistical samples. This is not a live-domain readiness gate.
    """
    if not math.isfinite(minimum_mass_ratio) or minimum_mass_ratio <= 1:
        raise ValueError("minimum_mass_ratio must be finite and greater than one")
    ids = [item.object_id for item in predictions]
    if len(set(ids)) != len(ids):
        raise ValueError("one prediction per independent object is required")
    groups = {"all_pairs": [], "within_class": []}
    for index, first in enumerate(predictions):
        for second in predictions[index + 1:]:
            light, heavy = sorted((first, second), key=lambda item: item.reference_mass_kg)
            if heavy.reference_mass_kg / light.reference_mass_kg < minimum_mass_ratio:
                continue
            outcome = (heavy.predicted_mass_kg > light.predicted_mass_kg) - (
                heavy.predicted_mass_kg < light.predicted_mass_kg)
            groups["all_pairs"].append(outcome)
            if light.semantic_class == heavy.semantic_class:
                groups["within_class"].append(outcome)
    report: dict[str, object] = {
        "independent_object_count": len(predictions),
        "minimum_reference_mass_ratio": minimum_mass_ratio,
        "sensor_domain_acceptance": False,
        "pair_dependence": "pairs share objects; descriptive metric only",
    }
    for name, outcomes in groups.items():
        report[name] = {
            "pair_count": len(outcomes), "correct": outcomes.count(1),
            "reversed": outcomes.count(-1), "ties": outcomes.count(0),
            "ordering_accuracy": outcomes.count(1) / len(outcomes) if outcomes else None,
        }
    return report

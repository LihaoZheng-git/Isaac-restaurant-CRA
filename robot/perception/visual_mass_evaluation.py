"""Offline Isaac-domain evaluation for inferred visual-mass estimates."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from robot.perception.physical_attributes import EstimateStatus, MassAttributes
from robot.perception.visual_mass_dataset import (
    VisualMassMetrics,
    VisualMassSample,
    evaluate_mass_predictions,
)
from robot.perception.visual_mass_runtime import VISUAL_MASS_RUNTIME_SOURCE


@dataclass(frozen=True)
class VisualMassDomainPrediction:
    """One runtime estimate retained without its oracle target."""

    sample_id: str
    predicted_mass_kg: float
    lower_bound_kg: float
    upper_bound_kg: float
    confidence: float
    status: EstimateStatus
    source: str
    checkpoint_sha256: str

    @classmethod
    def from_attributes(
        cls,
        sample_id: str,
        attributes: MassAttributes,
        *,
        checkpoint_sha256: str,
    ) -> "VisualMassDomainPrediction":
        if not isinstance(attributes, MassAttributes):
            raise TypeError("attributes must be MassAttributes")
        estimate = attributes.mass_kg
        if not estimate.is_available:
            raise ValueError("mass estimate must be available for evaluation")
        return cls(
            sample_id,
            estimate.value,
            estimate.lower_bound,
            estimate.upper_bound,
            estimate.confidence,
            estimate.status,
            estimate.source,
            checkpoint_sha256,
        )

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id.strip():
            raise ValueError("sample_id must be nonempty text")
        values = (
            self.predicted_mass_kg,
            self.lower_bound_kg,
            self.upper_bound_kg,
            self.confidence,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("prediction values must be finite")
        if not 0 < self.lower_bound_kg <= self.predicted_mass_kg <= self.upper_bound_kg:
            raise ValueError("mass prediction bounds are inconsistent")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must lie in [0, 1]")
        if self.status is not EstimateStatus.INFERRED:
            raise ValueError("Isaac visual-mass evaluation requires inferred estimates")
        if self.source != VISUAL_MASS_RUNTIME_SOURCE:
            raise ValueError("prediction source must be the visual-mass runtime adapter")
        digest = self.checkpoint_sha256.lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("checkpoint_sha256 must be a hexadecimal SHA-256 digest")
        object.__setattr__(self, "checkpoint_sha256", digest)


@dataclass(frozen=True)
class VisualMassClassEvaluation:
    semantic_class: str
    independent_object_count: int
    metrics: VisualMassMetrics
    interval_coverage: float


@dataclass(frozen=True)
class VisualMassDomainEvaluationConfig:
    minimum_independent_test_objects: int = 10
    minimum_objects_per_reported_class: int = 2
    sensor_derived_geometry: bool = False

    def __post_init__(self) -> None:
        for name in (
            "minimum_independent_test_objects",
            "minimum_objects_per_reported_class",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.sensor_derived_geometry, bool):
            raise TypeError("sensor_derived_geometry must be boolean")


@dataclass(frozen=True)
class VisualMassDomainEvaluation:
    independent_object_count: int
    sample_count: int
    checkpoint_sha256: str | None
    overall_metrics: VisualMassMetrics | None
    interval_coverage: float | None
    per_class: tuple[VisualMassClassEvaluation, ...]
    blockers: tuple[str, ...]

    @property
    def ready_for_sensor_domain_claim(self) -> bool:
        return not self.blockers


def evaluate_visual_mass_isaac_domain(
    samples: Iterable[VisualMassSample],
    predictions: Iterable[VisualMassDomainPrediction],
    *,
    config: VisualMassDomainEvaluationConfig | None = None,
) -> VisualMassDomainEvaluation:
    """Average repeated views per object before computing test metrics."""

    settings = config or VisualMassDomainEvaluationConfig()
    test_samples = tuple(item for item in samples if item.split == "test")
    prediction_items = tuple(predictions)
    by_sample = {item.sample_id: item for item in prediction_items}
    if len(by_sample) != len(prediction_items):
        raise ValueError("predictions contain duplicate sample IDs")
    expected_ids = {item.sample_id for item in test_samples}
    extra = set(by_sample) - expected_ids
    missing = expected_ids - set(by_sample)
    blockers = []
    if not settings.sensor_derived_geometry:
        blockers.append("geometry inputs are not identified as sensor-derived")
    if extra:
        blockers.append(f"predictions contain non-test samples: {sorted(extra)}")
    if missing:
        blockers.append(f"test samples missing predictions: {sorted(missing)}")
    checkpoint_hashes = {item.checkpoint_sha256 for item in prediction_items}
    if len(checkpoint_hashes) > 1:
        blockers.append("predictions use multiple checkpoints")
    complete_samples = tuple(item for item in test_samples if item.sample_id in by_sample)
    objects: dict[str, list[VisualMassSample]] = {}
    for item in complete_samples:
        objects.setdefault(item.object_id, []).append(item)
    object_rows = []
    for object_id, views in sorted(objects.items()):
        truths = {item.mass_kg for item in views}
        classes = {item.semantic_class for item in views}
        if len(truths) != 1 or len(classes) != 1:
            raise ValueError(f"object {object_id} has inconsistent offline labels")
        estimates = [by_sample[item.sample_id] for item in views]
        object_rows.append(
            (
                next(iter(classes)),
                next(iter(truths)),
                sum(item.predicted_mass_kg for item in estimates) / len(estimates),
                sum(item.lower_bound_kg for item in estimates) / len(estimates),
                sum(item.upper_bound_kg for item in estimates) / len(estimates),
            )
        )
    if len(object_rows) < settings.minimum_independent_test_objects:
        blockers.append(
            "independent test objects below minimum: "
            f"{len(object_rows)} < {settings.minimum_independent_test_objects}"
        )
    if not object_rows:
        return VisualMassDomainEvaluation(
            0,
            len(test_samples),
            next(iter(checkpoint_hashes), None),
            None,
            None,
            (),
            tuple(blockers),
        )
    overall = evaluate_mass_predictions(
        (item[1] for item in object_rows),
        (item[2] for item in object_rows),
    )
    coverage = sum(item[3] <= item[1] <= item[4] for item in object_rows) / len(
        object_rows
    )
    per_class = []
    for name in sorted({item[0] for item in object_rows}):
        rows = tuple(item for item in object_rows if item[0] == name)
        if len(rows) < settings.minimum_objects_per_reported_class:
            blockers.append(
                f"{name}: independent test objects below per-class minimum: "
                f"{len(rows)} < {settings.minimum_objects_per_reported_class}"
            )
        per_class.append(
            VisualMassClassEvaluation(
                name,
                len(rows),
                evaluate_mass_predictions(
                    (item[1] for item in rows),
                    (item[2] for item in rows),
                ),
                sum(item[3] <= item[1] <= item[4] for item in rows) / len(rows),
            )
        )
    return VisualMassDomainEvaluation(
        len(object_rows),
        len(test_samples),
        next(iter(checkpoint_hashes), None) if len(checkpoint_hashes) == 1 else None,
        overall,
        coverage,
        tuple(per_class),
        tuple(blockers),
    )

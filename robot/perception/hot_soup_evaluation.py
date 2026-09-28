"""Held-out fixed-label metrics for hot-soup box detections."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from robot.perception.hot_soup_dataset import (
    HOT_SOUP_DETECTION_CLASSES,
    HotSoupBox,
    HotSoupSample,
)
from robot.perception.hot_soup_detector import HotSoupDetectionFrame


@dataclass(frozen=True)
class HotSoupPrediction:
    class_name: str
    confidence: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        if self.class_name not in HOT_SOUP_DETECTION_CLASSES:
            raise ValueError("unsupported hot-soup prediction class")
        values = tuple(
            float(getattr(self, name))
            for name in ("confidence", "x_min", "y_min", "x_max", "y_max")
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("prediction values must be finite")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must lie in [0, 1]")
        if not 0 <= self.x_min < self.x_max <= 1:
            raise ValueError("horizontal coordinates must define a normalized box")
        if not 0 <= self.y_min < self.y_max <= 1:
            raise ValueError("vertical coordinates must define a normalized box")


@dataclass(frozen=True)
class HotSoupClassMetrics:
    class_name: str
    true_positive_count: int
    false_positive_count: int
    false_negative_count: int
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class HotSoupEvaluationReport:
    sample_count: int
    split_counts: tuple[tuple[str, int], ...]
    class_metrics: tuple[HotSoupClassMetrics, ...]
    hard_negative_sample_count: int
    hard_negative_plume_false_positive_count: int


def predictions_from_hot_soup_frame(
    frame: HotSoupDetectionFrame,
) -> tuple[HotSoupPrediction, ...]:
    """Normalize one runtime detector frame for held-out evaluation."""

    if not isinstance(frame, HotSoupDetectionFrame):
        raise TypeError("frame must be HotSoupDetectionFrame")
    return tuple(
        HotSoupPrediction(
            item.semantic_class,
            item.confidence,
            item.bounding_box.x_min / frame.image_width,
            item.bounding_box.y_min / frame.image_height,
            item.bounding_box.x_max / frame.image_width,
            item.bounding_box.y_max / frame.image_height,
        )
        for item in frame.detections
    )


def evaluate_hot_soup_predictions(
    samples: tuple[HotSoupSample, ...],
    predictions_by_sample_id: Mapping[str, tuple[HotSoupPrediction, ...]],
    *,
    confidence_threshold: float = 0.35,
    iou_threshold: float = 0.5,
) -> HotSoupEvaluationReport:
    """Evaluate each fixed class independently with greedy IoU matching."""

    confidence = float(confidence_threshold)
    iou = float(iou_threshold)
    if not 0 <= confidence <= 1 or not 0 < iou <= 1:
        raise ValueError("confidence and IoU thresholds are invalid")
    sample_ids = {item.sample_id for item in samples}
    unknown = set(predictions_by_sample_id) - sample_ids
    if unknown:
        raise ValueError(f"predictions reference unknown samples: {sorted(unknown)}")
    totals = {name: [0, 0, 0] for name in HOT_SOUP_DETECTION_CLASSES}
    hard_negative_count = 0
    hard_negative_false_positive_count = 0
    split_counts = {name: 0 for name in ("train", "val", "test")}
    for sample in samples:
        split_counts[sample.split] += 1
        predictions = tuple(
            item
            for item in predictions_by_sample_id.get(sample.sample_id, ())
            if item.confidence >= confidence
        )
        if sample.hard_negative_tags:
            hard_negative_count += 1
            if any(
                item.class_name == "visible_hot_soup_plume" for item in predictions
            ):
                hard_negative_false_positive_count += 1
        for class_name in HOT_SOUP_DETECTION_CLASSES:
            truth = [box for box in sample.boxes if box.class_name == class_name]
            candidates = sorted(
                (item for item in predictions if item.class_name == class_name),
                key=lambda item: item.confidence,
                reverse=True,
            )
            unmatched = set(range(len(truth)))
            true_positives = 0
            for prediction in candidates:
                ranked = sorted(
                    (
                        (_box_iou(prediction, truth[index]), index)
                        for index in unmatched
                    ),
                    reverse=True,
                )
                if ranked and ranked[0][0] >= iou:
                    unmatched.remove(ranked[0][1])
                    true_positives += 1
            totals[class_name][0] += true_positives
            totals[class_name][1] += len(candidates) - true_positives
            totals[class_name][2] += len(unmatched)
    metrics = []
    for class_name in HOT_SOUP_DETECTION_CLASSES:
        true_positive, false_positive, false_negative = totals[class_name]
        precision = _ratio(true_positive, true_positive + false_positive)
        recall = _ratio(true_positive, true_positive + false_negative)
        f1 = _ratio(2 * precision * recall, precision + recall)
        metrics.append(
            HotSoupClassMetrics(
                class_name,
                true_positive,
                false_positive,
                false_negative,
                precision,
                recall,
                f1,
            )
        )
    return HotSoupEvaluationReport(
        sample_count=len(samples),
        split_counts=tuple(split_counts.items()),
        class_metrics=tuple(metrics),
        hard_negative_sample_count=hard_negative_count,
        hard_negative_plume_false_positive_count=(
            hard_negative_false_positive_count
        ),
    )


def _ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _box_iou(prediction: HotSoupPrediction, truth: HotSoupBox) -> float:
    intersection_width = max(
        0.0,
        min(prediction.x_max, truth.x_max) - max(prediction.x_min, truth.x_min),
    )
    intersection_height = max(
        0.0,
        min(prediction.y_max, truth.y_max) - max(prediction.y_min, truth.y_min),
    )
    intersection = intersection_width * intersection_height
    prediction_area = (
        (prediction.x_max - prediction.x_min)
        * (prediction.y_max - prediction.y_min)
    )
    truth_area = (truth.x_max - truth.x_min) * (truth.y_max - truth.y_min)
    return _ratio(intersection, prediction_area + truth_area - intersection)

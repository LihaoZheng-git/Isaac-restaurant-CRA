"""Evaluate visual mass on persisted Isaac captures without runtime oracle use."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot.perception.physical_attributes import (
    EstimateStatus,
    PhysicalAttributes,
    ScalarEstimate,
)
from robot.perception.semantic_detection import PixelBoundingBox, SemanticDetection
from robot.perception.visual_mass_dataset import read_visual_mass_manifest
from robot.perception.visual_mass_evaluation import (
    VisualMassDomainEvaluationConfig,
    VisualMassDomainPrediction,
    evaluate_visual_mass_isaac_domain,
)
from robot.perception.visual_mass_runtime import (
    VisualMassRuntimeConfig,
    VisualMassRuntimeEstimator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--minimum-test-objects", type=int, default=10)
    parser.add_argument("--minimum-per-class", type=int, default=2)
    return parser.parse_args()


def _scalar(
    value: float, *, status: EstimateStatus, source: str
) -> ScalarEstimate:
    return ScalarEstimate(
        value,
        value,
        value,
        1.0,
        "m",
        source,
        0.0,
        status,
    )


def main() -> None:
    args = parse_args()
    samples = read_visual_mass_manifest(args.manifest)
    estimator = VisualMassRuntimeEstimator(
        VisualMassRuntimeConfig(args.checkpoint, device=args.device)
    )
    predictions = []
    for item in samples:
        if item.split != "test":
            continue
        image = np.load(args.dataset_root / item.rgb_path, allow_pickle=False)
        if item.mask_path is None:
            raise RuntimeError(f"test sample requires an instance mask: {item.sample_id}")
        mask = np.load(args.dataset_root / item.mask_path, allow_pickle=False)
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            raise RuntimeError(f"test mask is empty: {item.sample_id}")
        detection = SemanticDetection(
            0,
            item.semantic_class,
            1.0,
            PixelBoundingBox(
                float(xs.min()),
                float(ys.min()),
                float(xs.max() + 1),
                float(ys.max() + 1),
            ),
            mask,
        )
        base = PhysicalAttributes.unknown(track_id=0, timestamp_seconds=0.0)
        geometry = replace(
            base.geometry,
            width_m=_scalar(
                item.width_m,
                status=item.geometry_status,
                source=item.geometry_source,
            ),
            depth_m=_scalar(
                item.depth_m,
                status=item.geometry_status,
                source=item.geometry_source,
            ),
            height_m=_scalar(
                item.height_m,
                status=item.geometry_status,
                source=item.geometry_source,
            ),
        )
        attributes = estimator.estimate(
            image,
            detection,
            geometry,
            timestamp_seconds=0.0,
        )
        predictions.append(
            VisualMassDomainPrediction.from_attributes(
                item.sample_id,
                attributes,
                geometry,
                checkpoint_sha256=estimator.checkpoint_sha256,
            )
        )
    report = evaluate_visual_mass_isaac_domain(
        samples,
        predictions,
        config=VisualMassDomainEvaluationConfig(
            args.minimum_test_objects,
            args.minimum_per_class,
        ),
    )
    payload = asdict(report)
    if report.overall_metrics is not None:
        payload["overall_metrics"] = asdict(report.overall_metrics)
    payload["oracle_boundary"] = (
        "authored mass read only after runtime prediction for offline scoring"
    )
    payload["geometry_evidence"] = sorted(
        {
            f"{item.geometry_status.value}:{item.geometry_source}"
            for item in samples
            if item.split == "test"
        }
    )
    payload["ready_for_sensor_domain_claim"] = (
        report.ready_for_sensor_domain_claim
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not report.ready_for_sensor_domain_claim:
        raise RuntimeError("Isaac visual-mass domain evaluation is not ready")


if __name__ == "__main__":
    main()

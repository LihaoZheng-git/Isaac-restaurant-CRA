"""Run a positive contact-part association on one held-out local image."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from robot.perception.contact_detection_adapter import (
    associate_contact_parts_to_objects,
)
from robot.perception.contact_replay import evaluate_contact_replay
from robot.perception.contact_geometry_features import (
    estimate_contact_geometry_features, estimate_exposed_edge_proxy,
)
from robot.perception.visualization import render_semantic_detection_overlay
from robot.perception.yolo_detector import (
    YoloSegmentationConfig,
    YoloSegmentationDetector,
)


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--object-checkpoint", type=Path, required=True)
    parser.add_argument("--contact-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--confidence", type=float, default=0.25)
    return parser.parse_args()


def detector(path: Path, args: argparse.Namespace, *, contact: bool):
    result = YoloSegmentationDetector(
        YoloSegmentationConfig(
            model_path=path,
            confidence_threshold=args.confidence,
            device=args.device,
            allowed_classes=("exposed_edge",) if contact else None,
        )
    )
    result.initialize()
    return result


def main() -> None:
    args = parse_args()
    paths = (args.image, args.object_checkpoint, args.contact_checkpoint)
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("image and both checkpoints must exist locally")
    image_rgb = np.asarray(Image.open(args.image).convert("RGB"))
    object_frame = detector(args.object_checkpoint, args, contact=False).detect(
        image_rgb, source_frame_id=0, source_timestamp_seconds=0.0
    )
    part_frame = detector(args.contact_checkpoint, args, contact=True).detect(
        image_rgb, source_frame_id=0, source_timestamp_seconds=0.0
    )
    association = associate_contact_parts_to_objects(object_frame, part_frame)
    report = evaluate_contact_replay(
        object_frame,
        part_frame,
        association,
        image_sha256=sha256_path(args.image),
        object_checkpoint_sha256=sha256_path(args.object_checkpoint),
        contact_checkpoint_sha256=sha256_path(args.contact_checkpoint),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(render_semantic_detection_overlay(image_rgb, object_frame)).save(
        args.output / "object_instances.png"
    )
    Image.fromarray(render_semantic_detection_overlay(image_rgb, part_frame)).save(
        args.output / "exposed_edges.png"
    )
    report.write_json(args.output / "diagnostic.json")
    proxy_rows = []
    for index, segmentation in association.segmentations_by_object_detection_index.items():
        features = estimate_contact_geometry_features(segmentation, timestamp_seconds=0.0)
        score = estimate_exposed_edge_proxy(features)
        proxy_rows.append({"object_detection_index": index, "value": score.value,
                           "lower_bound": score.lower_bound, "upper_bound": score.upper_bound,
                           "confidence": score.confidence, "status": score.status.value,
                           "source": score.source, "evidence": score.evidence,
                           "physical_sharpness": features.sharpness.status.value,
                           "hardness": features.hardness.status.value})
    (args.output / "visual_proxy.json").write_text(json.dumps(proxy_rows, indent=2) + "\n", encoding="utf-8")
    print(report.to_dict())
    if not report.positive_association_valid:
        raise RuntimeError("no positive contact part-to-parent association was accepted")


if __name__ == "__main__":
    main()

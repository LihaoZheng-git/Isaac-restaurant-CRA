"""Repeat contact segmentation on one stationary frame and report stability."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.iterations < 2:
        raise ValueError("iterations must be at least two")
    if not args.checkpoint.is_file() or not args.image.is_file():
        raise FileNotFoundError("checkpoint and stationary image must exist")
    from ultralytics import YOLO

    model = YOLO(str(args.checkpoint))
    model.predict(
        str(args.image), device=args.device, conf=args.confidence, verbose=False
    )
    durations = []
    signatures = []
    counts = []
    confidences = []
    for _ in range(args.iterations):
        start = perf_counter()
        result = model.predict(
            str(args.image), device=args.device, conf=args.confidence, verbose=False
        )[0]
        durations.append((perf_counter() - start) * 1000)
        counts.append(len(result.boxes))
        confidences.append(
            tuple(round(float(value), 6) for value in result.boxes.conf.cpu().numpy())
        )
        mask = (
            result.masks.data.cpu().numpy().astype(np.bool_)
            if result.masks is not None
            else np.empty((0,), dtype=np.bool_)
        )
        signatures.append(hashlib.sha256(mask.tobytes()).hexdigest())
    report = {
        "checkpoint": str(args.checkpoint),
        "image": str(args.image),
        "iterations": args.iterations,
        "confidence_threshold": args.confidence,
        "median_wall_ms": float(np.median(durations)),
        "p95_wall_ms": float(np.percentile(durations, 95)),
        "detection_counts": sorted(set(counts)),
        "unique_mask_signatures": len(set(signatures)),
        "unique_confidence_vectors": len(set(confidences)),
        "scope": "fixed-frame visual geometry inference only",
        "unsupported": [
            "physical_sharpness",
            "material_hardness",
            "route_orientation",
            "contact_probability",
            "injury_severity",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if len(set(counts)) != 1 or len(set(signatures)) != 1:
        raise RuntimeError("stationary segmentation output changed between runs")


if __name__ == "__main__":
    main()

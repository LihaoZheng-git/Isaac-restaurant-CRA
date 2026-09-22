"""Stationary offline diagnostic for a trained visual-mass checkpoint."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot.perception.abo_visual_mass import read_abo_subset_manifest
from robot.perception.visual_mass_model import (
    AboVisualMassDataset,
    load_visual_mass_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--repetitions", type=int, default=100)
    return parser.parse_args()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = parse_args()
    if args.repetitions <= 0:
        raise ValueError("repetitions must be positive")
    device = torch.device(args.device)
    candidates = read_abo_subset_manifest(args.manifest)
    test_items = tuple(item for item in candidates if item.split == "test")
    dataset = AboVisualMassDataset(test_items, args.image_root, training=False)
    model, checkpoint = load_visual_mass_checkpoint(
        args.checkpoint, map_location=device
    )
    pixels, dimensions, log_mass, object_id, semantic_class = dataset[0]
    pixels = pixels.unsqueeze(0).to(device)
    dimensions = dimensions.unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(10):
            model(pixels, dimensions)
        _synchronize(device)
        durations = []
        for _ in range(args.repetitions):
            start = time.perf_counter()
            prediction = model.predict_mass_kg(pixels, dimensions)
            _synchronize(device)
            durations.append((time.perf_counter() - start) * 1000)

    print("STATIONARY OFFLINE DIAGNOSTIC: no robot or wheel commands")
    print(f"object: {object_id} ({semantic_class})")
    print(f"prediction: {float(prediction):.4f} kg (model-inferred)")
    print(f"catalog label: {math.exp(float(log_mass)):.4f} kg (offline metadata)")
    print("runtime CRA evidence status: unavailable")
    print(f"device: {device}")
    print(f"median model latency: {statistics.median(durations):.3f} ms")
    print(f"p95 model latency: {sorted(durations)[int(0.95 * len(durations)) - 1]:.3f} ms")
    print(f"checkpoint pretrained: {checkpoint.get('pretrained')}")


if __name__ == "__main__":
    main()

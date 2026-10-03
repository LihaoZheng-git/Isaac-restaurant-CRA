"""Evaluate the frozen mass model on every object in the existing ABO test split."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from robot.perception.abo_visual_mass import read_abo_subset_manifest
from robot.perception.relative_heaviness import HeavinessPrediction, evaluate_relative_heaviness
from robot.perception.visual_mass_model import AboVisualMassDataset, load_visual_mass_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--minimum-mass-ratio", type=float, default=3.0)
    args = parser.parse_args()
    candidates = tuple(item for item in read_abo_subset_manifest(args.manifest) if item.split == "test")
    model, _ = load_visual_mass_checkpoint(args.checkpoint, map_location=args.device)
    model.eval()
    rows = []
    with torch.inference_mode():
        for images, dimensions, log_mass, ids, classes in DataLoader(
            AboVisualMassDataset(candidates, args.image_root, training=False), batch_size=32,
        ):
            estimates = model.predict_mass_kg(images.to(args.device), dimensions.to(args.device)).cpu()
            for object_id, label, target, estimate in zip(ids, classes, log_mass.exp(), estimates):
                rows.append(HeavinessPrediction(object_id, label, float(target), float(estimate)))
    report = evaluate_relative_heaviness(tuple(rows), minimum_mass_ratio=args.minimum_mass_ratio)
    report.update({
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "input_domain": "ABO catalog RGB and external catalog dimensions",
        "label_role": "held-out offline catalog metadata",
        "training_performed": False,
        "predictions": [asdict(item) for item in rows],
        "limitations": ["not robot partial-envelope validation", "not accurate kilograms", "no live confidence cap change"],
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "predictions"}, indent=2))


if __name__ == "__main__":
    main()

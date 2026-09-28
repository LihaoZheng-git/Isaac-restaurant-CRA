"""Train a local fixed-label hot-soup detector after dataset readiness passes."""

from __future__ import annotations

import argparse
from importlib import metadata
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot.perception.hot_soup_dataset import (
    export_yolo_hot_soup_dataset,
    read_hot_soup_manifest,
)
from robot.perception.hot_soup_training import (
    assess_hot_soup_training_readiness,
    write_hot_soup_training_provenance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", default="hot_soup_yolo11n")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.base_model.is_file():
        raise FileNotFoundError(args.base_model)
    samples = read_hot_soup_manifest(args.manifest)
    readiness = assess_hot_soup_training_readiness(samples, args.dataset_root)
    print(readiness)
    if not readiness.ready:
        raise RuntimeError(
            "hot-soup training blocked: " + "; ".join(readiness.blockers)
        )
    export_root = args.project / f"{args.name}_dataset"
    data = export_yolo_hot_soup_dataset(samples, args.dataset_root, export_root)
    from ultralytics import YOLO

    model = YOLO(str(args.base_model))
    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=False,
        pretrained=True,
        plots=True,
        verbose=True,
    )
    checkpoint = args.project / args.name / "weights" / "best.pt"
    package = metadata.metadata("ultralytics")
    write_hot_soup_training_provenance(
        args.project / args.name / "training_provenance.json",
        readiness=readiness,
        manifest_path=args.manifest,
        base_checkpoint_path=args.base_model,
        trained_checkpoint_path=checkpoint,
        package_version=metadata.version("ultralytics"),
        package_license=package.get("License", "unavailable"),
        training_arguments={
            "epochs": args.epochs,
            "image_size": args.image_size,
            "batch": args.batch,
            "device": args.device,
            "workers": args.workers,
            "seed": args.seed,
        },
    )


if __name__ == "__main__":
    main()

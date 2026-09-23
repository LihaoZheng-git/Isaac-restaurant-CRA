"""Train a lightweight fixed-label contact-geometry segmentation baseline."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", default="yolo11n_seg_paco_geometry")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"model weights not found: {args.model}")
    if not args.data.is_file():
        raise FileNotFoundError(f"dataset config not found: {args.data}")
    from ultralytics import YOLO

    model = YOLO(str(args.model))
    model.train(
        data=str(args.data),
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


if __name__ == "__main__":
    main()

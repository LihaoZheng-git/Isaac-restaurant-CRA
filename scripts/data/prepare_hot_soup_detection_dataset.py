"""Validate an annotated hot-soup manifest and export YOLO detection data."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot.perception.hot_soup_dataset import (
    export_yolo_hot_soup_dataset,
    read_hot_soup_manifest,
    validate_hot_soup_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = read_hot_soup_manifest(args.manifest)
    report = validate_hot_soup_dataset(samples, args.dataset_root)
    print(report)
    if not report.valid:
        raise RuntimeError("hot-soup dataset validation failed")
    print(export_yolo_hot_soup_dataset(samples, args.dataset_root, args.output_root))


if __name__ == "__main__":
    main()


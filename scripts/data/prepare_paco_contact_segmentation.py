"""Convert PACO hazard manifests into local YOLO segmentation labels."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot.perception.contact_segmentation_dataset import (
    write_yolo_contact_split,
    write_yolo_dataset_yaml,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = []
    for manifest_path in sorted(args.manifest_dir.glob("*.json")):
        split = manifest_path.stem
        report = write_yolo_contact_split(manifest_path, args.dataset_root)
        splits.append(split)
        print(f"{split}: {report}")
    if not splits:
        raise FileNotFoundError(f"no manifests found in {args.manifest_dir}")
    print(f"dataset config: {write_yolo_dataset_yaml(args.dataset_root, splits)}")


if __name__ == "__main__":
    main()

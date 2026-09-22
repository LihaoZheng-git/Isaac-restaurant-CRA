"""Inventory fixed-label PACO knife-blade masks without loading full JSON."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot.perception.paco_contact_dataset import (
    inventory_paco_contact_parts,
    write_paco_contact_inventory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = tuple(sorted(args.annotation_dir.glob("paco_lvis_v1_*.json")))
    if not paths:
        raise FileNotFoundError(f"no PACO-LVIS annotations in {args.annotation_dir}")
    report = inventory_paco_contact_parts(paths)
    write_paco_contact_inventory(report, args.output)
    print(f"knife-blade images across splits: {report.total_images}")
    print(f"knife-blade instances across splits: {report.total_instances}")
    for item in report.split_counts:
        print(f"{item.split}: {item.image_count} images, {item.instance_count} masks")
    print(f"annotation license status: {report.annotation_license_status}")
    print("runtime model status: unavailable")


if __name__ == "__main__":
    main()

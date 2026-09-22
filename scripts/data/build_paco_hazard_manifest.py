"""Build bounded PACO supervision manifests for general contact geometry."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot.perception.paco_contact_dataset import (
    build_paco_hazard_manifest,
    write_paco_hazard_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_paco_hazard_manifest(args.annotation)
    write_paco_hazard_manifest(manifest, args.output)
    print(f"split: {manifest['split']}")
    print(f"selected images: {manifest['image_count']}")
    print(f"geometry-mask records: {manifest['record_count']}")
    print("sharpness/material/contact severity/route orientation: unavailable")


if __name__ == "__main__":
    main()

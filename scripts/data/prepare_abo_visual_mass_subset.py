"""Create a deterministic restaurant-object subset from local ABO metadata."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot.perception.abo_visual_mass import (
    DEFAULT_RESTAURANT_PRODUCT_TYPES,
    iter_abo_listings,
    load_abo_image_paths,
    select_abo_mass_subset,
    write_abo_selection_report,
    write_abo_subset_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--listings-dir",
        type=Path,
        required=True,
        help="Directory containing listings_*.json.gz from abo-listings.tar",
    )
    parser.add_argument(
        "--image-metadata",
        type=Path,
        required=True,
        help="ABO images/metadata/images.csv.gz",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-per-class", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    listing_paths = tuple(sorted(args.listings_dir.glob("listings_*.json.gz")))
    if not listing_paths:
        raise FileNotFoundError(f"no ABO listing shards found in {args.listings_dir}")

    # The compact image index fits in memory; rich listing records stream from
    # disk so the 147k-product metadata corpus is never retained as one tuple.
    image_paths = load_abo_image_paths(args.image_metadata)
    candidates, report = select_abo_mass_subset(
        iter_abo_listings(listing_paths),
        image_paths,
        categories=DEFAULT_RESTAURANT_PRODUCT_TYPES,
        max_objects_per_class=args.max_per_class,
    )
    write_abo_subset_manifest(candidates, args.output_dir / "manifest.jsonl")
    write_abo_selection_report(report, args.output_dir / "selection_report.json")

    print(f"selected {report.selected_records} ABO objects")
    print(f"classes: {dict(report.class_counts)}")
    print(f"splits: {dict(report.split_counts)}")
    print("labels: external catalog metadata; runtime evidence: unavailable")


if __name__ == "__main__":
    main()

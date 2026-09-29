"""Audit and import an extracted external YOLO thermal dataset for relabeling."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot.perception.external_thermal_dataset import (
    import_external_thermal_images,
    write_external_thermal_audit,
    write_external_thermal_collection,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-reference", required=True)
    parser.add_argument("--usage-rights", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit, items = import_external_thermal_images(
        args.export_root,
        args.dataset_root,
        source_reference=args.source_reference,
        usage_rights=args.usage_rights,
    )
    audit_path = args.dataset_root / "external-source-audit.json"
    manifest_path = args.dataset_root / "external-relabel-collection.jsonl"
    write_external_thermal_audit(audit, audit_path)
    write_external_thermal_collection(items, manifest_path)
    print(
        f"images={audit.image_count}; groups={audit.source_group_count}; "
        f"published_split_leaks={len(audit.published_split_leak_groups)}; "
        f"plume_without_source={audit.plume_without_source_count}"
    )
    print("imported_split=train; annotations=pending_project_relabel")
    print("external data is not eligible for project validation or test")
    print(f"audit={audit_path}; collection={manifest_path}")


if __name__ == "__main__":
    main()

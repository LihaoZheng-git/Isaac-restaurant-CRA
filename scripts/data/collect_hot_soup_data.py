"""Collect owned stationary images or video frames for hot-soup annotation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot.perception.hot_soup_collection import (
    collect_hot_soup_images,
    extract_hot_soup_video_frames,
    write_hot_soup_collection_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--images", type=Path, nargs="+")
    source.add_argument("--video", type=Path)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--capture-group", required=True)
    parser.add_argument("--source-reference", required=True)
    parser.add_argument("--usage-rights", required=True)
    parser.add_argument(
        "--domain", choices=("real_user_owned", "isaac_synthetic"), default="real_user_owned"
    )
    parser.add_argument("--every-n-frames", type=int, default=30)
    parser.add_argument("--max-frames", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.video is not None:
        if args.domain != "real_user_owned":
            raise ValueError("video extraction currently accepts real user-owned clips")
        items = extract_hot_soup_video_frames(
            args.video,
            args.dataset_root,
            capture_group_id=args.capture_group,
            source_reference=args.source_reference,
            usage_rights=args.usage_rights,
            every_n_frames=args.every_n_frames,
            max_frames=args.max_frames,
        )
    else:
        source_kind = "real_image" if args.domain == "real_user_owned" else "isaac_render"
        items = collect_hot_soup_images(
            args.images,
            args.dataset_root,
            capture_group_id=args.capture_group,
            domain=args.domain,
            source_kind=source_kind,
            source_reference=args.source_reference,
            usage_rights=args.usage_rights,
        )
    manifest = args.dataset_root / f"collection-{args.capture_group}.jsonl"
    write_hot_soup_collection_manifest(items, manifest)
    print(f"collected={len(items)}; annotations=pending; manifest={manifest}")


if __name__ == "__main__":
    main()


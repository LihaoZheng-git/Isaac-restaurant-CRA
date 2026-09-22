"""Download only images named by an ABO visual-mass subset manifest."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot.perception.abo_visual_mass import read_abo_subset_manifest


ABO_SMALL_IMAGE_BASE_URL = (
    "https://amazon-berkeley-objects.s3.amazonaws.com/images/small"
)


@dataclass(frozen=True)
class DownloadReport:
    requested: int
    downloaded: int
    reused: int
    failed: tuple[tuple[str, str], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    return parser.parse_args()


def _download(url: str, target: Path, timeout_seconds: float) -> bool:
    if target.is_file() and target.stat().st_size > 0:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    try:
        with urlopen(url, timeout=timeout_seconds) as response:
            temporary.write_bytes(response.read())
        if temporary.stat().st_size == 0:
            raise ValueError("empty response")
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return True


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    candidates = read_abo_subset_manifest(args.manifest)
    downloaded = 0
    reused = 0
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for item in candidates:
            target = args.image_root / item.image_path
            url = f"{ABO_SMALL_IMAGE_BASE_URL}/{item.image_path}"
            futures[executor.submit(_download, url, target, args.timeout_seconds)] = (
                item.object_id,
                target,
            )
        for future in as_completed(futures):
            object_id, target = futures[future]
            try:
                if future.result():
                    downloaded += 1
                else:
                    reused += 1
            except Exception as error:
                failures.append((object_id, f"{type(error).__name__}: {error}"))
                if target.exists():
                    target.unlink()

    report = DownloadReport(
        requested=len(candidates),
        downloaded=downloaded,
        reused=reused,
        failed=tuple(sorted(failures)),
    )
    report_path = args.image_root.parent / "download_report.json"
    report_path.write_text(
        json.dumps(asdict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"requested {report.requested}; downloaded {downloaded}; "
        f"reused {reused}; failed {len(failures)}"
    )
    if failures:
        raise RuntimeError(f"{len(failures)} image downloads failed; see {report_path}")


if __name__ == "__main__":
    main()

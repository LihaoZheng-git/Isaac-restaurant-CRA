"""Download only images referenced by local PACO contact manifests."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class DownloadReport:
    requested: int
    downloaded: int
    existing: int
    failures: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--timeout-seconds", type=float, default=30)
    return parser.parse_args()


def download(url: str, target: Path, timeout_seconds: float) -> bool:
    if target.exists() and target.stat().st_size > 0:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    if not url.startswith(("http://images.cocodataset.org/", "https://images.cocodataset.org/")):
        raise ValueError("image URL is not on the expected COCO image host")
    request = Request(url, headers={"User-Agent": "CRA-research/1"})
    with urlopen(request, timeout=timeout_seconds) as response:
        payload = response.read()
    if not payload:
        raise ValueError("downloaded image is empty")
    temporary = target.with_suffix(target.suffix + ".partial")
    temporary.write_bytes(payload)
    temporary.replace(target)
    return True


def main() -> None:
    args = parse_args()
    jobs = {}
    for manifest_path in sorted(args.manifest_dir.glob("*.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        split = str(manifest["split"])
        for record in manifest["records"]:
            image = record["image"]
            image_id = int(record["image_id"])
            jobs[(split, image_id)] = (
                str(image["coco_url"]),
                args.dataset_root / "images" / split / f"{image_id:012d}.jpg",
            )
    downloaded = 0
    existing = 0
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download, url, target, args.timeout_seconds): key
            for key, (url, target) in jobs.items()
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                if future.result():
                    downloaded += 1
                else:
                    existing += 1
            except Exception as error:
                failures.append(f"{key}: {type(error).__name__}: {error}")
    report = DownloadReport(len(jobs), downloaded, existing, tuple(sorted(failures)))
    report_path = args.dataset_root / "download_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(asdict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"requested={report.requested} downloaded={downloaded} existing={existing}")
    print(f"failures={len(failures)} report={report_path}")
    if failures:
        raise RuntimeError("one or more PACO contact image downloads failed")


if __name__ == "__main__":
    main()

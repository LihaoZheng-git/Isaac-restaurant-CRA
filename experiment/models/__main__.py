"""Download the two public checkpoints used in the recorded CRA demo."""

import argparse
from http.client import HTTPException
import json
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

from experiment.models import MODEL_DIR, MODEL_FILES, digest


SOURCES = {
    "sam": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
    "person": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n-seg.pt",
}


def download(url, target, expected):
    if target.exists():
        if digest(target) != expected:
            raise ValueError(f"{target} differs from the recorded checkpoint. Move it aside to download the original.")
        print(f"Already verified: {target.name}")
        return

    print(f"Downloading {target.name} ...", flush=True)
    with NamedTemporaryFile(dir=target.parent, suffix=".part", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with urlopen(url, timeout=60) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        if digest(temporary) != expected:
            raise ValueError(f"Downloaded {target.name} differs from the recorded SHA-256.")
        temporary.replace(target)
        print(f"Verified: {target.name}")
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=MODEL_DIR, help="destination (default: experiment/models)")
    args = parser.parse_args()
    expected = json.loads((MODEL_DIR.parent / "recorded-run.json").read_text(encoding="utf-8"))["models_sha256"]
    try:
        args.model_root.mkdir(parents=True, exist_ok=True)
        for key, url in SOURCES.items():
            _, name = MODEL_FILES[key]
            download(url, args.model_root / name, expected[key])
    except (OSError, ValueError, HTTPException) as error:
        parser.exit(1, f"Checkpoint setup failed: {error}\n")
    print("For the two CRA-trained checkpoints, see experiment/models/README.md.")


if __name__ == "__main__":
    main()

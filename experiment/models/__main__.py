"""Download public checkpoints or check the four models used in the restaurant demo."""

import argparse
from http.client import HTTPException
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

from experiment.models import (
    MODEL_DIR, MODEL_FILES, checkpoint_path, digest, recorded_model_hashes,
)


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
    parser.add_argument("--check", action="store_true", help="verify all four local checkpoints without downloading")
    args = parser.parse_args()
    expected = recorded_model_hashes()
    try:
        if not args.check:
            args.model_root.mkdir(parents=True, exist_ok=True)
            for key, url in SOURCES.items():
                _, name = MODEL_FILES[key]
                target = args.model_root / name
                if checkpoint_path(args.model_root, key) == target.resolve():
                    download(url, target, expected[key])
        missing = []
        mismatches = []
        for key in MODEL_FILES:
            target = checkpoint_path(args.model_root, key)
            if not target.is_file():
                missing.append(str(target))
            elif digest(target) != expected[key]:
                mismatches.append(str(target))
        if mismatches:
            raise ValueError("SHA-256 differs from the recorded demo:\n  " + "\n  ".join(mismatches))
        if missing:
            message = "Still needed for the full mission:\n  " + "\n  ".join(missing)
            message += "\nSee experiment/models/README.md for checkpoint sources."
            if args.check:
                parser.exit(1, message + "\n")
            print(message)
        else:
            print("All four restaurant mission checkpoints match the recorded demo.")
    except (OSError, ValueError, HTTPException) as error:
        parser.exit(1, f"Checkpoint setup failed: {error}\n")


if __name__ == "__main__":
    main()

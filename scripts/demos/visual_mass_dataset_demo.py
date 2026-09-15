"""Stationary offline visual-mass dataset contract diagnostic."""

from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot.perception.visual_mass_collection import (
    VisualMassCapture,
    persist_visual_mass_capture,
)
from robot.perception.visual_mass_dataset import (
    deterministic_object_split,
    validate_visual_mass_dataset,
    write_visual_mass_manifest,
)


def main() -> None:
    print("STATIONARY DATASET DIAGNOSTIC: synthetic image; no robot or model")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        capture = VisualMassCapture(
            "chair-a-view-001",
            "chair-a",
            "chair",
            "controlled_restaurant",
            deterministic_object_split("chair-a"),
            8.0,
            "isaac_authored_mass",
            (0.5, 0.5, 1.0),
            "front_hawk_left",
            17,
            np.zeros((8, 8, 3), dtype=np.uint8),
            np.ones((8, 8), dtype=np.bool_),
        )
        sample = persist_visual_mass_capture(capture, root)
        write_visual_mass_manifest([sample], root / "manifest.jsonl")
        report = validate_visual_mass_dataset([sample], root)
        print(
            f"samples={report.sample_count}, objects={report.object_count}, valid={report.valid}"
        )
        if not report.valid:
            raise RuntimeError("visual mass dataset diagnostic failed")
    print("PASS: oracle label stayed in offline dataset metadata only")


if __name__ == "__main__":
    main()

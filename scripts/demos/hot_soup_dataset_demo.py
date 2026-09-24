"""Stationary synthetic diagnostic for the hot-soup dataset pipeline."""

from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot.perception.hot_soup_dataset import (
    HotSoupBox,
    HotSoupSample,
    export_yolo_hot_soup_dataset,
    sha256_file,
    validate_hot_soup_dataset,
)


def main() -> None:
    print("STATIONARY DATASET DIAGNOSTIC: synthetic labels; no model or robot")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        image_path = root / "images" / "synthetic.png"
        image_path.parent.mkdir(parents=True)
        assert cv2.imwrite(str(image_path), np.zeros((24, 32, 3), dtype=np.uint8))
        boxes = (
            HotSoupBox(
                "visible_hot_soup_plume", 0.4, 0.1, 0.6, 0.5, "isaac_authored_oracle"
            ),
            HotSoupBox(
                "open_container", 0.2, 0.45, 0.8, 0.9, "isaac_authored_oracle"
            ),
        )
        sample = HotSoupSample(
            sample_id="isaac-plume-1",
            capture_group_id="isaac-scene-1",
            split="train",
            image_path="images/synthetic.png",
            image_sha256=sha256_file(image_path),
            width=32,
            height=24,
            domain="isaac_synthetic",
            source_reference="generated stationary USD plume scene",
            usage_rights="project-authored synthetic fixture",
            plume_state="present",
            container_state="present",
            boxes=boxes,
        )
        report = validate_hot_soup_dataset([sample], root)
        if not report.valid:
            raise RuntimeError(report)
        config = export_yolo_hot_soup_dataset([sample], root, root / "yolo")
        print(f"valid={report.valid}; yolo_config={config}")
    print("PASS: synthetic oracle labels remained offline and explicitly labeled")


if __name__ == "__main__":
    main()

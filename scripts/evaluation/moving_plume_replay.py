"""Replay recorded stationary RGB windows through registered plume estimation.

Optional pixel shifts are synthetic registration tests, not moving-camera sensor
captures. Historical frozen LiDAR timestamps are retained with their declared
0.5 s stationary replay gate; the live default remains 0.15 s.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import cv2
import numpy as np
from PIL import Image
from robot.perception.camera_lidar_fusion import TrackSemanticAssociation
from robot.perception.moving_camera_plume import (
    MovingCameraPlumeConfig,
    MovingCameraObjectPlumeEstimator,
    TrackedRgbObjectFrame,
)
from robot.perception.semantic_detection import (
    SemanticDetection,
    SemanticDetectionFrame,
    PixelBoundingBox,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pixel-shift", type=float, nargs=2, default=(0.0, 0.0))
    args = parser.parse_args()
    vision = json.loads((args.capture / "vision.json").read_text(encoding="utf-8"))
    capture = json.loads((args.capture / "stationary.json").read_text(encoding="utf-8"))
    linked = TrackSemanticAssociation(**vision["target_association"])
    reference_id = vision["camera_info"]["read_index"]
    reference_time = vision["camera_info"]["simulation_time_seconds"]
    track_time = capture["sensor_attributes"]["geometry"]["width_m"][
        "timestamp_seconds"
    ]
    reference = np.asarray(Image.open(args.capture / "camera.png").convert("RGB"))
    window = np.load(args.capture / "plume-window.npz")

    def observed(image, index, at):
        mask = np.load(args.capture / f"sam-mask-frame-{index}.npy")
        dx, dy = np.array(args.pixel_shift) * (index - reference_id)
        height, width = image.shape[:2]
        transform = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], np.float32)
        image = cv2.warpAffine(
            image, transform, (width, height), borderMode=cv2.BORDER_REFLECT
        )
        mask = cv2.warpAffine(mask.astype(np.uint8), transform, (width, height)).astype(
            bool
        )
        yy, xx = np.where(mask)
        detection = SemanticDetection(
            0,
            "unknown",
            0.75,
            PixelBoundingBox(
                float(xx.min()),
                float(yy.min()),
                float(xx.max() + 1),
                float(yy.max() + 1),
            ),
            mask,
        )
        frame = SemanticDetectionFrame(
            index, float(at), height, width, (detection,), "recorded_sam_instance", 0.0
        )
        return TrackedRgbObjectFrame(image, frame, linked, track_time)

    estimator = MovingCameraObjectPlumeEstimator(
        MovingCameraPlumeConfig(maximum_track_age_seconds=0.5)
    )
    estimator.set_reference(
        observed(reference, reference_id, reference_time),
        clean_reference_validated=True,
    )
    rows = []
    for offset, (image, timestamp) in enumerate(
        zip(window["rgb"], window["timestamps"])
    ):
        result = estimator.update(
            observed(image, reference_id + 1 + offset, timestamp),
            negative_validation_scope="recorded controlled stationary isolated-object clean-reference fixture",
        )
        rows.append(
            {
                "thermal": asdict(result.thermal),
                "rejection_reasons": result.rejection_reasons,
                "registration": estimator.last_diagnostics.get(linked.track_id, {}),
            }
        )
    payload = {
        "scope": "stationary sensor replay with synthetic pixel shifts"
        if any(args.pixel_shift)
        else "stationary sensor replay only",
        "source_capture": str(args.capture.resolve()),
        "pixel_shift_per_frame": list(args.pixel_shift),
        "maximum_track_age_seconds": 0.5,
        "reference_timestamp": reference_time,
        "track_timestamp": track_time,
        "rows": rows,
        "final_cue_available": result.thermal.heat_source_likelihood.is_available,
        "final_cue": result.thermal.heat_source_likelihood.value,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: payload[key] for key in ("scope", "final_cue_available", "final_cue")}
        )
    )


if __name__ == "__main__":
    main()

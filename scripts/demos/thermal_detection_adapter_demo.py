"""Stationary fixed-label thermal detection adapter diagnostic."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot.perception.semantic_detection import (
    PixelBoundingBox,
    SemanticDetection,
    SemanticDetectionFrame,
)
from robot.perception.thermal_cues import estimate_thermal_cues
from robot.perception.thermal_detection_adapter import (
    thermal_cue_frame_from_detections,
)


def frame(frame_id: int, timestamp: float) -> SemanticDetectionFrame:
    mask = np.ones((8, 8), dtype=np.bool_)
    detections = (
        SemanticDetection(
            0, "fire", 0.8, PixelBoundingBox(0, 0, 8, 8), mask
        ),
        SemanticDetection(
            1, "smoke", 0.9, PixelBoundingBox(0, 0, 8, 8), mask
        ),
    )
    return SemanticDetectionFrame(
        frame_id, timestamp, 8, 8, detections, "fixed_label_fixture", 0.01
    )


def main() -> None:
    cues = tuple(
        thermal_cue_frame_from_detections(frame(index, timestamp))
        for index, timestamp in enumerate((1.0, 1.2, 1.4))
    )
    result = estimate_thermal_cues(cues, timestamp_seconds=1.5)
    print("STATIONARY THERMAL DETECTION ADAPTER: fixture detections, no robot motion")
    print(f"flame={result.heat_source_likelihood.value:.2f}")
    print(f"smoke={cues[-1].smoke_likelihood:.2f}; steam=unavailable")
    print("temperature/container/spill/track interaction=unavailable")
    assert not result.steam_likelihood.is_available
    assert not result.open_container_likelihood.is_available
    print("PASS: smoke was retained but not relabeled as steam")


if __name__ == "__main__":
    main()

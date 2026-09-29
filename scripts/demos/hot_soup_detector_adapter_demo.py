"""Stationary box-association diagnostic; no trained model or robot motion."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot.perception.hot_soup_detector import (
    HotSoupDetection,
    HotSoupDetectionFrame,
    estimate_hot_soup_from_detections,
)
from robot.perception.semantic_detection import PixelBoundingBox


def main() -> None:
    print("STATIONARY DETECTOR-ADAPTER DIAGNOSTIC: oracle boxes; no model or robot")
    frame = HotSoupDetectionFrame(
        source_frame_id=1,
        source_timestamp_seconds=5.0,
        image_height=480,
        image_width=640,
        detections=(
            HotSoupDetection(
                0,
                "visible_hot_soup_plume",
                0.88,
                PixelBoundingBox(270, 80, 370, 245),
            ),
            HotSoupDetection(
                1,
                "open_container",
                0.84,
                PixelBoundingBox(220, 235, 420, 390),
            ),
        ),
        model_name="oracle-box-fixture-no-checkpoint",
        inference_wall_seconds=0,
    )
    result = estimate_hot_soup_from_detections(frame, timestamp_seconds=5.1)
    thermal = result.cue_result.thermal
    print(f"pair_count={result.candidate_pair_count}")
    print(f"spatial_score={result.spatial_association_score:.3f}")
    print(f"heat_proxy={thermal.heat_source_likelihood.value:.3f}")
    print("temperature=unavailable; steam=unavailable; interaction=unavailable")
    if thermal.source_type.value != "contextual_visible_plume_proxy":
        raise RuntimeError("stationary detector adapter diagnostic failed")
    print("PASS: oracle boxes exercised context geometry without model claims")


if __name__ == "__main__":
    main()

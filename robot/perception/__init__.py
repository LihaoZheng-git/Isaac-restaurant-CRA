"""Camera-based semantic perception interfaces and adapters."""

from robot.perception.semantic_detection import (
    PixelBoundingBox,
    SemanticDetection,
    SemanticDetectionFrame,
)
from robot.perception.yolo_detector import (
    YoloSegmentationConfig,
    YoloSegmentationDetector,
)


__all__ = [
    "PixelBoundingBox",
    "SemanticDetection",
    "SemanticDetectionFrame",
    "YoloSegmentationConfig",
    "YoloSegmentationDetector",
]

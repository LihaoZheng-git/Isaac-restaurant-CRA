"""Camera-based semantic perception interfaces and adapters."""

from robot.perception.physical_attributes import (
    PHYSICAL_ATTRIBUTE_SCHEMA_VERSION,
    CategoricalEstimate,
    ContactAttributes,
    EstimateStatus,
    GeometryAttributes,
    MassAttributes,
    MotionAttributes,
    PhysicalAttributes,
    ScalarEstimate,
    StabilityAttributes,
    ThermalAttributes,
)
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
    "PHYSICAL_ATTRIBUTE_SCHEMA_VERSION",
    "CategoricalEstimate",
    "ContactAttributes",
    "EstimateStatus",
    "GeometryAttributes",
    "MassAttributes",
    "MotionAttributes",
    "PhysicalAttributes",
    "PixelBoundingBox",
    "ScalarEstimate",
    "SemanticDetection",
    "SemanticDetectionFrame",
    "StabilityAttributes",
    "ThermalAttributes",
    "YoloSegmentationConfig",
    "YoloSegmentationDetector",
]

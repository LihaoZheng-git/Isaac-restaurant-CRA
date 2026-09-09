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
from robot.perception.physical_geometry import (
    GEOMETRY_SOURCE,
    LidarGeometryConfig,
    estimate_track_geometry,
)
from robot.perception.motion_attributes import (
    MOTION_SOURCE,
    ROUTE_SOURCE,
    MotionAttributeConfig,
    RouteMotionContext,
    estimate_track_motion,
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
    "GEOMETRY_SOURCE",
    "LidarGeometryConfig",
    "MassAttributes",
    "MOTION_SOURCE",
    "MotionAttributeConfig",
    "MotionAttributes",
    "PhysicalAttributes",
    "PixelBoundingBox",
    "ROUTE_SOURCE",
    "RouteMotionContext",
    "ScalarEstimate",
    "SemanticDetection",
    "SemanticDetectionFrame",
    "StabilityAttributes",
    "ThermalAttributes",
    "YoloSegmentationConfig",
    "YoloSegmentationDetector",
    "estimate_track_geometry",
    "estimate_track_motion",
]

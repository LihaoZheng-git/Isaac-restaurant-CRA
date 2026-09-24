"""Camera-based semantic perception interfaces and adapters."""

from robot.perception.fall_height import (
    FALL_HEIGHT_SOURCE as FALL_HEIGHT_SOURCE,
    FallSurfaceEvidence as FallSurfaceEvidence,
    estimate_fall_height as estimate_fall_height,
)
from robot.perception.live_physical_evidence import (
    LivePhysicalEvidenceContext as LivePhysicalEvidenceContext,
    build_live_physical_attributes as build_live_physical_attributes,
    build_tracking_physical_attributes as build_tracking_physical_attributes,
)
from robot.perception.hot_soup_cues import (
    HOT_SOUP_SOURCE,
    HotSoupCueConfig,
    HotSoupCueResult,
    estimate_hot_soup_cues,
)
from robot.perception.lidar_fall_surfaces import (
    LIDAR_SURFACE_SOURCE,
    LidarFallSurfaceConfig,
    LidarFallSurfaceResult,
    estimate_lidar_fall_surfaces,
)

from robot.perception.mass_estimation import (
    MASS_FUSION_SOURCE,
    MASS_PRIOR_SCHEMA_VERSION,
    MASS_PRIOR_SOURCE,
    VOLUME_DENSITY_MASS_SOURCE,
    VOLUME_SOURCE,
    BoxVolumeConfig,
    ClassMassPrior,
    ClassMassPriorTable,
    estimate_box_volume,
    estimate_mass_from_volume_density,
    estimate_track_mass,
    fuse_mass_estimates,
)
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
    combine_estimate_statuses,
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
    "MASS_FUSION_SOURCE",
    "MASS_PRIOR_SCHEMA_VERSION",
    "MASS_PRIOR_SOURCE",
    "PHYSICAL_ATTRIBUTE_SCHEMA_VERSION",
    "VOLUME_DENSITY_MASS_SOURCE",
    "VOLUME_SOURCE",
    "BoxVolumeConfig",
    "CategoricalEstimate",
    "ClassMassPrior",
    "ClassMassPriorTable",
    "ContactAttributes",
    "EstimateStatus",
    "GeometryAttributes",
    "GEOMETRY_SOURCE",
    "HOT_SOUP_SOURCE",
    "HotSoupCueConfig",
    "HotSoupCueResult",
    "LidarGeometryConfig",
    "LIDAR_SURFACE_SOURCE",
    "LidarFallSurfaceConfig",
    "LidarFallSurfaceResult",
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
    "build_live_physical_attributes",
    "build_tracking_physical_attributes",
    "combine_estimate_statuses",
    "estimate_box_volume",
    "estimate_hot_soup_cues",
    "estimate_mass_from_volume_density",
    "estimate_lidar_fall_surfaces",
    "estimate_track_geometry",
    "estimate_track_mass",
    "estimate_track_motion",
    "fuse_mass_estimates",
]

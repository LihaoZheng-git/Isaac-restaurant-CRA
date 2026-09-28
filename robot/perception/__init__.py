"""Camera-based semantic perception interfaces and adapters."""

from robot.perception.fall_height import (
    FALL_HEIGHT_SOURCE as FALL_HEIGHT_SOURCE,
    FallSurfaceEvidence as FallSurfaceEvidence,
    estimate_fall_height as estimate_fall_height,
)
from robot.perception.contact_detection_adapter import (
    ContactPartAssociationConfig,
    ContactPartAssociationDiagnostics,
    ContactPartAssociationResult,
    associate_contact_parts_to_objects,
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
from robot.perception.hot_soup_detector import (
    HOT_SOUP_DETECTOR_SOURCE,
    HotSoupAssociationConfig,
    HotSoupDetection,
    HotSoupDetectionCueResult,
    HotSoupDetectionFrame,
    HotSoupDetectorConfig,
    YoloHotSoupDetector,
    estimate_hot_soup_from_detections,
)
from robot.perception.hot_soup_evaluation import (
    HotSoupClassMetrics,
    HotSoupEvaluationReport,
    HotSoupPrediction,
    evaluate_hot_soup_predictions,
    predictions_from_hot_soup_frame,
)
from robot.perception.hot_soup_training import (
    HotSoupTrainingReadiness,
    HotSoupTrainingReadinessConfig,
    assess_hot_soup_training_readiness,
    write_hot_soup_training_provenance,
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
from robot.perception.specialist_track_evidence import (
    SpecialistAssociationFrame,
    SpecialistTrackDiagnostics,
    TrackSpecialistEvidence,
    build_track_specialist_evidence,
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
    "ContactPartAssociationConfig",
    "ContactPartAssociationDiagnostics",
    "ContactPartAssociationResult",
    "EstimateStatus",
    "GeometryAttributes",
    "GEOMETRY_SOURCE",
    "HOT_SOUP_SOURCE",
    "HOT_SOUP_DETECTOR_SOURCE",
    "HotSoupAssociationConfig",
    "HotSoupCueConfig",
    "HotSoupCueResult",
    "HotSoupDetection",
    "HotSoupDetectionCueResult",
    "HotSoupDetectionFrame",
    "HotSoupDetectorConfig",
    "HotSoupClassMetrics",
    "HotSoupEvaluationReport",
    "HotSoupPrediction",
    "HotSoupTrainingReadiness",
    "HotSoupTrainingReadinessConfig",
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
    "SpecialistAssociationFrame",
    "SpecialistTrackDiagnostics",
    "ThermalAttributes",
    "TrackSpecialistEvidence",
    "YoloSegmentationConfig",
    "YoloSegmentationDetector",
    "YoloHotSoupDetector",
    "build_live_physical_attributes",
    "build_tracking_physical_attributes",
    "build_track_specialist_evidence",
    "assess_hot_soup_training_readiness",
    "combine_estimate_statuses",
    "associate_contact_parts_to_objects",
    "estimate_box_volume",
    "estimate_hot_soup_cues",
    "estimate_hot_soup_from_detections",
    "evaluate_hot_soup_predictions",
    "estimate_mass_from_volume_density",
    "estimate_lidar_fall_surfaces",
    "estimate_track_geometry",
    "estimate_track_mass",
    "estimate_track_motion",
    "fuse_mass_estimates",
    "predictions_from_hot_soup_frame",
    "write_hot_soup_training_provenance",
]

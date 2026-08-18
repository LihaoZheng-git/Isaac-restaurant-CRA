"""Transactional camera--LiDAR semantics for the Safe-SAGE runtime.

The bridge joins four independently tested components without moving sensor
or model details into the controller:

1. the Safe-SAGE runtime updates one cloned LiDAR tracker;
2. camera masks label those exact track identifiers;
3. temporal memory confirms or conservatively persists the labels; and
4. the runtime builds and commits the complete safety field.

Tracker state, semantic memory, and the field advance together only after all
four stages succeed. Camera detections never create obstacle geometry.
"""

import copy
import math
from dataclasses import dataclass
from numbers import Real

import numpy as np

from robot.perception.camera_lidar_fusion import (
    CameraLidarFusionResult,
    CameraLidarSemanticFusion,
    FThetaCameraCalibration,
)
from robot.perception.semantic_detection import SemanticDetectionFrame
from robot.perception.semantic_track_memory import (
    SemanticTrackMemory,
    SemanticTrackMemoryResult,
)
from robot.safety.obstacle_tracking import ObstacleTrackingResult
from robot.safety.safe_sage_live_field import SafeSAGELiveFieldStatus
from robot.safety.safe_sage_runtime import (
    SafeSAGEPerceptionObservation,
    SafeSAGEPerceptionRuntime,
    SafeSAGERuntimeField,
)


def _finite_real(value: object, *, name: str) -> float:
    """Return one finite real number while rejecting booleans."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _finite_points_xyz(value: object, *, name: str) -> np.ndarray:
    """Return one immutable finite ``(N, 3)`` point array."""

    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3).")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values.")
    result = np.array(result, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


def _finite_transform(value: object, *, name: str) -> np.ndarray:
    """Return one immutable homogeneous transform."""

    result = np.asarray(value, dtype=np.float64)
    if result.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4).")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values.")
    if not np.allclose(result[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8):
        raise ValueError(f"{name} must be a homogeneous transform.")
    result = np.array(result, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class SafeSAGESemanticBridgeConfig:
    """Synchronization policy for camera and LiDAR observations."""

    maximum_sensor_time_offset_seconds: float = 0.12

    def __post_init__(self) -> None:
        maximum_offset = _finite_real(
            self.maximum_sensor_time_offset_seconds,
            name="maximum_sensor_time_offset_seconds",
        )
        if maximum_offset < 0.0:
            raise ValueError(
                "maximum_sensor_time_offset_seconds cannot be negative."
            )
        object.__setattr__(
            self,
            "maximum_sensor_time_offset_seconds",
            maximum_offset,
        )


@dataclass(frozen=True)
class SafeSAGECameraLidarObservation:
    """One synchronized input bundle for semantic field construction."""

    perception: SafeSAGEPerceptionObservation
    obstacle_points_source_xyz: np.ndarray
    detection_frame: SemanticDetectionFrame
    camera_from_source: np.ndarray
    calibration: FThetaCameraCalibration

    def __post_init__(self) -> None:
        if not isinstance(self.perception, SafeSAGEPerceptionObservation):
            raise TypeError(
                "perception must be a SafeSAGEPerceptionObservation."
            )
        if not isinstance(self.detection_frame, SemanticDetectionFrame):
            raise TypeError(
                "detection_frame must be a SemanticDetectionFrame."
            )
        if not isinstance(self.calibration, FThetaCameraCalibration):
            raise TypeError("calibration must be an FThetaCameraCalibration.")
        points = _finite_points_xyz(
            self.obstacle_points_source_xyz,
            name="obstacle_points_source_xyz",
        )
        expected_count = self.perception.obstacle_points_robot_xy.shape[0]
        if points.shape[0] != expected_count:
            raise ValueError(
                "obstacle_points_source_xyz must preserve one-to-one "
                "alignment with perception.obstacle_points_robot_xy."
            )
        object.__setattr__(self, "obstacle_points_source_xyz", points)
        object.__setattr__(
            self,
            "camera_from_source",
            _finite_transform(
                self.camera_from_source,
                name="camera_from_source",
            ),
        )

    @property
    def sensor_time_offset_seconds(self) -> float:
        """Return absolute camera--LiDAR timestamp separation."""

        return abs(
            self.detection_frame.source_timestamp_seconds
            - self.perception.sensor_timestamp_seconds
        )


@dataclass(frozen=True)
class SafeSAGESemanticPerceptionResult:
    """One accepted field plus visual and temporal diagnostics."""

    field_status: SafeSAGELiveFieldStatus[SafeSAGERuntimeField]
    fusion: CameraLidarFusionResult
    temporal_semantics: SemanticTrackMemoryResult
    sensor_time_offset_seconds: float

    @property
    def known_semantic_track_count(self) -> int:
        """Return the number of confirmed, non-unknown tracks."""

        return self.temporal_semantics.confirmed_track_count

    @property
    def unknown_semantic_track_count(self) -> int:
        """Return current tracks handled by conservative fallback policy."""

        return (
            len(self.temporal_semantics.tracks)
            - self.temporal_semantics.confirmed_track_count
        )


class SafeSAGESemanticPerceptionBridge:
    """Feed temporally fused camera labels into one Safe-SAGE runtime."""

    def __init__(
        self,
        *,
        perception_runtime: SafeSAGEPerceptionRuntime,
        fusion: CameraLidarSemanticFusion,
        semantic_memory: SemanticTrackMemory,
        config: SafeSAGESemanticBridgeConfig | None = None,
    ) -> None:
        if not isinstance(perception_runtime, SafeSAGEPerceptionRuntime):
            raise TypeError(
                "perception_runtime must be a SafeSAGEPerceptionRuntime."
            )
        if not isinstance(fusion, CameraLidarSemanticFusion):
            raise TypeError("fusion must be a CameraLidarSemanticFusion.")
        if not isinstance(semantic_memory, SemanticTrackMemory):
            raise TypeError(
                "semantic_memory must be a SemanticTrackMemory."
            )
        resolved_config = config or SafeSAGESemanticBridgeConfig()
        if not isinstance(resolved_config, SafeSAGESemanticBridgeConfig):
            raise TypeError("config must be a SafeSAGESemanticBridgeConfig.")
        self.perception_runtime = perception_runtime
        self.fusion = fusion
        self._semantic_memory = semantic_memory
        self.config = resolved_config
        self._last_result: SafeSAGESemanticPerceptionResult | None = None

    @property
    def initialized(self) -> bool:
        """Return whether the underlying field runtime is initialized."""

        return self.perception_runtime.initialized

    @property
    def semantic_memory_track_count(self) -> int:
        """Return semantic histories committed with the latest field."""

        return self._semantic_memory.active_track_count

    @property
    def last_result(self) -> SafeSAGESemanticPerceptionResult:
        """Return the latest accepted semantic field result."""

        if self._last_result is None:
            raise RuntimeError("No semantic Safe-SAGE field has been accepted.")
        return self._last_result

    def initialize(
        self,
        observation: SafeSAGECameraLidarObservation,
        *,
        anchor_position_world_xy: object | None = None,
        anchor_yaw: float | None = None,
    ) -> SafeSAGESemanticPerceptionResult:
        """Build and transactionally commit the first semantic field."""

        self._validate_observation(observation)
        if self.initialized:
            raise RuntimeError("The semantic perception bridge is initialized.")
        return self._resolve_and_commit(
            observation,
            initialize=True,
            anchor_position_world_xy=anchor_position_world_xy,
            anchor_yaw=anchor_yaw,
        )

    def update(
        self,
        observation: SafeSAGECameraLidarObservation,
    ) -> SafeSAGESemanticPerceptionResult:
        """Build and transactionally commit one newer semantic field."""

        self._validate_observation(observation)
        if not self.initialized:
            raise RuntimeError(
                "Initialize the semantic perception bridge before update."
            )
        return self._resolve_and_commit(observation, initialize=False)

    def _resolve_and_commit(
        self,
        observation: SafeSAGECameraLidarObservation,
        *,
        initialize: bool,
        anchor_position_world_xy: object | None = None,
        anchor_yaw: float | None = None,
    ) -> SafeSAGESemanticPerceptionResult:
        time_offset = observation.sensor_time_offset_seconds
        if (
            time_offset
            > self.config.maximum_sensor_time_offset_seconds
        ):
            raise RuntimeError(
                "Camera--LiDAR timestamp separation exceeds the semantic "
                "fusion limit: "
                f"{time_offset:.6f} s > "
                f"{self.config.maximum_sensor_time_offset_seconds:.6f} s."
            )

        candidate_memory = copy.deepcopy(self._semantic_memory)
        fusion_result: CameraLidarFusionResult | None = None
        temporal_result: SemanticTrackMemoryResult | None = None

        def resolve_labels(
            tracking_result: ObstacleTrackingResult,
        ):
            nonlocal fusion_result, temporal_result
            fusion_result = self.fusion.fuse(
                detection_frame=observation.detection_frame,
                obstacle_points_source_xyz=(
                    observation.obstacle_points_source_xyz
                ),
                tracking_result=tracking_result,
                camera_from_source=observation.camera_from_source,
                calibration=observation.calibration,
            )
            temporal_result = candidate_memory.update(
                fusion_result.semantic_labels,
                timestamp_seconds=(
                    observation.perception.sensor_timestamp_seconds
                ),
            )
            return temporal_result.semantic_labels

        if initialize:
            status = self.perception_runtime.initialize_from_tracking(
                observation.perception,
                resolve_labels,
                anchor_position_world_xy=anchor_position_world_xy,
                anchor_yaw=anchor_yaw,
            )
        else:
            status = self.perception_runtime.update_from_tracking(
                observation.perception,
                resolve_labels,
            )
        if fusion_result is None or temporal_result is None:
            raise RuntimeError(
                "Semantic label resolution did not produce diagnostics."
            )
        result = SafeSAGESemanticPerceptionResult(
            field_status=status,
            fusion=fusion_result,
            temporal_semantics=temporal_result,
            sensor_time_offset_seconds=time_offset,
        )
        self._semantic_memory = candidate_memory
        self._last_result = result
        return result

    def _validate_observation(
        self,
        observation: object,
    ) -> None:
        if not isinstance(observation, SafeSAGECameraLidarObservation):
            raise TypeError(
                "observation must be a SafeSAGECameraLidarObservation."
            )

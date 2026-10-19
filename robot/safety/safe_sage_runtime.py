"""Reusable live-perception and safety-control runtime for Safe-SAGE.

The numerical field solvers, MPC, and analytical CBF remain independent
modules.  This file owns the behavior that was previously repeated by Isaac
Sim demos:

* update a cloned LiDAR tracker and attach replaceable semantic labels;
* build a complete fixed-anchor field before committing any state;
* estimate the temporal derivative from consecutive accepted fields;
* enforce live-field freshness through the transactional buffer; and
* pass one field consistently to predictive MPC and analytical backup.

No Isaac Sim API is imported here.  Sensor adapters and wheel application
remain at the scenario boundary.
"""

import copy
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Real
from typing import Sequence

import numpy as np

from robot.control.unicycle_model import lookahead_state
from robot.safety.class_safety_policy import ClassSafetyPolicy
from robot.safety.cra_safe_sage_policy import CRASafeSAGEPolicy
from robot.safety.cra_track_policy import (
    CRARiskProfileProvider,
    CRATrackPolicyResult,
    resolve_cra_track_policy,
)
from robot.safety.local_occupancy_grid import (
    LocalOccupancyGridConfig,
    build_local_occupancy_grid,
)
from robot.safety.obstacle_tracking import (
    ObstacleTracker,
    ObstacleTrackingResult,
)
from robot.safety.safe_sage_field_pipeline import (
    SafeSAGEFieldPipelineConfig,
    SafeSAGEFieldPipelineResult,
    build_safe_sage_field_pipeline,
)
from robot.safety.safe_sage_guidance_policy import SafeSAGEGuidancePolicy
from robot.safety.safe_sage_live_field import (
    SafeSAGELiveFieldBuffer,
    SafeSAGELiveFieldConfig,
    SafeSAGELiveFieldSnapshot,
    SafeSAGELiveFieldStatus,
)
from robot.safety.safe_sage_temporal_derivative import (
    SafeSAGETemporalDerivativeEstimator,
    SafeSAGETemporalDerivativeResult,
    TemporalSafetyDerivativeInput,
)
from robot.safety.safe_sage_unicycle_cbf_filter import (
    SafeSAGEUnicycleCBFFilter,
    SafeSAGEUnicycleCBFResult,
)
from robot.safety.safe_sage_unicycle_mpc import (
    SafeSAGEUnicycleMPC,
    SafeSAGEUnicycleMPCResult,
)
from robot.safety.semantic_label_provider import (
    SemanticLabelProvider,
    SemanticLabelResult,
)


def _finite_real(value: object, *, name: str) -> float:
    """Return one finite real number while rejecting booleans."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


@dataclass
class SafeSAGEPlanExecutionLease:
    """Separate old optimizer candidates from freshly verified execution.

    The caller must validate the entire horizon from the current state against
    the current field with exact DTCBF/dense geometry checks before recording
    success. Source age is never renewed. This does not extend field/evidence
    freshness, and the analytical CBF still checks each executed physics step.
    """

    source_timestamp_seconds: float
    maximum_candidate_age_seconds: float = 0.2
    execution_interval_seconds: float = 0.2
    revalidation_interval_seconds: float = 0.15
    validated_timestamp_seconds: float | None = None
    validated_generation: int | None = None

    def __post_init__(self):
        for name in (
            "source_timestamp_seconds",
            "maximum_candidate_age_seconds",
            "execution_interval_seconds",
            "revalidation_interval_seconds",
        ):
            value = _finite_real(getattr(self, name), name=name)
            if value < 0 or (name != "source_timestamp_seconds" and value == 0):
                raise ValueError(f"{name} must be nonnegative/positive")
            setattr(self, name, value)
        if self.revalidation_interval_seconds >= self.execution_interval_seconds:
            raise ValueError(
                "revalidation must precede the verified first interval's expiry"
            )
        if (
            self.validated_timestamp_seconds is not None
            or self.validated_generation is not None
        ):
            raise ValueError("a new optimizer candidate is never prevalidated")

    def candidate_available(self, now):
        age = _finite_real(now, name="current time") - self.source_timestamp_seconds
        return 0 <= age <= self.maximum_candidate_age_seconds + 1e-9

    def needs_validation(self, now, generation):
        return (
            self.validated_timestamp_seconds is None
            or self.validated_generation != generation
            or _finite_real(now, name="current time") - self.validated_timestamp_seconds
            >= self.revalidation_interval_seconds
        )

    def record_validation(self, now, generation, *, accepted):
        if not isinstance(accepted, bool):
            raise TypeError("horizon validation must supply an explicit boolean result")
        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 0
        ):
            raise ValueError("field generation must be a nonnegative integer")
        self.validated_timestamp_seconds = None
        self.validated_generation = None
        if accepted and self.candidate_available(now):
            self.validated_timestamp_seconds = _finite_real(now, name="current time")
            self.validated_generation = generation

    def can_execute(self, now, generation, *, field_fresh):
        if not isinstance(field_fresh, bool):
            raise TypeError("field freshness must be an explicit boolean")
        return bool(
            field_fresh
            and self.candidate_available(now)
            and self.validated_timestamp_seconds is not None
            and self.validated_generation == generation
            and 0
            <= now - self.validated_timestamp_seconds
            <= self.execution_interval_seconds + 1e-9
        )


def _finite_vector(value: object, *, size: int, name: str) -> np.ndarray:
    """Return one immutable finite vector."""

    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},).")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values.")
    result = np.array(result, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


def _finite_points_xy(value: object, *, name: str) -> np.ndarray:
    """Return one immutable finite ``(N, 2)`` point array."""

    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2).")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values.")
    result = np.array(result, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


def _nonnegative_integer(value: object, *, name: str) -> int:
    """Return one nonnegative integer while rejecting booleans."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} cannot be negative.")
    return value


@dataclass(frozen=True)
class SafeSAGEPerceptionObservation:
    """One complete obstacle observation at a known robot pose."""

    sensor_frame_id: int
    sensor_timestamp_seconds: float
    simulation_time_seconds: float
    obstacle_points_robot_xy: np.ndarray
    robot_position_world_xy: np.ndarray
    robot_yaw: float

    def __post_init__(self) -> None:
        frame_id = _nonnegative_integer(
            self.sensor_frame_id,
            name="sensor_frame_id",
        )
        sensor_time = _finite_real(
            self.sensor_timestamp_seconds,
            name="sensor_timestamp_seconds",
        )
        simulation_time = _finite_real(
            self.simulation_time_seconds,
            name="simulation_time_seconds",
        )
        yaw = _finite_real(self.robot_yaw, name="robot_yaw")
        if sensor_time < 0.0:
            raise ValueError("sensor_timestamp_seconds cannot be negative.")
        if simulation_time < 0.0:
            raise ValueError("simulation_time_seconds cannot be negative.")
        object.__setattr__(self, "sensor_frame_id", frame_id)
        object.__setattr__(self, "sensor_timestamp_seconds", sensor_time)
        object.__setattr__(self, "simulation_time_seconds", simulation_time)
        object.__setattr__(
            self,
            "obstacle_points_robot_xy",
            _finite_points_xy(
                self.obstacle_points_robot_xy,
                name="obstacle_points_robot_xy",
            ),
        )
        object.__setattr__(
            self,
            "robot_position_world_xy",
            _finite_vector(
                self.robot_position_world_xy,
                size=2,
                name="robot_position_world_xy",
            ),
        )
        object.__setattr__(self, "robot_yaw", yaw)


@dataclass(frozen=True)
class SafeSAGERuntimeField:
    """One accepted semantic field and its perception diagnostics."""

    sensor_frame_id: int
    sensor_timestamp_seconds: float
    anchor_position_world_xy: np.ndarray
    anchor_yaw: float
    all_obstacle_point_count: int
    selected_obstacle_point_count: int
    tracking_result: ObstacleTrackingResult
    label_result: SemanticLabelResult
    pipeline: SafeSAGEFieldPipelineResult
    temporal_safety_derivative: TemporalSafetyDerivativeInput
    build_wall_seconds: float
    cra_policy_result: CRATrackPolicyResult | None = None

    def __post_init__(self) -> None:
        frame_id = _nonnegative_integer(
            self.sensor_frame_id,
            name="sensor_frame_id",
        )
        sensor_time = _finite_real(
            self.sensor_timestamp_seconds,
            name="sensor_timestamp_seconds",
        )
        all_count = _nonnegative_integer(
            self.all_obstacle_point_count,
            name="all_obstacle_point_count",
        )
        selected_count = _nonnegative_integer(
            self.selected_obstacle_point_count,
            name="selected_obstacle_point_count",
        )
        anchor_yaw = _finite_real(self.anchor_yaw, name="anchor_yaw")
        build_time = _finite_real(
            self.build_wall_seconds,
            name="build_wall_seconds",
        )
        if selected_count > all_count:
            raise ValueError(
                "selected_obstacle_point_count cannot exceed the total."
            )
        if not isinstance(self.tracking_result, ObstacleTrackingResult):
            raise TypeError("tracking_result must be an ObstacleTrackingResult.")
        if not isinstance(self.label_result, SemanticLabelResult):
            raise TypeError("label_result must be a SemanticLabelResult.")
        if not isinstance(self.pipeline, SafeSAGEFieldPipelineResult):
            raise TypeError("pipeline must be a SafeSAGEFieldPipelineResult.")
        temporal_derivative = self.temporal_safety_derivative
        if not isinstance(
            temporal_derivative,
            (SafeSAGETemporalDerivativeResult, Real),
        ) or isinstance(temporal_derivative, bool):
            raise TypeError(
                "temporal_safety_derivative must be a scalar or temporal field."
            )
        if isinstance(temporal_derivative, Real):
            temporal_derivative = _finite_real(
                temporal_derivative,
                name="temporal_safety_derivative",
            )
        if build_time < 0.0:
            raise ValueError("build_wall_seconds cannot be negative.")
        if self.cra_policy_result is not None and not isinstance(
            self.cra_policy_result,
            CRATrackPolicyResult,
        ):
            raise TypeError(
                "cra_policy_result must be a CRATrackPolicyResult or None."
            )
        object.__setattr__(self, "sensor_frame_id", frame_id)
        object.__setattr__(self, "sensor_timestamp_seconds", sensor_time)
        object.__setattr__(self, "all_obstacle_point_count", all_count)
        object.__setattr__(self, "selected_obstacle_point_count", selected_count)
        object.__setattr__(
            self,
            "anchor_position_world_xy",
            _finite_vector(
                self.anchor_position_world_xy,
                size=2,
                name="anchor_position_world_xy",
            ),
        )
        object.__setattr__(self, "anchor_yaw", anchor_yaw)
        object.__setattr__(
            self,
            "temporal_safety_derivative",
            temporal_derivative,
        )
        object.__setattr__(self, "build_wall_seconds", build_time)

    @property
    def selected_track_ids(self) -> tuple[int, ...]:
        """Return tracks whose real LiDAR points entered the field."""

        return self.label_result.selected_track_ids

    @property
    def primary_track_id(self) -> int:
        """Return the first selected track for controlled single-proxy reports."""

        if not self.selected_track_ids:
            raise RuntimeError("The runtime field has no selected track.")
        return self.selected_track_ids[0]

    @property
    def primary_association_distance(self) -> float | None:
        """Return the first label association distance when available."""

        if not self.label_result.assignments:
            return None
        return self.label_result.assignments[0].association_distance

    @property
    def maximum_selected_track_speed(self) -> float:
        """Return the largest selected world-frame track speed."""

        speeds = tuple(
            math.hypot(*obstacle.velocity_world_xy)
            for obstacle in self.label_result.semantic_obstacles
        )
        return 0.0 if not speeds else max(speeds)

    @property
    def has_cra_policy(self) -> bool:
        """Return whether this field used per-track CRA parameters."""

        return self.cra_policy_result is not None

    @property
    def cra_track_ids(self) -> tuple[int, ...]:
        """Return tracks carrying CRA policy data in this field."""

        if self.cra_policy_result is None:
            return ()
        return self.cra_policy_result.track_ids

    @property
    def cra_fallback_track_ids(self) -> tuple[int, ...]:
        """Return tracks whose unavailable CRA profile failed closed."""

        if self.cra_policy_result is None:
            return ()
        return self.cra_policy_result.fallback_track_ids


class SafeSAGEPerceptionRuntime:
    """Build and commit complete live Safe-SAGE fields transactionally."""

    def __init__(
        self,
        *,
        obstacle_tracker: ObstacleTracker,
        class_policy: ClassSafetyPolicy,
        guidance_policy: SafeSAGEGuidancePolicy,
        grid_config: LocalOccupancyGridConfig,
        field_config: SafeSAGEFieldPipelineConfig,
        temporal_estimator: SafeSAGETemporalDerivativeEstimator | None = None,
        live_field_config: SafeSAGELiveFieldConfig | None = None,
        enable_temporal_updates: bool = True,
        cra_policy: CRASafeSAGEPolicy | None = None,
    ) -> None:
        if not isinstance(obstacle_tracker, ObstacleTracker):
            raise TypeError("obstacle_tracker must be an ObstacleTracker.")
        if not isinstance(class_policy, ClassSafetyPolicy):
            raise TypeError("class_policy must be a ClassSafetyPolicy.")
        if not isinstance(guidance_policy, SafeSAGEGuidancePolicy):
            raise TypeError("guidance_policy must be a SafeSAGEGuidancePolicy.")
        if not isinstance(grid_config, LocalOccupancyGridConfig):
            raise TypeError("grid_config must be a LocalOccupancyGridConfig.")
        if not isinstance(field_config, SafeSAGEFieldPipelineConfig):
            raise TypeError("field_config must be a SafeSAGEFieldPipelineConfig.")
        estimator = temporal_estimator or SafeSAGETemporalDerivativeEstimator()
        if not isinstance(estimator, SafeSAGETemporalDerivativeEstimator):
            raise TypeError(
                "temporal_estimator must be a "
                "SafeSAGETemporalDerivativeEstimator."
            )
        if not isinstance(enable_temporal_updates, bool):
            raise TypeError("enable_temporal_updates must be a boolean.")
        if cra_policy is not None and not isinstance(
            cra_policy,
            CRASafeSAGEPolicy,
        ):
            raise TypeError("cra_policy must be a CRASafeSAGEPolicy or None.")
        self._obstacle_tracker = obstacle_tracker
        self.class_policy = class_policy
        self.guidance_policy = guidance_policy
        self.grid_config = grid_config
        self.field_config = field_config
        self.temporal_estimator = estimator
        self.enable_temporal_updates = enable_temporal_updates
        self.cra_policy = cra_policy
        self.field_buffer: SafeSAGELiveFieldBuffer[SafeSAGERuntimeField] = (
            SafeSAGELiveFieldBuffer(live_field_config)
        )
        self._anchor_position_world_xy: np.ndarray | None = None
        self._anchor_yaw: float | None = None

    @property
    def initialized(self) -> bool:
        """Whether an initial complete field has been accepted."""

        return self.field_buffer.initialized

    @property
    def active_track_count(self) -> int:
        """Return retained tracker state from the last accepted update."""

        return self._obstacle_tracker.active_track_count

    @property
    def field(self) -> SafeSAGERuntimeField:
        """Return the latest accepted runtime field."""

        return self.field_buffer.snapshot.payload

    def initialize(
        self,
        observation: SafeSAGEPerceptionObservation,
        label_provider: SemanticLabelProvider,
        *,
        anchor_position_world_xy: object | None = None,
        anchor_yaw: float | None = None,
    ) -> SafeSAGELiveFieldStatus[SafeSAGERuntimeField]:
        """Build and commit the first field in one immutable anchor frame."""

        if not isinstance(label_provider, SemanticLabelProvider):
            raise TypeError("label_provider must implement SemanticLabelProvider.")
        return self.initialize_from_tracking(
            observation,
            lambda tracking_result: label_provider.assign(
                tracking_result.tracked_obstacles
            ),
            anchor_position_world_xy=anchor_position_world_xy,
            anchor_yaw=anchor_yaw,
        )

    def initialize_from_tracking(
        self,
        observation: SafeSAGEPerceptionObservation,
        label_resolver: Callable[[ObstacleTrackingResult], SemanticLabelResult],
        *,
        anchor_position_world_xy: object | None = None,
        anchor_yaw: float | None = None,
    ) -> SafeSAGELiveFieldStatus[SafeSAGERuntimeField]:
        """Initialize with labels resolved from the exact tracking result.

        This entry point is intended for camera--LiDAR fusion. The callback
        receives the tracking result produced by the runtime's cloned
        candidate tracker, so semantic association cannot accidentally use a
        second tracker update or misaligned point-to-track identifiers.
        """

        if self.initialized:
            raise RuntimeError("The Safe-SAGE perception runtime is initialized.")
        self._validate_observation(observation)
        self._validate_label_resolver(label_resolver)
        anchor_position = (
            observation.robot_position_world_xy
            if anchor_position_world_xy is None
            else _finite_vector(
                anchor_position_world_xy,
                size=2,
                name="anchor_position_world_xy",
            )
        )
        resolved_anchor_yaw = (
            observation.robot_yaw
            if anchor_yaw is None
            else _finite_real(anchor_yaw, name="anchor_yaw")
        )
        candidate_tracker = copy.deepcopy(self._obstacle_tracker)
        field = self._build_candidate(
            observation,
            label_resolver,
            obstacle_tracker=candidate_tracker,
            anchor_position_world_xy=anchor_position,
            anchor_yaw=resolved_anchor_yaw,
            previous_snapshot=None,
        )
        status = self.field_buffer.initialize(self._snapshot(observation, field))
        self._obstacle_tracker = candidate_tracker
        self._anchor_position_world_xy = np.array(anchor_position, copy=True)
        self._anchor_position_world_xy.setflags(write=False)
        self._anchor_yaw = resolved_anchor_yaw
        return status

    def initialize_with_cra(
        self,
        observation: SafeSAGEPerceptionObservation,
        label_provider: SemanticLabelProvider,
        risk_profile_provider: CRARiskProfileProvider,
        *,
        anchor_position_world_xy: object | None = None,
        anchor_yaw: float | None = None,
    ) -> SafeSAGELiveFieldStatus[SafeSAGERuntimeField]:
        """Build the first field using per-track CRA parameters."""

        if not isinstance(label_provider, SemanticLabelProvider):
            raise TypeError(
                "label_provider must implement SemanticLabelProvider."
            )
        return self.initialize_from_tracking_with_cra(
            observation,
            lambda tracking_result: label_provider.assign(
                tracking_result.tracked_obstacles
            ),
            risk_profile_provider,
            anchor_position_world_xy=anchor_position_world_xy,
            anchor_yaw=anchor_yaw,
        )

    def initialize_from_tracking_with_cra(
        self,
        observation: SafeSAGEPerceptionObservation,
        label_resolver: Callable[
            [ObstacleTrackingResult],
            SemanticLabelResult,
        ],
        risk_profile_provider: CRARiskProfileProvider,
        *,
        anchor_position_world_xy: object | None = None,
        anchor_yaw: float | None = None,
    ) -> SafeSAGELiveFieldStatus[SafeSAGERuntimeField]:
        """Initialize from candidate tracks, labels, and contextual risk."""

        self._validate_cra_profile_provider(risk_profile_provider)
        if self.initialized:
            raise RuntimeError(
                "The Safe-SAGE perception runtime is initialized."
            )
        self._validate_observation(observation)
        self._validate_label_resolver(label_resolver)
        anchor_position = (
            observation.robot_position_world_xy
            if anchor_position_world_xy is None
            else _finite_vector(
                anchor_position_world_xy,
                size=2,
                name="anchor_position_world_xy",
            )
        )
        resolved_anchor_yaw = (
            observation.robot_yaw
            if anchor_yaw is None
            else _finite_real(anchor_yaw, name="anchor_yaw")
        )
        candidate_tracker = copy.deepcopy(self._obstacle_tracker)
        field = self._build_candidate(
            observation,
            label_resolver,
            obstacle_tracker=candidate_tracker,
            anchor_position_world_xy=anchor_position,
            anchor_yaw=resolved_anchor_yaw,
            previous_snapshot=None,
            risk_profile_provider=risk_profile_provider,
        )
        status = self.field_buffer.initialize(
            self._snapshot(observation, field)
        )
        self._obstacle_tracker = candidate_tracker
        self._anchor_position_world_xy = np.array(
            anchor_position,
            copy=True,
        )
        self._anchor_position_world_xy.setflags(write=False)
        self._anchor_yaw = resolved_anchor_yaw
        return status

    def initialize_from_tracking_with_cra_resolver(
        self,
        observation: SafeSAGEPerceptionObservation,
        label_resolver: Callable[
            [ObstacleTrackingResult],
            SemanticLabelResult,
        ],
        risk_profile_resolver: Callable[
            [ObstacleTrackingResult, SemanticLabelResult],
            CRARiskProfileProvider,
        ],
        *,
        anchor_position_world_xy: object | None = None,
        anchor_yaw: float | None = None,
    ) -> SafeSAGELiveFieldStatus[SafeSAGERuntimeField]:
        """Build CRA profiles from the exact candidate tracks and labels."""

        self._validate_cra_profile_resolver(risk_profile_resolver)
        if self.initialized:
            raise RuntimeError(
                "The Safe-SAGE perception runtime is initialized."
            )
        self._validate_observation(observation)
        self._validate_label_resolver(label_resolver)
        anchor_position = (
            observation.robot_position_world_xy
            if anchor_position_world_xy is None
            else _finite_vector(
                anchor_position_world_xy,
                size=2,
                name="anchor_position_world_xy",
            )
        )
        resolved_anchor_yaw = (
            observation.robot_yaw
            if anchor_yaw is None
            else _finite_real(anchor_yaw, name="anchor_yaw")
        )
        candidate_tracker = copy.deepcopy(self._obstacle_tracker)
        field = self._build_candidate(
            observation,
            label_resolver,
            obstacle_tracker=candidate_tracker,
            anchor_position_world_xy=anchor_position,
            anchor_yaw=resolved_anchor_yaw,
            previous_snapshot=None,
            risk_profile_resolver=risk_profile_resolver,
        )
        status = self.field_buffer.initialize(
            self._snapshot(observation, field)
        )
        self._obstacle_tracker = candidate_tracker
        self._anchor_position_world_xy = np.array(
            anchor_position,
            copy=True,
        )
        self._anchor_position_world_xy.setflags(write=False)
        self._anchor_yaw = resolved_anchor_yaw
        return status

    def update(
        self,
        observation: SafeSAGEPerceptionObservation,
        label_provider: SemanticLabelProvider,
    ) -> SafeSAGELiveFieldStatus[SafeSAGERuntimeField]:
        """Atomically accept one newer complete field and tracker state."""

        if not isinstance(label_provider, SemanticLabelProvider):
            raise TypeError("label_provider must implement SemanticLabelProvider.")
        return self.update_from_tracking(
            observation,
            lambda tracking_result: label_provider.assign(
                tracking_result.tracked_obstacles
            ),
        )

    def update_from_tracking(
        self,
        observation: SafeSAGEPerceptionObservation,
        label_resolver: Callable[[ObstacleTrackingResult], SemanticLabelResult],
    ) -> SafeSAGELiveFieldStatus[SafeSAGERuntimeField]:
        """Update using semantic labels resolved on the candidate tracks."""

        self._validate_observation(observation)
        self._validate_label_resolver(label_resolver)
        if not self.initialized:
            raise RuntimeError("Initialize the perception runtime before update.")
        if self._anchor_position_world_xy is None or self._anchor_yaw is None:
            raise RuntimeError("The perception runtime anchor is unavailable.")
        candidate_tracker = copy.deepcopy(self._obstacle_tracker)
        field = self._build_candidate(
            observation,
            label_resolver,
            obstacle_tracker=candidate_tracker,
            anchor_position_world_xy=self._anchor_position_world_xy,
            anchor_yaw=self._anchor_yaw,
            previous_snapshot=self.field_buffer.snapshot,
        )
        status = self.field_buffer.commit(self._snapshot(observation, field))
        self._obstacle_tracker = candidate_tracker
        return status

    def update_with_cra(
        self,
        observation: SafeSAGEPerceptionObservation,
        label_provider: SemanticLabelProvider,
        risk_profile_provider: CRARiskProfileProvider,
    ) -> SafeSAGELiveFieldStatus[SafeSAGERuntimeField]:
        """Atomically update tracking, semantics, CRA, and the live field."""

        if not isinstance(label_provider, SemanticLabelProvider):
            raise TypeError(
                "label_provider must implement SemanticLabelProvider."
            )
        return self.update_from_tracking_with_cra(
            observation,
            lambda tracking_result: label_provider.assign(
                tracking_result.tracked_obstacles
            ),
            risk_profile_provider,
        )

    def update_from_tracking_with_cra(
        self,
        observation: SafeSAGEPerceptionObservation,
        label_resolver: Callable[
            [ObstacleTrackingResult],
            SemanticLabelResult,
        ],
        risk_profile_provider: CRARiskProfileProvider,
    ) -> SafeSAGELiveFieldStatus[SafeSAGERuntimeField]:
        """Update from the exact candidate tracks, labels, and CRA profiles."""

        self._validate_cra_profile_provider(risk_profile_provider)
        self._validate_observation(observation)
        self._validate_label_resolver(label_resolver)
        if not self.initialized:
            raise RuntimeError(
                "Initialize the perception runtime before update."
            )
        if self._anchor_position_world_xy is None or self._anchor_yaw is None:
            raise RuntimeError(
                "The perception runtime anchor is unavailable."
            )
        candidate_tracker = copy.deepcopy(self._obstacle_tracker)
        field = self._build_candidate(
            observation,
            label_resolver,
            obstacle_tracker=candidate_tracker,
            anchor_position_world_xy=self._anchor_position_world_xy,
            anchor_yaw=self._anchor_yaw,
            previous_snapshot=self.field_buffer.snapshot,
            risk_profile_provider=risk_profile_provider,
        )
        status = self.field_buffer.commit(
            self._snapshot(observation, field)
        )
        self._obstacle_tracker = candidate_tracker
        return status

    def update_from_tracking_with_cra_resolver(
        self,
        observation: SafeSAGEPerceptionObservation,
        label_resolver: Callable[
            [ObstacleTrackingResult],
            SemanticLabelResult,
        ],
        risk_profile_resolver: Callable[
            [ObstacleTrackingResult, SemanticLabelResult],
            CRARiskProfileProvider,
        ],
    ) -> SafeSAGELiveFieldStatus[SafeSAGERuntimeField]:
        """Atomically resolve current tracks, labels, CRA, and the field."""

        self._validate_cra_profile_resolver(risk_profile_resolver)
        self._validate_observation(observation)
        self._validate_label_resolver(label_resolver)
        if not self.initialized:
            raise RuntimeError(
                "Initialize the perception runtime before update."
            )
        if self._anchor_position_world_xy is None or self._anchor_yaw is None:
            raise RuntimeError(
                "The perception runtime anchor is unavailable."
            )
        candidate_tracker = copy.deepcopy(self._obstacle_tracker)
        field = self._build_candidate(
            observation,
            label_resolver,
            obstacle_tracker=candidate_tracker,
            anchor_position_world_xy=self._anchor_position_world_xy,
            anchor_yaw=self._anchor_yaw,
            previous_snapshot=self.field_buffer.snapshot,
            risk_profile_resolver=risk_profile_resolver,
        )
        status = self.field_buffer.commit(
            self._snapshot(observation, field)
        )
        self._obstacle_tracker = candidate_tracker
        return status

    def reuse(
        self,
        *,
        current_simulation_time_seconds: float,
    ) -> SafeSAGELiveFieldStatus[SafeSAGERuntimeField]:
        """Reuse the last accepted field subject to freshness guards."""

        return self.field_buffer.reuse(
            current_simulation_time_seconds=current_simulation_time_seconds
        )

    def inspect(
        self,
        *,
        current_simulation_time_seconds: float,
    ) -> SafeSAGELiveFieldStatus[SafeSAGERuntimeField]:
        """Inspect freshness without increasing the reuse count."""

        return self.field_buffer.inspect(
            current_simulation_time_seconds=current_simulation_time_seconds
        )

    def fresh_execution_step_budget(
        self,
        *,
        current_simulation_time_seconds: float,
        physics_time_step_seconds: float,
        requested_steps: int,
    ) -> int:
        """Cap command steps so the accepted field cannot become stale.

        A delayed sensor frame may leave less than one full MPC interval in
        the current field-age budget.  Returning a shorter interval lets the
        scenario revisit perception sooner while the robot remains governed
        by a fresh field.  A zero budget means that no wheel command should be
        applied until perception advances.
        """

        if isinstance(requested_steps, bool) or not isinstance(
            requested_steps,
            int,
        ):
            raise TypeError("requested_steps must be an integer.")
        if requested_steps < 0:
            raise ValueError("requested_steps cannot be negative.")
        physics_time_step = _finite_real(
            physics_time_step_seconds,
            name="physics_time_step_seconds",
        )
        if physics_time_step <= 0.0:
            raise ValueError(
                "physics_time_step_seconds must be greater than zero."
            )
        status = self.inspect(
            current_simulation_time_seconds=current_simulation_time_seconds
        )
        if status.fail_closed:
            return 0
        remaining_age = max(
            0.0,
            self.field_buffer.config.maximum_field_age_seconds
            - status.field_age_seconds,
        )
        tolerance = self.field_buffer.config.clock_tolerance_seconds
        available_steps = int(
            math.floor((remaining_age + tolerance) / physics_time_step)
        )
        return min(requested_steps, max(0, available_steps))

    @staticmethod
    def _validate_observation(observation: object) -> None:
        if not isinstance(observation, SafeSAGEPerceptionObservation):
            raise TypeError(
                "observation must be a SafeSAGEPerceptionObservation."
            )

    @staticmethod
    def _validate_label_resolver(label_resolver: object) -> None:
        if not callable(label_resolver):
            raise TypeError("label_resolver must be callable.")

    def _validate_cra_profile_provider(
        self,
        profile_provider: object,
    ) -> None:
        if self.cra_policy is None:
            raise RuntimeError(
                "Configure cra_policy before using CRA runtime entry points."
            )
        if not isinstance(profile_provider, CRARiskProfileProvider):
            raise TypeError(
                "risk_profile_provider must implement "
                "CRARiskProfileProvider."
            )

    def _validate_cra_profile_resolver(self, resolver: object) -> None:
        if self.cra_policy is None:
            raise RuntimeError(
                "Configure cra_policy before using CRA runtime entry points."
            )
        if not callable(resolver):
            raise TypeError("risk_profile_resolver must be callable.")

    def _build_candidate(
        self,
        observation: SafeSAGEPerceptionObservation,
        label_resolver: Callable[
            [ObstacleTrackingResult],
            SemanticLabelResult,
        ],
        *,
        obstacle_tracker: ObstacleTracker,
        anchor_position_world_xy: np.ndarray,
        anchor_yaw: float,
        previous_snapshot: (
            SafeSAGELiveFieldSnapshot[SafeSAGERuntimeField] | None
        ),
        risk_profile_provider: CRARiskProfileProvider | None = None,
        risk_profile_resolver: Callable[
            [ObstacleTrackingResult, SemanticLabelResult],
            CRARiskProfileProvider,
        ]
        | None = None,
    ) -> SafeSAGERuntimeField:
        if risk_profile_provider is not None and risk_profile_resolver is not None:
            raise ValueError(
                "risk profile provider and resolver cannot both be supplied."
            )
        if observation.obstacle_points_robot_xy.shape[0] == 0:
            raise RuntimeError("The observation contains no obstacle points.")
        start_wall_time = time.perf_counter()
        tracking_result = obstacle_tracker.update(
            observation.obstacle_points_robot_xy,
            robot_position_world_xy=observation.robot_position_world_xy,
            robot_yaw=observation.robot_yaw,
            timestamp_seconds=observation.sensor_timestamp_seconds,
        )
        label_result = label_resolver(tracking_result)
        if not isinstance(label_result, SemanticLabelResult):
            raise TypeError(
                "label_resolver must return SemanticLabelResult."
            )
        if not label_result.assignments:
            raise RuntimeError(
                "No geometric track received a semantic label in this frame."
            )
        if risk_profile_resolver is not None:
            risk_profile_provider = risk_profile_resolver(
                tracking_result,
                label_result,
            )
            if not isinstance(risk_profile_provider, CRARiskProfileProvider):
                raise TypeError(
                    "risk_profile_resolver must return a "
                    "CRARiskProfileProvider."
                )
        cra_policy_result: CRATrackPolicyResult | None = None
        if risk_profile_provider is not None:
            if self.cra_policy is None:
                raise RuntimeError(
                    "Internal CRA runtime call has no configured policy."
                )
            cra_policy_result = resolve_cra_track_policy(
                label_result.semantic_obstacles,
                profile_provider=risk_profile_provider,
                policy=self.cra_policy,
            )
        selected_track_ids = np.asarray(
            label_result.selected_track_ids,
            dtype=np.int64,
        )
        selected_mask = np.isin(
            tracking_result.point_track_ids,
            selected_track_ids,
        )
        selected_points_robot = observation.obstacle_points_robot_xy[
            selected_mask
        ]
        selected_point_track_ids = tracking_result.point_track_ids[
            selected_mask
        ]
        if selected_points_robot.shape[0] == 0:
            raise RuntimeError(
                "Semantically selected tracks contain no aligned LiDAR points."
            )
        selected_points_anchor = robot_points_in_anchor_frame(
            selected_points_robot,
            robot_position_world_xy=observation.robot_position_world_xy,
            robot_yaw=observation.robot_yaw,
            anchor_position_world_xy=anchor_position_world_xy,
            anchor_yaw=anchor_yaw,
        )
        source_grid = build_local_occupancy_grid(
            selected_points_anchor,
            point_track_ids=selected_point_track_ids,
            semantic_obstacles=label_result.semantic_obstacles,
            class_safety_policy=self.class_policy,
            track_safety_parameters=(
                None
                if cra_policy_result is None
                else cra_policy_result.safety_parameters_by_track_id
            ),
            config=self.grid_config,
        )
        pipeline = build_safe_sage_field_pipeline(
            source_grid,
            guidance_policy=self.guidance_policy,
            config=self.field_config,
            semantic_obstacles=label_result.semantic_obstacles,
            track_guidance_parameters=(
                None
                if cra_policy_result is None
                else cra_policy_result.guidance_parameters_by_track_id
            ),
        )
        temporal_derivative: TemporalSafetyDerivativeInput = 0.0
        if previous_snapshot is not None and self.enable_temporal_updates:
            previous_temporal = (
                previous_snapshot.payload.temporal_safety_derivative
            )
            previous_estimate = (
                previous_temporal
                if isinstance(
                    previous_temporal,
                    SafeSAGETemporalDerivativeResult,
                )
                else None
            )
            temporal_derivative = self.temporal_estimator.estimate(
                previous_h_full=previous_snapshot.payload.pipeline.h_full,
                current_h_full=pipeline.h_full,
                grid_config=self.grid_config,
                time_delta_seconds=(
                    observation.sensor_timestamp_seconds
                    - previous_snapshot.sensor_timestamp_seconds
                ),
                # Every accepted field uses the same immutable anchor frame.
                egomotion_translation_grid_xy=(0.0, 0.0),
                previous_estimate=previous_estimate,
            )
        return SafeSAGERuntimeField(
            sensor_frame_id=observation.sensor_frame_id,
            sensor_timestamp_seconds=observation.sensor_timestamp_seconds,
            anchor_position_world_xy=anchor_position_world_xy,
            anchor_yaw=anchor_yaw,
            all_obstacle_point_count=int(
                observation.obstacle_points_robot_xy.shape[0]
            ),
            selected_obstacle_point_count=int(selected_points_robot.shape[0]),
            tracking_result=tracking_result,
            label_result=label_result,
            pipeline=pipeline,
            temporal_safety_derivative=temporal_derivative,
            build_wall_seconds=time.perf_counter() - start_wall_time,
            cra_policy_result=cra_policy_result,
        )

    @staticmethod
    def _snapshot(
        observation: SafeSAGEPerceptionObservation,
        field: SafeSAGERuntimeField,
    ) -> SafeSAGELiveFieldSnapshot[SafeSAGERuntimeField]:
        return SafeSAGELiveFieldSnapshot(
            sensor_frame_id=observation.sensor_frame_id,
            sensor_timestamp_seconds=observation.sensor_timestamp_seconds,
            simulation_time_seconds=observation.simulation_time_seconds,
            build_wall_seconds=field.build_wall_seconds,
            payload=field,
        )


class SafeSAGEControlRuntime:
    """Apply one accepted runtime field consistently to MPC and CBF."""

    def __init__(
        self,
        *,
        mpc: SafeSAGEUnicycleMPC,
        backup_filter: SafeSAGEUnicycleCBFFilter,
    ) -> None:
        if not isinstance(mpc, SafeSAGEUnicycleMPC):
            raise TypeError("mpc must be a SafeSAGEUnicycleMPC.")
        if not isinstance(backup_filter, SafeSAGEUnicycleCBFFilter):
            raise TypeError(
                "backup_filter must be a SafeSAGEUnicycleCBFFilter."
            )
        self.mpc = mpc
        self.backup_filter = backup_filter

    def plan(
        self,
        chassis_state_grid: Sequence[float] | np.ndarray,
        nominal_commands: Sequence[float] | np.ndarray,
        field: SafeSAGERuntimeField,
        *,
        initial_command_guess: Sequence[float] | np.ndarray | None = None,
        preferred_lateral_direction_grid_y: float | None = None,
        nominal_goal_grid_xy: Sequence[float] | np.ndarray | None = None,
    ) -> SafeSAGEUnicycleMPCResult:
        """Run predictive filtering against one accepted field snapshot."""

        self._validate_field(field)
        extra = {}
        if nominal_goal_grid_xy is not None:
            from robot.safety.laplace_guidance_tracking_mpc import (
                LaplaceGuidanceTrackingMPC,
            )

            if not isinstance(self.mpc, LaplaceGuidanceTrackingMPC):
                raise ValueError(
                    "nominal goal preference is only available in the explicit guidance ablation"
                )
            extra["nominal_goal_grid_xy"] = _finite_vector(
                nominal_goal_grid_xy, size=2, name="nominal_goal_grid_xy"
            )
        return self.mpc.plan(
            chassis_state_grid,
            nominal_commands,
            initial_command_guess=initial_command_guess,
            configuration_space=field.pipeline.configuration_space,
            h_full=field.pipeline.h_full,
            guidance_fields=field.pipeline.guidance_fields,
            temporal_safety_derivative=field.temporal_safety_derivative,
            preferred_lateral_direction_grid_y=(preferred_lateral_direction_grid_y),
            **extra,
        )

    def filter_command(
        self,
        command: Sequence[float] | np.ndarray,
        chassis_state_grid: Sequence[float] | np.ndarray,
        field: SafeSAGERuntimeField,
    ) -> SafeSAGEUnicycleCBFResult:
        """Recheck one executable command at the measured chassis state."""

        self._validate_field(field)
        resolved_command = _finite_vector(command, size=2, name="command")
        chassis_state = _finite_vector(
            chassis_state_grid,
            size=3,
            name="chassis_state_grid",
        )
        query_state = lookahead_state(
            chassis_state,
            lookahead_distance=self.backup_filter.config.lookahead_distance,
        )
        return self.backup_filter.filter_command(
            float(resolved_command[0]),
            float(resolved_command[1]),
            query_position_grid_xy=query_state[:2],
            query_yaw_radians=float(query_state[2]),
            configuration_space=field.pipeline.configuration_space,
            h_full=field.pipeline.h_full,
            guidance_fields=field.pipeline.guidance_fields,
            temporal_safety_derivative=field.temporal_safety_derivative,
        )

    @staticmethod
    def shifted_warm_start(result: SafeSAGEUnicycleMPCResult) -> np.ndarray:
        """Shift an accepted safe plan and append a terminal stop."""

        if not isinstance(result, SafeSAGEUnicycleMPCResult):
            raise TypeError("result must be a SafeSAGEUnicycleMPCResult.")
        return np.ascontiguousarray(
            np.vstack(
                (
                    result.command_sequence[1:],
                    np.zeros((1, 2), dtype=np.float64),
                )
            ),
            dtype=np.float64,
        )

    @staticmethod
    def _validate_field(field: object) -> None:
        if not isinstance(field, SafeSAGERuntimeField):
            raise TypeError("field must be a SafeSAGERuntimeField.")


def robot_points_in_anchor_frame(
    points_robot_xy: object,
    *,
    robot_position_world_xy: object,
    robot_yaw: float,
    anchor_position_world_xy: object,
    anchor_yaw: float,
) -> np.ndarray:
    """Transform current robot-frame points into one fixed anchor frame."""

    points = _finite_points_xy(points_robot_xy, name="points_robot_xy")
    robot_position = _finite_vector(
        robot_position_world_xy,
        size=2,
        name="robot_position_world_xy",
    )
    anchor_position = _finite_vector(
        anchor_position_world_xy,
        size=2,
        name="anchor_position_world_xy",
    )
    robot_heading = _finite_real(robot_yaw, name="robot_yaw")
    anchor_heading = _finite_real(anchor_yaw, name="anchor_yaw")
    robot_cosine = math.cos(robot_heading)
    robot_sine = math.sin(robot_heading)
    robot_to_world = np.array(
        [[robot_cosine, -robot_sine], [robot_sine, robot_cosine]],
        dtype=np.float64,
    )
    world_points = points @ robot_to_world.T + robot_position
    anchor_cosine = math.cos(anchor_heading)
    anchor_sine = math.sin(anchor_heading)
    world_to_anchor = np.array(
        [[anchor_cosine, anchor_sine], [-anchor_sine, anchor_cosine]],
        dtype=np.float64,
    )
    return np.ascontiguousarray(
        (world_points - anchor_position) @ world_to_anchor.T,
        dtype=np.float64,
    )

"""Convert LiDAR tracker motion into CRA physical attributes.

Velocity estimates come from the existing world-frame centroid tracker.
Route-crossing likelihood is a transparent geometric heuristic: it compares
the obstacle's constant-velocity prediction with the current robot-to-waypoint
segment.  It is contextual evidence for CRA, not a collision constraint and
not a replacement for the distance-aware CBF safety layer.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real

import numpy as np

from robot.perception.physical_attributes import (
    DIMENSIONLESS_UNIT,
    EstimateStatus,
    MotionAttributes,
    ScalarEstimate,
)
from robot.safety.obstacle_tracking import TrackedObstacle


MOTION_SOURCE = "lidar_centroid_tracker"
ROUTE_SOURCE = "tracked_motion_waypoint_intersection"


def _finite_real(value: object, *, name: str) -> float:
    """Return one finite real value while rejecting booleans."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _vector2(value: object, *, name: str) -> tuple[float, float]:
    """Return one immutable finite planar vector."""

    array = np.asarray(value, dtype=np.float64)
    if array.shape != (2,):
        raise ValueError(f"{name} must have shape (2,).")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return float(array[0]), float(array[1])


@dataclass(frozen=True)
class RouteMotionContext:
    """Current world-frame route segment and nominal translational speed."""

    robot_position_world_xy: tuple[float, float]
    waypoint_world_xy: tuple[float, float]
    nominal_robot_speed_mps: float

    def __post_init__(self) -> None:
        robot_position = _vector2(
            self.robot_position_world_xy,
            name="robot_position_world_xy",
        )
        waypoint = _vector2(
            self.waypoint_world_xy,
            name="waypoint_world_xy",
        )
        speed = _finite_real(
            self.nominal_robot_speed_mps,
            name="nominal_robot_speed_mps",
        )
        if speed < 0.0:
            raise ValueError("nominal_robot_speed_mps cannot be negative.")
        object.__setattr__(self, "robot_position_world_xy", robot_position)
        object.__setattr__(self, "waypoint_world_xy", waypoint)
        object.__setattr__(self, "nominal_robot_speed_mps", speed)


@dataclass(frozen=True)
class MotionAttributeConfig:
    """Confidence and route-intersection settings for tracked motion."""

    minimum_observation_count: int = 2
    full_confidence_observation_count: int = 6
    minimum_velocity_confidence: float = 0.35
    maximum_velocity_confidence: float = 0.90
    velocity_uncertainty_mps: float = 0.10
    prediction_horizon_seconds: float = 5.0
    route_corridor_half_width_m: float = 0.50
    route_score_uncertainty: float = 0.15
    unconfirmed_route_confidence: float = 0.20
    minimum_route_length_m: float = 1e-3

    def __post_init__(self) -> None:
        minimum_observations = _positive_integer(
            self.minimum_observation_count,
            name="minimum_observation_count",
        )
        full_observations = _positive_integer(
            self.full_confidence_observation_count,
            name="full_confidence_observation_count",
        )
        if full_observations < minimum_observations:
            raise ValueError(
                "full_confidence_observation_count cannot be smaller than "
                "minimum_observation_count."
            )
        for name in (
            "velocity_uncertainty_mps",
            "prediction_horizon_seconds",
            "route_corridor_half_width_m",
            "minimum_route_length_m",
        ):
            value = _finite_real(getattr(self, name), name=name)
            if value <= 0.0:
                raise ValueError(f"{name} must be greater than zero.")
        confidence_values = (
            self.minimum_velocity_confidence,
            self.maximum_velocity_confidence,
            self.route_score_uncertainty,
            self.unconfirmed_route_confidence,
        )
        converted = tuple(
            _finite_real(value, name="confidence or uncertainty")
            for value in confidence_values
        )
        if not all(0.0 <= value <= 1.0 for value in converted):
            raise ValueError(
                "Confidence and route-score uncertainty values must lie in [0, 1]."
            )
        if self.minimum_velocity_confidence > self.maximum_velocity_confidence:
            raise ValueError(
                "minimum_velocity_confidence cannot exceed "
                "maximum_velocity_confidence."
            )


def estimate_track_motion(
    tracked_obstacle: TrackedObstacle,
    *,
    route_context: RouteMotionContext,
    timestamp_seconds: float,
    config: MotionAttributeConfig | None = None,
) -> MotionAttributes:
    """Estimate motion and route-crossing evidence for one tracked obstacle.

    Velocity X/Y are explicitly world-frame components.  Relative speed uses
    the waypoint-aligned nominal robot velocity.  The route score measures
    spatial overlap between the waypoint segment and a constant-velocity
    obstacle prediction; the safety filter continues to own exact clearance.
    """

    if not isinstance(tracked_obstacle, TrackedObstacle):
        raise TypeError("tracked_obstacle must be a TrackedObstacle.")
    if not isinstance(route_context, RouteMotionContext):
        raise TypeError("route_context must be a RouteMotionContext.")
    settings = config or MotionAttributeConfig()
    if not isinstance(settings, MotionAttributeConfig):
        raise TypeError("config must be a MotionAttributeConfig.")
    timestamp = _finite_real(timestamp_seconds, name="timestamp_seconds")
    if timestamp < 0.0:
        raise ValueError("timestamp_seconds cannot be negative.")

    obstacle_position = np.asarray(
        _vector2(
            tracked_obstacle.centroid_world_xy,
            name="tracked_obstacle.centroid_world_xy",
        ),
        dtype=np.float64,
    )
    obstacle_velocity = np.asarray(
        _vector2(
            tracked_obstacle.velocity_world_xy,
            name="tracked_obstacle.velocity_world_xy",
        ),
        dtype=np.float64,
    )
    observation_count = _positive_integer(
        tracked_obstacle.observation_count,
        name="tracked_obstacle.observation_count",
    )
    velocity_is_confirmed = bool(
        tracked_obstacle.matched_previous_track
        and observation_count >= settings.minimum_observation_count
    )

    route_start = np.asarray(
        route_context.robot_position_world_xy,
        dtype=np.float64,
    )
    route_end = np.asarray(route_context.waypoint_world_xy, dtype=np.float64)
    route_vector = route_end - route_start
    route_length = float(np.linalg.norm(route_vector))
    if route_length >= settings.minimum_route_length_m:
        route_direction = route_vector / route_length
        robot_velocity = (
            route_context.nominal_robot_speed_mps * route_direction
        )
    else:
        robot_velocity = np.zeros(2, dtype=np.float64)

    if velocity_is_confirmed:
        velocity_confidence = _observation_confidence(
            observation_count,
            settings,
        )
        velocity_x = _velocity_estimate(
            float(obstacle_velocity[0]),
            confidence=velocity_confidence,
            timestamp=timestamp,
            settings=settings,
            component="world x",
        )
        velocity_y = _velocity_estimate(
            float(obstacle_velocity[1]),
            confidence=velocity_confidence,
            timestamp=timestamp,
            settings=settings,
            component="world y",
        )
        relative_speed = float(
            np.linalg.norm(obstacle_velocity - robot_velocity)
        )
        speed_uncertainty = math.sqrt(2.0) * settings.velocity_uncertainty_mps
        relative = ScalarEstimate(
            value=relative_speed,
            lower_bound=max(0.0, relative_speed - speed_uncertainty),
            upper_bound=relative_speed + speed_uncertainty,
            confidence=velocity_confidence,
            unit="m/s",
            source=MOTION_SOURCE,
            timestamp_seconds=timestamp,
            status=EstimateStatus.INFERRED,
            evidence=(
                "obstacle velocity minus waypoint-aligned nominal robot velocity",
                f"tracker observation count={observation_count}",
            ),
        )
        route_confidence = velocity_confidence
        predicted_velocity = obstacle_velocity
        route_status = EstimateStatus.INFERRED
        route_motion_evidence = (
            "constant-velocity obstacle prediction uses confirmed tracker output",
        )
    else:
        evidence = (
            f"track has {observation_count} observation(s)",
            (
                "velocity requires a matched track with at least "
                f"{settings.minimum_observation_count} observations"
            ),
        )
        velocity_x = _unavailable_velocity(timestamp, evidence)
        velocity_y = _unavailable_velocity(timestamp, evidence)
        relative = _unavailable_velocity(timestamp, evidence)
        route_confidence = settings.unconfirmed_route_confidence
        predicted_velocity = np.zeros(2, dtype=np.float64)
        route_status = EstimateStatus.FALLBACK
        route_motion_evidence = (
            "unconfirmed track is treated as stationary for route screening",
            "route-score bounds are widened to [0, 1]",
        )

    route_crossing = _route_crossing_estimate(
        route_start=route_start,
        route_end=route_end,
        obstacle_position=obstacle_position,
        obstacle_velocity=predicted_velocity,
        route_length=route_length,
        confidence=route_confidence,
        status=route_status,
        timestamp=timestamp,
        settings=settings,
        motion_evidence=route_motion_evidence,
    )
    return MotionAttributes(
        velocity_x_mps=velocity_x,
        velocity_y_mps=velocity_y,
        relative_speed_mps=relative,
        route_crossing_likelihood=route_crossing,
    )


def _positive_integer(value: object, *, name: str) -> int:
    """Return one positive integer while rejecting booleans."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    return result


def _observation_confidence(
    observation_count: int,
    settings: MotionAttributeConfig,
) -> float:
    """Map track history monotonically into velocity confidence."""

    support_span = (
        settings.full_confidence_observation_count
        - settings.minimum_observation_count
    )
    if support_span == 0:
        fraction = 1.0
    else:
        fraction = min(
            1.0,
            (observation_count - settings.minimum_observation_count)
            / support_span,
        )
    return settings.minimum_velocity_confidence + fraction * (
        settings.maximum_velocity_confidence
        - settings.minimum_velocity_confidence
    )


def _velocity_estimate(
    value: float,
    *,
    confidence: float,
    timestamp: float,
    settings: MotionAttributeConfig,
    component: str,
) -> ScalarEstimate:
    """Construct one signed velocity-component estimate."""

    uncertainty = settings.velocity_uncertainty_mps
    return ScalarEstimate(
        value=value,
        lower_bound=value - uncertainty,
        upper_bound=value + uncertainty,
        confidence=confidence,
        unit="m/s",
        source=MOTION_SOURCE,
        timestamp_seconds=timestamp,
        status=EstimateStatus.INFERRED,
        evidence=(
            f"smoothed centroid velocity resolved along {component}",
        ),
    )


def _unavailable_velocity(
    timestamp: float,
    evidence: tuple[str, ...],
) -> ScalarEstimate:
    """Return explicitly unavailable tracker velocity evidence."""

    return ScalarEstimate.unavailable(
        unit="m/s",
        source=MOTION_SOURCE,
        timestamp_seconds=timestamp,
        evidence=evidence,
    )


def _route_crossing_estimate(
    *,
    route_start: np.ndarray,
    route_end: np.ndarray,
    obstacle_position: np.ndarray,
    obstacle_velocity: np.ndarray,
    route_length: float,
    confidence: float,
    status: EstimateStatus,
    timestamp: float,
    settings: MotionAttributeConfig,
    motion_evidence: tuple[str, ...],
) -> ScalarEstimate:
    """Construct route-crossing evidence from two planar segments."""

    if route_length < settings.minimum_route_length_m:
        return ScalarEstimate.unavailable(
            unit=DIMENSIONLESS_UNIT,
            source=ROUTE_SOURCE,
            timestamp_seconds=timestamp,
            evidence=("robot-to-waypoint route segment has zero length",),
        )
    obstacle_end = (
        obstacle_position
        + settings.prediction_horizon_seconds * obstacle_velocity
    )
    separation = _segment_distance(
        route_start,
        route_end,
        obstacle_position,
        obstacle_end,
    )
    score = float(
        np.clip(
            1.0 - separation / settings.route_corridor_half_width_m,
            0.0,
            1.0,
        )
    )
    if status is EstimateStatus.FALLBACK:
        lower = 0.0
        upper = 1.0
    else:
        lower = max(0.0, score - settings.route_score_uncertainty)
        upper = min(1.0, score + settings.route_score_uncertainty)
    return ScalarEstimate(
        value=score,
        lower_bound=lower,
        upper_bound=upper,
        confidence=confidence,
        unit=DIMENSIONLESS_UNIT,
        source=ROUTE_SOURCE,
        timestamp_seconds=timestamp,
        status=status,
        evidence=(
            "distance between waypoint segment and predicted obstacle segment",
            (
                f"minimum segment separation={separation:.4f} m, "
                f"corridor half-width={settings.route_corridor_half_width_m:.4f} m"
            ),
            (
                f"constant-velocity horizon="
                f"{settings.prediction_horizon_seconds:.3f} s"
            ),
            "this contextual score does not replace CBF clearance constraints",
            *motion_evidence,
        ),
    )


def _segment_distance(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> float:
    """Return the Euclidean distance between two closed 2D segments."""

    if _segments_intersect(first_start, first_end, second_start, second_end):
        return 0.0
    return min(
        _point_segment_distance(first_start, second_start, second_end),
        _point_segment_distance(first_end, second_start, second_end),
        _point_segment_distance(second_start, first_start, first_end),
        _point_segment_distance(second_end, first_start, first_end),
    )


def _point_segment_distance(
    point: np.ndarray,
    segment_start: np.ndarray,
    segment_end: np.ndarray,
) -> float:
    """Return the Euclidean distance from one point to a closed segment."""

    segment = segment_end - segment_start
    squared_length = float(np.dot(segment, segment))
    if squared_length <= 1e-24:
        return float(np.linalg.norm(point - segment_start))
    fraction = float(np.dot(point - segment_start, segment) / squared_length)
    projection = segment_start + np.clip(fraction, 0.0, 1.0) * segment
    return float(np.linalg.norm(point - projection))


def _segments_intersect(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> bool:
    """Return whether two closed 2D segments intersect."""

    tolerance = 1e-12

    def cross(origin: np.ndarray, first: np.ndarray, second: np.ndarray) -> float:
        first_offset = first - origin
        second_offset = second - origin
        return float(
            first_offset[0] * second_offset[1]
            - first_offset[1] * second_offset[0]
        )

    def on_segment(start: np.ndarray, point: np.ndarray, end: np.ndarray) -> bool:
        return bool(
            min(start[0], end[0]) - tolerance
            <= point[0]
            <= max(start[0], end[0]) + tolerance
            and min(start[1], end[1]) - tolerance
            <= point[1]
            <= max(start[1], end[1]) + tolerance
        )

    orientation_1 = cross(first_start, first_end, second_start)
    orientation_2 = cross(first_start, first_end, second_end)
    orientation_3 = cross(second_start, second_end, first_start)
    orientation_4 = cross(second_start, second_end, first_end)
    if (
        orientation_1 * orientation_2 < -tolerance
        and orientation_3 * orientation_4 < -tolerance
    ):
        return True
    return bool(
        (abs(orientation_1) <= tolerance and on_segment(first_start, second_start, first_end))
        or (abs(orientation_2) <= tolerance and on_segment(first_start, second_end, first_end))
        or (abs(orientation_3) <= tolerance and on_segment(second_start, first_start, second_end))
        or (abs(orientation_4) <= tolerance and on_segment(second_start, first_end, second_end))
    )

"""Assemble physical attributes from one accepted live semantic field."""

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Callable

import numpy as np

from robot.perception.fall_height import FallSurfaceEvidence, estimate_fall_height
from robot.perception.motion_attributes import RouteMotionContext, estimate_track_motion
from robot.perception.physical_attributes import MassAttributes, PhysicalAttributes
from robot.perception.camera_lidar_fusion import TrackSemanticAssociation
from robot.perception.physical_geometry import estimate_track_geometry
from robot.safety.obstacle_tracking import ObstacleTrackingResult
from robot.safety.semantic_label_provider import SemanticLabelResult


if TYPE_CHECKING:
    from robot.safety.safe_sage_runtime import SafeSAGERuntimeField


@dataclass(frozen=True)
class LivePhysicalEvidenceContext:
    obstacle_points_robot_xyz: np.ndarray
    floor_height_robot_m: float
    robot_position_world_xy: tuple[float, float]
    waypoint_world_xy: tuple[float, float]
    measured_robot_velocity_world_xy: tuple[float, float]


def build_live_physical_attributes(
    field: "SafeSAGERuntimeField",
    context: LivePhysicalEvidenceContext,
    *,
    mass_provider: Callable[[str, float], MassAttributes] | None = None,
    fall_surface_provider: Callable[[int], FallSurfaceEvidence | None] | None = None,
) -> tuple[PhysicalAttributes, ...]:
    """Build selected track records; absent optional providers stay unavailable."""
    from robot.safety.safe_sage_runtime import SafeSAGERuntimeField

    if not isinstance(field, SafeSAGERuntimeField):
        raise TypeError("field must be SafeSAGERuntimeField")
    return build_tracking_physical_attributes(
        tracking_result=field.tracking_result,
        label_result=field.label_result,
        sensor_timestamp_seconds=field.sensor_timestamp_seconds,
        context=context,
        mass_provider=mass_provider,
        fall_surface_provider=fall_surface_provider,
    )


def build_tracking_physical_attributes(
    *,
    tracking_result: ObstacleTrackingResult,
    label_result: SemanticLabelResult,
    sensor_timestamp_seconds: float,
    context: LivePhysicalEvidenceContext,
    associations: tuple[TrackSemanticAssociation, ...] = (),
    mass_provider: Callable[[str, float], MassAttributes] | None = None,
    fall_surface_provider: Callable[[int], FallSurfaceEvidence | None] | None = None,
) -> tuple[PhysicalAttributes, ...]:
    """Build physical evidence directly from accepted tracking and labels."""
    if not isinstance(tracking_result, ObstacleTrackingResult):
        raise TypeError("tracking_result must be ObstacleTrackingResult")
    if not isinstance(label_result, SemanticLabelResult):
        raise TypeError("label_result must be SemanticLabelResult")
    timestamp = float(sensor_timestamp_seconds)
    if not np.isfinite(timestamp) or timestamp < 0.0:
        raise ValueError("sensor_timestamp_seconds must be finite and nonnegative")
    points = np.asarray(context.obstacle_points_robot_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("obstacle_points_robot_xyz must be finite (N, 3)")
    if points.shape[0] != tracking_result.point_track_ids.size:
        raise ValueError("3D points must align with the accepted tracking result")
    association_by_track_id = {item.track_id: item for item in associations}
    if len(association_by_track_id) != len(associations):
        raise ValueError("associations contains duplicate track IDs")
    tracks = {item.track_id: item for item in tracking_result.tracked_obstacles}
    results = []
    for assignment in label_result.assignments:
        obstacle = assignment.obstacle
        if obstacle.track_id not in tracks:
            raise ValueError("label_result references a track that is not present")
        track = tracks[obstacle.track_id]
        base = PhysicalAttributes.unknown(
            track_id=track.track_id,
            semantic_class=obstacle.semantic_class,
            semantic_confidence=obstacle.semantic_confidence,
            semantic_source=obstacle.label_source,
            timestamp_seconds=timestamp,
        )
        geometry = estimate_track_geometry(
            points,
            tracking_result=tracking_result,
            track_id=track.track_id,
            floor_height_robot_m=context.floor_height_robot_m,
            association=association_by_track_id.get(track.track_id),
        )
        if fall_surface_provider is not None:
            geometry = replace(
                geometry,
                fall_height_m=estimate_fall_height(
                    fall_surface_provider(track.track_id),
                    timestamp_seconds=timestamp,
                ),
            )
        motion = estimate_track_motion(
            track,
            timestamp_seconds=timestamp,
            route_context=RouteMotionContext(
                context.robot_position_world_xy,
                context.waypoint_world_xy,
                0.0,
                context.measured_robot_velocity_world_xy,
            ),
        )
        mass = (
            base.mass
            if mass_provider is None
            else mass_provider(obstacle.semantic_class, timestamp)
        )
        results.append(replace(base, mass=mass, geometry=geometry, motion=motion))
    return tuple(results)

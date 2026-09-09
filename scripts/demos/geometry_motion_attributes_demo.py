"""Validate LiDAR geometry and waypoint-relative motion attributes.

This dependency-light demo exercises the exact data boundaries used at
runtime without launching Isaac Sim:

1. Feed a short sequence of obstacle point clusters to the repository's
   existing LiDAR obstacle tracker.
2. Preserve its point-to-track IDs and world-frame velocity estimate.
3. Attach representative camera--LiDAR association provenance.
4. Estimate the track's visible metric envelope.
5. Compare its predicted motion with a deliberately straight waypoint route.
6. Insert only those observed attributes into ``PhysicalAttributes``.

No mass, stability, contact, or thermal model runs here.  Those fields remain
explicitly unavailable, and this demo does not compute CRA risk or control the
robot.

Run from the project root:

    python -u scripts/demos/geometry_motion_attributes_demo.py
"""

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from robot.perception.camera_lidar_fusion import TrackSemanticAssociation
from robot.perception.motion_attributes import (
    RouteMotionContext,
    estimate_track_motion,
)
from robot.perception.physical_attributes import PhysicalAttributes
from robot.perception.physical_geometry import estimate_track_geometry
from robot.safety.obstacle_tracking import ObstacleTracker


def make_cluster(center_xy: tuple[float, float]) -> np.ndarray:
    """Return one connected, repeatable 3D LiDAR obstacle cluster."""

    offsets_xy = np.array(
        [
            [-0.08, -0.20],
            [-0.08, 0.20],
            [0.08, -0.20],
            [0.08, 0.20],
            [0.00, -0.10],
            [0.00, 0.10],
            [-0.04, 0.00],
            [0.04, 0.00],
            [-0.06, -0.05],
            [0.06, 0.05],
            [-0.02, 0.05],
            [0.02, -0.05],
        ],
        dtype=np.float64,
    )
    heights = np.linspace(0.08, 1.48, offsets_xy.shape[0])
    return np.column_stack(
        (
            center_xy[0] + offsets_xy[:, 0],
            center_xy[1] + offsets_xy[:, 1],
            heights,
        )
    )


def main() -> None:
    """Run the controlled geometry-and-motion adapter validation."""

    tracker = ObstacleTracker()
    tracking_result = None
    points_robot_xyz = np.empty((0, 3), dtype=np.float64)

    # The object moves in world -Y at 0.5 m/s.  The stationary robot has zero
    # yaw here, so its current robot-frame and world-frame points are aligned.
    for frame_index in range(6):
        timestamp = 0.2 * frame_index
        center_y = 1.25 - 0.10 * frame_index
        points_robot_xyz = make_cluster((2.0, center_y))
        tracking_result = tracker.update(
            points_robot_xyz[:, :2],
            robot_position_world_xy=np.array([0.0, 0.0]),
            robot_yaw=0.0,
            timestamp_seconds=timestamp,
        )

    assert tracking_result is not None
    tracked_obstacle = tracking_result.tracked_obstacles[0]
    association = TrackSemanticAssociation(
        track_id=tracked_obstacle.track_id,
        status="matched",
        visible_point_count=12,
        detection_index=0,
        mask_hit_count=10,
        overlap_fraction=10.0 / 12.0,
        association_score=0.75,
        projected_median_inside_box=True,
        projected_median_xy=(318.0, 240.0),
    )

    geometry = estimate_track_geometry(
        points_robot_xyz,
        tracking_result=tracking_result,
        track_id=tracked_obstacle.track_id,
        floor_height_robot_m=0.0,
        association=association,
    )
    route_context = RouteMotionContext(
        robot_position_world_xy=(0.0, 0.0),
        waypoint_world_xy=(4.0, 0.0),
        nominal_robot_speed_mps=0.2,
    )
    motion = estimate_track_motion(
        tracked_obstacle,
        route_context=route_context,
        timestamp_seconds=tracking_result.timestamp_seconds,
    )
    unknown = PhysicalAttributes.unknown(
        track_id=tracked_obstacle.track_id,
        semantic_class="person",
        semantic_confidence=0.90,
        semantic_source="yolo_camera_lidar_fusion",
        timestamp_seconds=tracking_result.timestamp_seconds,
        evidence=("specialist estimators have not populated this family",),
    )
    attributes = replace(
        unknown,
        geometry=geometry,
        motion=motion,
    )

    geometry_available = all(
        estimate.is_available
        for estimate in (
            geometry.width_m,
            geometry.depth_m,
            geometry.height_m,
            geometry.center_height_m,
        )
    )
    specialist_fields_unavailable = all(
        not estimate.is_available
        for group in (
            attributes.mass,
            attributes.stability,
            attributes.contact,
            attributes.thermal,
        )
        for estimate in group.__dict__.values()
    )
    checks = {
        "LiDAR geometry available": geometry_available,
        "Fall height withheld": not geometry.fall_height_m.is_available,
        "Tracked world-Y velocity": (
            abs(float(motion.velocity_y_mps.value) + 0.5) < 1e-9
        ),
        "Straight route crossing found": (
            float(motion.route_crossing_likelihood.value) > 0.95
        ),
        "Specialist fields withheld": specialist_fields_unavailable,
        "Expected available count": attributes.available_estimate_count == 8,
    }

    print("\n" + "=" * 72)
    print("CRA GEOMETRY AND MOTION ATTRIBUTE REPORT")
    print("=" * 72)
    print("\nInput boundaries:")
    print(f"  Track ID:                    {attributes.track_id}")
    print(f"  Semantic class:              {attributes.semantic_class}")
    print(
        "  Camera-LiDAR overlap:       "
        f"{association.overlap_fraction:.3f}"
    )
    print(f"  LiDAR track points:          {tracked_obstacle.point_count}")
    print(f"  Tracker observations:        {tracked_obstacle.observation_count}")

    print("\nVisible LiDAR geometry:")
    print(
        "  Width (robot Y):             "
        f"{geometry.width_m.value:.4f} m"
    )
    print(
        "  Depth (robot X):             "
        f"{geometry.depth_m.value:.4f} m"
    )
    print(f"  Visible height:              {geometry.height_m.value:.4f} m")
    print(
        "  Envelope-center height:      "
        f"{geometry.center_height_m.value:.4f} m"
    )
    print("  Available fall height:       unavailable")

    print("\nTracked motion and route context:")
    print(
        "  World velocity:              "
        f"({motion.velocity_x_mps.value:.4f}, "
        f"{motion.velocity_y_mps.value:.4f}) m/s"
    )
    print(
        "  Relative speed:              "
        f"{motion.relative_speed_mps.value:.4f} m/s"
    )
    print(
        "  Route-crossing likelihood:   "
        f"{motion.route_crossing_likelihood.value:.4f}"
    )
    print("  Nominal waypoint route:      straight from (0, 0) to (4, 0)")

    print("\nAttribute record:")
    print(
        "  Available physical fields:   "
        f"{attributes.available_estimate_count} / 24"
    )
    print("  Mass/contact/thermal/stability: explicitly unavailable")

    print("\nControlled checks:")
    for name, passed in checks.items():
        print(f"  {'[PASS]' if passed else '[FAIL]'} {name}")

    print("\nInterpretation:")
    if all(checks.values()):
        print(
            "  [PASS] Existing LiDAR tracking and camera-LiDAR association "
            "contracts produced metric geometry and waypoint-relative motion "
            "evidence without inventing unsupported physical attributes."
        )
    else:
        print("  [FAIL] Inspect the failed controlled checks above.")
        raise RuntimeError("Geometry and motion attribute demo failed.")
    print("  No CRA score, Safe-SAGE policy, or robot command was computed.")
    print("=" * 72)


if __name__ == "__main__":
    main()

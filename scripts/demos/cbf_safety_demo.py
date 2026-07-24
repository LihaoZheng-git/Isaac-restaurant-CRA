"""Validate the CBF LiDAR perception pipeline without robot motion.

This demo:

1. Loads the restaurant robot scene.
2. Creates a temporary static obstacle ahead of Nova Carter.
3. Keeps the robot stationary.
4. Captures one PandarXT-32 scan.
5. Transforms the scan into the chassis frame.
6. Extracts obstacle points.
7. Prints a detailed coordinate-frame report.

Run from the project root:

    python -u scripts/demos/cbf_safety_demo.py
"""

import sys
import traceback
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from robot.config import (
    EXPECTED_METERS_PER_UNIT,
    PANDAR_XT32_LIDAR_PRIM_PATH,
    PHYSICS_DEVICE,
    PHYSICS_DT,
    ROBOT_BASE_PRIM_PATH,
    ROBOT_PRIM_PATH,
    ROBOT_SCENE_PATH,
)

CENTER_CORRIDOR_HALF_WIDTH = 0.35

from isaacsim import SimulationApp


simulation_app = SimulationApp({"headless": False})


import numpy as np
import isaacsim.core.experimental.utils.app as app_utils
from isaacsim.core.simulation_manager import SimulationManager
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from robot.control.differential_drive import NovaCarterController
from robot.runtime.simulation_loop import (
    advance_playing_steps,
    wait_until_playing,
)
from robot.safety.obstacle_detection import (
    ObstacleDetectionConfig,
    detect_obstacle_points,
)
from robot.scene.loader import load_robot_scene
from robot.scene.transforms import (
    compute_relative_prim_transform,
)
from robot.sensor.lidar import (
    LidarFrameInfo,
    NovaCarterLidar,
)
from robot.sensor.lidar_geometry import transform_points


TEST_OBSTACLE_PRIM_PATH = "/World/CBFTestObstacle"

TEST_OBSTACLE_FORWARD_DISTANCE = 1.50
TEST_OBSTACLE_DIMENSIONS = (
    0.50,
    0.50,
    1.00,
)

SETTLE_STEPS = 60
MAXIMUM_SCAN_WAIT_STEPS = 300

MINIMUM_HEIGHT_ABOVE_FLOOR = 0.05
MAXIMUM_HEIGHT_ABOVE_FLOOR = 1.00

MINIMUM_DETECTION_RANGE = 0.15
MAXIMUM_DETECTION_RANGE = 5.00

ANGULAR_BIN_SIZE_DEGREES = 2.0

TRANSIENT_LIDAR_ERRORS = (
    "No LiDAR data is available yet",
    "contains no points",
    "no valid XYZ points remained",
)

Z_PERCENTILES = (
    0,
    1,
    5,
    25,
    50,
    75,
    95,
    99,
    100,
)


def get_prim_world_transform(
    stage: Usd.Stage,
    prim_path: str,
) -> Gf.Matrix4d:
    """Return the current USD local-to-world transform."""

    prim = stage.GetPrimAtPath(prim_path)

    if not prim.IsValid():
        raise RuntimeError(
            f"Prim was not found: {prim_path}"
        )

    xformable = UsdGeom.Xformable(prim)

    if not xformable:
        raise RuntimeError(
            f"Prim is not transformable: {prim_path}"
        )

    return xformable.ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )


def create_test_obstacle(
    stage: Usd.Stage,
) -> tuple[float, float, float]:
    """Create a session-only static cube ahead of the robot."""

    existing_prim = stage.GetPrimAtPath(
        TEST_OBSTACLE_PRIM_PATH
    )

    if existing_prim.IsValid():
        raise RuntimeError(
            "The test-obstacle prim already exists: "
            f"{TEST_OBSTACLE_PRIM_PATH}"
        )

    robot_to_world = get_prim_world_transform(
        stage,
        ROBOT_BASE_PRIM_PATH,
    )

    forward_position_world = robot_to_world.Transform(
        Gf.Vec3d(
            TEST_OBSTACLE_FORWARD_DISTANCE,
            0.0,
            0.0,
        )
    )

    obstacle_height = TEST_OBSTACLE_DIMENSIONS[2]

    obstacle_position = Gf.Vec3d(
        float(forward_position_world[0]),
        float(forward_position_world[1]),
        obstacle_height / 2.0,
    )

    # Author the obstacle in the session layer. It will disappear when
    # the stage closes and cannot alter restaurant_robot.usda.
    with Usd.EditContext(
        stage,
        stage.GetSessionLayer(),
    ):
        cube = UsdGeom.Cube.Define(
            stage,
            TEST_OBSTACLE_PRIM_PATH,
        )

        cube.CreateSizeAttr(1.0)

        cube.CreateExtentAttr(
            [
                Gf.Vec3f(-0.5, -0.5, -0.5),
                Gf.Vec3f(0.5, 0.5, 0.5),
            ]
        )

        cube.CreateDisplayColorAttr(
            [
                Gf.Vec3f(1.0, 0.1, 0.1),
            ]
        )

        cube.AddTranslateOp().Set(
            obstacle_position
        )

        cube.AddOrientOp().Set(
            Gf.Quatf(1.0)
        )

        cube.AddScaleOp().Set(
            Gf.Vec3f(
                *TEST_OBSTACLE_DIMENSIONS
            )
        )

        collision_api = UsdPhysics.CollisionAPI.Apply(
            cube.GetPrim()
        )

        if not collision_api:
            raise RuntimeError(
                "Could not apply collision to the test obstacle."
            )

    print("\nTemporary CBF test obstacle created:")
    print(f"  Prim:       {TEST_OBSTACLE_PRIM_PATH}")
    print(
        "  Position:   "
        f"({obstacle_position[0]:.3f}, "
        f"{obstacle_position[1]:.3f}, "
        f"{obstacle_position[2]:.3f}) m"
    )
    print(
        "  Dimensions: "
        f"{TEST_OBSTACLE_DIMENSIONS} m"
    )
    print(
        "  Forward distance from chassis center: "
        f"{TEST_OBSTACLE_FORWARD_DISTANCE:.3f} m"
    )
    print("  Layer:      session only")
    print("  Body type:  static")

    return (
        float(obstacle_position[0]),
        float(obstacle_position[1]),
        float(obstacle_position[2]),
    )


def estimate_floor_height_in_robot_frame(
    stage: Usd.Stage,
) -> float:
    """Estimate world-Z-zero height in the chassis frame."""

    robot_to_world = get_prim_world_transform(
        stage,
        ROBOT_BASE_PRIM_PATH,
    )

    robot_origin_world = (
        robot_to_world.ExtractTranslation()
    )

    ground_point_world = Gf.Vec3d(
        robot_origin_world[0],
        robot_origin_world[1],
        0.0,
    )

    world_to_robot = robot_to_world.GetInverse()

    ground_point_robot = world_to_robot.Transform(
        ground_point_world
    )

    return float(ground_point_robot[2])


def wait_for_first_scan(
    lidar: NovaCarterLidar,
    controller: NovaCarterController,
) -> tuple[np.ndarray, LidarFrameInfo, int]:
    """Advance simulation until the first valid LiDAR scan arrives."""

    last_transient_error: RuntimeError | None = None

    for completed_steps in range(
        1,
        MAXIMUM_SCAN_WAIT_STEPS + 1,
    ):
        if not simulation_app.is_running():
            raise RuntimeError(
                "Isaac Sim closed before a LiDAR scan arrived."
            )

        if not app_utils.is_playing():
            simulation_app.update()
            continue

        controller.stop()
        simulation_app.update()

        try:
            points = lidar.get_point_cloud()
        except RuntimeError as error:
            if any(
                message in str(error)
                for message in TRANSIENT_LIDAR_ERRORS
            ):
                last_transient_error = error
                continue

            raise

        frame_info = lidar.get_frame_info()

        return points, frame_info, completed_steps

    detail = (
        f"\nLast LiDAR error: {last_transient_error}"
        if last_transient_error is not None
        else ""
    )

    raise TimeoutError(
        "No valid LiDAR scan arrived after "
        f"{MAXIMUM_SCAN_WAIT_STEPS} rendered updates."
        f"{detail}"
    )


def print_validation_report(
    *,
    robot_from_lidar: np.ndarray,
    points_sensor_frame: np.ndarray,
    points_robot_frame: np.ndarray,
    obstacle_points: np.ndarray,
    frame_info: LidarFrameInfo,
    floor_height_robot: float,
    scan_wait_steps: int,
) -> None:
    """Print the Stage A coordinate and obstacle report."""

    lidar_origin_robot = robot_from_lidar[:3, 3]

    z_values = points_robot_frame[:, 2]

    percentile_values = np.percentile(
        z_values,
        Z_PERCENTILES,
    )

    print("\n" + "=" * 70)
    print("CBF STAGE A VALIDATION REPORT")
    print("=" * 70)

    print("\nCoordinate frames:")
    print(f"  Source: {PANDAR_XT32_LIDAR_PRIM_PATH}")
    print(f"  Target: {ROBOT_BASE_PRIM_PATH}")

    print("\nRobot-from-LiDAR transform:")
    print(
        np.array2string(
            robot_from_lidar,
            precision=5,
            suppress_small=True,
        )
    )

    print("\nLiDAR origin in chassis frame:")
    print(
        "  "
        f"x={lidar_origin_robot[0]:.4f} m, "
        f"y={lidar_origin_robot[1]:.4f} m, "
        f"z={lidar_origin_robot[2]:.4f} m"
    )

    print("\nLiDAR scan:")
    print(f"  Sensor frame ID: {frame_info.sensor_frame_id}")
    print(
        "  Sensor timestamp: "
        f"{frame_info.sensor_timestamp_nanoseconds} ns"
    )
    print(f"  Waited updates: {scan_wait_steps}")
    print(
        "  Sensor-frame points: "
        f"{points_sensor_frame.shape[0]}"
    )
    print(
        "  Robot-frame points:  "
        f"{points_robot_frame.shape[0]}"
    )
    print(
        "  Detected obstacles:   "
        f"{obstacle_points.shape[0]}"
    )

    print("\nHeight interpretation:")
    print(
        "  Estimated floor in chassis frame: "
        f"{floor_height_robot:.4f} m"
    )
    print(
        "  Detection height interval:         "
        f"[{floor_height_robot + MINIMUM_HEIGHT_ABOVE_FLOOR:.4f}, "
        f"{floor_height_robot + MAXIMUM_HEIGHT_ABOVE_FLOOR:.4f}] m"
    )

    print("\nRobot-frame point-cloud Z percentiles:")

    for percentile, value in zip(
        Z_PERCENTILES,
        percentile_values,
        strict=True,
    ):
        print(
            f"  {percentile:>3}%: {value: .4f} m"
        )

    if obstacle_points.shape[0] == 0:
        print("\n[WARNING] No relevant obstacle points were detected.")
    else:
        obstacle_distances = np.linalg.norm(
            obstacle_points,
            axis=1,
        )

        nearest_index = int(
            np.argmin(obstacle_distances)
        )

        nearest_point = obstacle_points[
            nearest_index
        ]

        nearest_distance = float(
            obstacle_distances[nearest_index]
        )

        expected_front_surface_distance = (
            TEST_OBSTACLE_FORWARD_DISTANCE
            - TEST_OBSTACLE_DIMENSIONS[0] / 2.0
        )

        print("\nNearest detected obstacle:")
        print(
            "  Point:    "
            f"x={nearest_point[0]:.4f} m, "
            f"y={nearest_point[1]:.4f} m"
        )
        print(
            f"  Distance: {nearest_distance:.4f} m"
        )
        print(
            "  Expected test-cube front surface: "
            f"approximately "
            f"{expected_front_surface_distance:.4f} m"
        )

        nearest_angle_degrees = float(
            np.degrees(
                np.arctan2(
                    nearest_point[1],
                    nearest_point[0],
                )
            )
        )

        print(
            f"  Direction: {nearest_angle_degrees:.2f} degrees"
        )

        center_mask = (
            (obstacle_points[:, 0] > 0.0)
            & (
                np.abs(obstacle_points[:, 1])
                <= CENTER_CORRIDOR_HALF_WIDTH
            )
        )

        center_points = obstacle_points[
            center_mask
        ]

        print("\nForward-center corridor:")
        print(
            "  Half width: "
            f"{CENTER_CORRIDOR_HALF_WIDTH:.2f} m"
        )
        print(
            f"  Detected points: {center_points.shape[0]}"
        )

        if center_points.shape[0] == 0:
            print(
                "  [WARNING] No obstacle point was found "
                "directly ahead."
            )
        else:
            center_distances = np.linalg.norm(
                center_points,
                axis=1,
            )

            nearest_center_index = int(
                np.argmin(center_distances)
            )

            nearest_center_point = center_points[
                nearest_center_index
            ]

            nearest_center_distance = float(
                center_distances[
                    nearest_center_index
                ]
            )

            print(
                "  Nearest point: "
                f"x={nearest_center_point[0]:.4f} m, "
                f"y={nearest_center_point[1]:.4f} m"
            )
            print(
                "  Nearest distance: "
                f"{nearest_center_distance:.4f} m"
            )
            print(
                "  Expected cube surface: "
                f"approximately "
                f"{expected_front_surface_distance:.4f} m"
            )

    print("\nExpected interpretation:")
    print("  - LiDAR Z should be above the chassis origin.")
    print("  - The estimated floor should match low scan returns.")
    print("  - The test obstacle should appear at positive X.")
    print("  - Its Y coordinate should be close to zero.")
    print("  - The robot remained stationary during this test.")
    print("=" * 70)


def main() -> None:
    """Run the stationary Stage A validation."""

    print("\n" + "=" * 70)
    print("CBF SAFETY DEMO - STAGE A")
    print("=" * 70)
    print("The robot will remain stationary.")

    loaded_scene = load_robot_scene(
        scene_path=ROBOT_SCENE_PATH,
        simulation_app=simulation_app,
        robot_prim_path=ROBOT_PRIM_PATH,
        expected_meters_per_unit=EXPECTED_METERS_PER_UNIT,
    )

    create_test_obstacle(
        loaded_scene.stage
    )

    controller = NovaCarterController(
        robot_prim_path=ROBOT_PRIM_PATH
    )

    SimulationManager.setup_simulation(
        dt=PHYSICS_DT,
        device=PHYSICS_DEVICE,
    )

    print("\nPress Play in Isaac Sim to begin.")

    if not wait_until_playing(simulation_app):
        return

    print("\nPlay detected. Initializing physics...")
    app_utils.update_app(steps=10)

    controller.verify_articulation()

    lidar = NovaCarterLidar(
        lidar_prim_path=PANDAR_XT32_LIDAR_PRIM_PATH
    )
    lidar.initialize()

    def hold_robot_still() -> None:
        controller.stop()

    print(
        f"\nHolding the robot still for "
        f"{SETTLE_STEPS} physics steps..."
    )

    if not advance_playing_steps(
        simulation_app,
        physics_steps=SETTLE_STEPS,
        on_step=hold_robot_still,
    ):
        return

    robot_from_lidar = compute_relative_prim_transform(
        loaded_scene.stage,
        source_prim_path=PANDAR_XT32_LIDAR_PRIM_PATH,
        target_prim_path=ROBOT_BASE_PRIM_PATH,
    )

    print("\nWaiting for a valid LiDAR scan...")

    (
        points_sensor_frame,
        frame_info,
        scan_wait_steps,
    ) = wait_for_first_scan(
        lidar,
        controller,
    )

    points_robot_frame = transform_points(
        points_sensor_frame,
        robot_from_lidar,
    )

    floor_height_robot = (
        estimate_floor_height_in_robot_frame(
            loaded_scene.stage
        )
    )

    detection_config = ObstacleDetectionConfig(
        minimum_height=(
            floor_height_robot
            + MINIMUM_HEIGHT_ABOVE_FLOOR
        ),
        maximum_height=(
            floor_height_robot
            + MAXIMUM_HEIGHT_ABOVE_FLOOR
        ),
        minimum_range=MINIMUM_DETECTION_RANGE,
        maximum_range=MAXIMUM_DETECTION_RANGE,
        forward_only=True,
        angular_bin_size_degrees=(
            ANGULAR_BIN_SIZE_DEGREES
        ),
    )

    obstacle_points = detect_obstacle_points(
        points_robot_frame,
        detection_config,
    )

    print_validation_report(
        robot_from_lidar=robot_from_lidar,
        points_sensor_frame=points_sensor_frame,
        points_robot_frame=points_robot_frame,
        obstacle_points=obstacle_points,
        frame_info=frame_info,
        floor_height_robot=floor_height_robot,
        scan_wait_steps=scan_wait_steps,
    )

    controller.stop()
    app_utils.update_app(steps=5)
    app_utils.stop()

    print("\nThe GUI will remain open for inspection.")
    print("Close the Isaac Sim window when finished.")

    while simulation_app.is_running():
        simulation_app.update()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\n" + "=" * 70)
        print("ERROR DURING CBF STAGE A")
        print("=" * 70)
        print(error)
        traceback.print_exc()
    finally:
        simulation_app.close()
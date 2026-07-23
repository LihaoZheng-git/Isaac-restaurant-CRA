"""Test Nova Carter motion and PandarXT-32 LiDAR capture.

The test loads restaurant_robot.usda, waits for the user to press Play,
captures a stationary point cloud, moves Nova Carter forward while capturing
additional scans, and captures one final scan after stopping.

Each scan is saved as:

    scan_XXXX_description.npy
    scan_XXXX_description_topdown.png

Run from the project root:

    python -u scripts/test/test_lidar.py
"""

import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def find_project_root() -> Path:
    """Find the project containing robot/sensor and scripts."""

    for candidate in Path(__file__).resolve().parents:
        if (
            (candidate / "robot").is_dir()
            and (candidate / "robot" / "sensor").is_dir()
            and (candidate / "scripts").is_dir()
        ):
            return candidate

    raise RuntimeError(
        "Could not find the project root containing "
        "robot/sensor/ and scripts/."
    )


PROJECT_ROOT = find_project_root()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from isaacsim import SimulationApp


# SimulationApp must exist before importing other Isaac Sim modules.
simulation_app = SimulationApp({"headless": False})


import isaacsim.core.experimental.utils.app as app_utils
from isaacsim.core.simulation_manager import SimulationManager

from robot.control.differential_drive import NovaCarterController
from robot.scene.loader import load_robot_scene
from robot.sensor.lidar import (
    LidarFrameInfo,
    NovaCarterLidar,
)


# ---------------------------------------------------------------------------
# Scene and simulation configuration
# ---------------------------------------------------------------------------

ROBOT_SCENE_PATH = (
    PROJECT_ROOT
    / "scenes"
    / "generated"
    / "restaurant_robot.usda"
)

ROBOT_PRIM_PATH = "/World/Robot"

PHYSICS_DT = 1.0 / 60.0
PHYSICS_DEVICE = "cpu"

SETTLE_STEPS = 60

FORWARD_SPEED = 0.5
ANGULAR_SPEED = 0.0
FORWARD_STEPS = 180

# Capture at 5 Hz while the PandarXT-32 runs at 10 Hz. This avoids repeatedly
# saving the same completed sensor scan.
CAPTURE_EVERY_N_STEPS = 30

FINAL_STOP_STEPS = 30
MAX_FIRST_SCAN_WAIT_STEPS = 300
MAX_FINAL_SCAN_WAIT_STEPS = 120

TOP_DOWN_IMAGE_SIZE = 600
TOP_DOWN_RANGE_METERS = 8.0

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "lidar_test"
    / datetime.now().strftime("%Y%m%d_%H%M%S")
)

ScanKey = tuple[int, int]

TRANSIENT_LIDAR_ERRORS = (
    "No LiDAR data is available yet",
    "latest LiDAR update contains no points",
    "no valid XYZ points remained",
)


def wait_for_user_to_press_play() -> bool:
    """Wait until the Isaac Sim timeline is playing."""

    print("\nThe scene is ready.")
    print("Press Play in Isaac Sim to begin the test.")

    while simulation_app.is_running():
        simulation_app.update()

        if app_utils.is_playing():
            return True

    return False


def run_motion_steps(
    controller: NovaCarterController,
    *,
    linear_velocity: float,
    angular_velocity: float,
    physics_steps: int,
) -> bool:
    """Apply one motion command for a fixed number of playing steps."""

    wheel_velocities = controller.wheel_velocities(
        [linear_velocity, angular_velocity]
    )

    completed_steps = 0

    while (
        completed_steps < physics_steps
        and simulation_app.is_running()
    ):
        if app_utils.is_playing():
            controller.apply_wheel_velocities(wheel_velocities)
            simulation_app.update()
            completed_steps += 1
        else:
            simulation_app.update()

    return simulation_app.is_running()


def scan_key(frame_info: LidarFrameInfo) -> ScanKey:
    """Create an identifier for one completed LiDAR scan."""

    return (
        frame_info.sensor_frame_id,
        frame_info.sensor_timestamp_nanoseconds,
    )


def try_read_new_scan(
    lidar: NovaCarterLidar,
    previous_key: ScanKey | None,
) -> tuple[np.ndarray, LidarFrameInfo, ScanKey] | None:
    """Read the latest scan if it differs from the previous scan."""

    try:
        points = lidar.get_point_cloud()
    except RuntimeError as error:
        if any(
            message in str(error)
            for message in TRANSIENT_LIDAR_ERRORS
        ):
            return None

        raise

    frame_info = lidar.get_frame_info()
    current_key = scan_key(frame_info)

    if previous_key is not None and current_key == previous_key:
        return None

    return points, frame_info, current_key


def wait_for_new_scan(
    lidar: NovaCarterLidar,
    controller: NovaCarterController,
    *,
    previous_key: ScanKey | None,
    linear_velocity: float,
    angular_velocity: float,
    maximum_steps: int,
    description: str,
) -> tuple[np.ndarray, LidarFrameInfo, ScanKey]:
    """Advance simulation until a new nonempty LiDAR scan arrives."""

    wheel_velocities = controller.wheel_velocities(
        [linear_velocity, angular_velocity]
    )

    completed_steps = 0
    last_transient_error: RuntimeError | None = None

    print(f"\nWaiting for {description} LiDAR scan...")

    while (
        completed_steps < maximum_steps
        and simulation_app.is_running()
    ):
        if not app_utils.is_playing():
            simulation_app.update()
            continue

        controller.apply_wheel_velocities(wheel_velocities)
        simulation_app.update()
        completed_steps += 1

        try:
            result = try_read_new_scan(
                lidar,
                previous_key,
            )
        except RuntimeError as error:
            if any(
                message in str(error)
                for message in TRANSIENT_LIDAR_ERRORS
            ):
                last_transient_error = error
                continue

            raise

        if result is None:
            continue

        points, frame_info, current_key = result

        print(
            f"Received {description} scan after "
            f"{completed_steps} rendered update(s)."
        )

        return points, frame_info, current_key

    if not simulation_app.is_running():
        raise RuntimeError(
            "Isaac Sim was closed before a LiDAR scan arrived."
        )

    detail = (
        f"\nLast LiDAR error: {last_transient_error}"
        if last_transient_error is not None
        else ""
    )

    raise TimeoutError(
        f"No new {description} LiDAR scan arrived after "
        f"{maximum_steps} rendered updates."
        f"{detail}"
    )


def create_top_down_image(
    points: np.ndarray,
) -> Image.Image:
    """Create a fixed-scale top-down visualization of a point cloud.

    In the generated image:

    - Forward is toward the top.
    - Left is toward the left.
    - The red cross represents the LiDAR origin.
    """

    image_array = np.zeros(
        (
            TOP_DOWN_IMAGE_SIZE,
            TOP_DOWN_IMAGE_SIZE,
            3,
        ),
        dtype=np.uint8,
    )

    x = points[:, 0]
    y = points[:, 1]

    visible_mask = (
        np.isfinite(x)
        & np.isfinite(y)
        & (np.abs(x) <= TOP_DOWN_RANGE_METERS)
        & (np.abs(y) <= TOP_DOWN_RANGE_METERS)
    )

    visible_x = x[visible_mask]
    visible_y = y[visible_mask]

    center = TOP_DOWN_IMAGE_SIZE // 2
    pixels_per_meter = (
        TOP_DOWN_IMAGE_SIZE
        / (2.0 * TOP_DOWN_RANGE_METERS)
    )

    # Sensor convention:
    #   +X is forward, so it maps toward the top of the image.
    #   +Y is left, so it maps toward the left of the image.
    rows = np.rint(
        center - visible_x * pixels_per_meter
    ).astype(np.int32)

    columns = np.rint(
        center - visible_y * pixels_per_meter
    ).astype(np.int32)

    rows = np.clip(
        rows,
        0,
        TOP_DOWN_IMAGE_SIZE - 1,
    )
    columns = np.clip(
        columns,
        0,
        TOP_DOWN_IMAGE_SIZE - 1,
    )

    image_array[rows, columns] = (235, 235, 235)

    image = Image.fromarray(image_array)
    draw = ImageDraw.Draw(image)

    origin_size = 5

    draw.line(
        (
            center - origin_size,
            center,
            center + origin_size,
            center,
        ),
        fill=(255, 60, 60),
        width=2,
    )

    draw.line(
        (
            center,
            center - origin_size,
            center,
            center + origin_size,
        ),
        fill=(255, 60, 60),
        width=2,
    )

    # Short green line showing the forward direction.
    draw.line(
        (
            center,
            center,
            center,
            center - 25,
        ),
        fill=(60, 255, 100),
        width=2,
    )

    return image


def save_point_cloud(
    points: np.ndarray,
    frame_info: LidarFrameInfo,
    filename_stem: str,
) -> tuple[Path, Path]:
    """Save a point cloud and its top-down preview."""

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    point_cloud_path = (
        OUTPUT_DIRECTORY / f"{filename_stem}.npy"
    )

    preview_path = (
        OUTPUT_DIRECTORY
        / f"{filename_stem}_topdown.png"
    )

    np.save(point_cloud_path, points)

    preview = create_top_down_image(points)
    preview.save(preview_path)

    distances = np.linalg.norm(points, axis=1)
    minimum_xyz = np.min(points, axis=0)
    maximum_xyz = np.max(points, axis=0)

    print(
        f"Saved {point_cloud_path.name} | "
        f"points={points.shape[0]} | "
        f"frame={frame_info.sensor_frame_id} | "
        f"time={frame_info.simulation_time_seconds:.3f} s"
    )

    print(
        "  XYZ bounds: "
        f"x=[{minimum_xyz[0]:.2f}, {maximum_xyz[0]:.2f}] m, "
        f"y=[{minimum_xyz[1]:.2f}, {maximum_xyz[1]:.2f}] m, "
        f"z=[{minimum_xyz[2]:.2f}, {maximum_xyz[2]:.2f}] m"
    )

    print(
        "  Range:      "
        f"min={np.min(distances):.2f} m, "
        f"mean={np.mean(distances):.2f} m, "
        f"max={np.max(distances):.2f} m"
    )

    print(f"  Preview:    {preview_path.name}")

    return point_cloud_path, preview_path


def capture_forward_motion(
    lidar: NovaCarterLidar,
    controller: NovaCarterController,
    *,
    starting_scan_index: int,
    previous_key: ScanKey,
) -> tuple[int, ScanKey, list[Path]]:
    """Move Nova Carter forward and save new LiDAR scans."""

    wheel_velocities = controller.wheel_velocities(
        [FORWARD_SPEED, ANGULAR_SPEED]
    )

    expected_scans = (
        FORWARD_STEPS // CAPTURE_EVERY_N_STEPS
    )

    print("\n" + "-" * 70)
    print("FORWARD MOTION AND LIDAR CAPTURE")
    print("-" * 70)
    print(f"Linear speed:          {FORWARD_SPEED} m/s")
    print(f"Angular speed:         {ANGULAR_SPEED} rad/s")
    print(f"Physics steps:         {FORWARD_STEPS}")
    print(f"Capture interval:      {CAPTURE_EVERY_N_STEPS} steps")
    print(f"Expected moving scans: {expected_scans}")

    completed_steps = 0
    next_capture_step = CAPTURE_EVERY_N_STEPS
    scan_index = starting_scan_index
    current_key = previous_key
    saved_paths: list[Path] = []

    while (
        completed_steps < FORWARD_STEPS
        and simulation_app.is_running()
    ):
        if not app_utils.is_playing():
            simulation_app.update()
            continue

        controller.apply_wheel_velocities(wheel_velocities)
        simulation_app.update()
        completed_steps += 1

        if completed_steps < next_capture_step:
            continue

        result = try_read_new_scan(
            lidar,
            current_key,
        )

        # If the sensor has not completed its next scan yet, continue moving
        # and try again on the following rendered update.
        if result is None:
            continue

        points, frame_info, current_key = result

        point_cloud_path, preview_path = save_point_cloud(
            points,
            frame_info,
            f"scan_{scan_index:04d}_moving",
        )

        saved_paths.extend(
            [point_cloud_path, preview_path]
        )

        scan_index += 1
        next_capture_step += CAPTURE_EVERY_N_STEPS

    captured_scans = (
        scan_index - starting_scan_index
    )

    if captured_scans < expected_scans:
        print(
            "\n[WARNING] Expected approximately "
            f"{expected_scans} moving scans, but captured "
            f"{captured_scans}."
        )

    return scan_index, current_key, saved_paths


def main() -> None:
    """Run the complete robot-motion and LiDAR-capture test."""

    print("\n" + "=" * 70)
    print("NOVA CARTER LIDAR AND MOTION TEST")
    print("=" * 70)

    loaded_scene = load_robot_scene(
        scene_path=ROBOT_SCENE_PATH,
        simulation_app=simulation_app,
        robot_prim_path=ROBOT_PRIM_PATH,
        expected_meters_per_unit=1.0,
    )

    print(
        f"\nValidated scene units: "
        f"{loaded_scene.meters_per_unit} m/unit"
    )

    controller = NovaCarterController(
        robot_prim_path=ROBOT_PRIM_PATH
    )
    controller.print_configuration()

    SimulationManager.setup_simulation(
        dt=PHYSICS_DT,
        device=PHYSICS_DEVICE,
    )

    if not wait_for_user_to_press_play():
        return

    print("\nPlay detected. Initializing robot physics...")

    # Initialize the articulation and physics handles.
    app_utils.update_app(steps=10)
    controller.verify_articulation()

    print("\nInitializing the PandarXT-32 LiDAR...")

    lidar = NovaCarterLidar()
    lidar.initialize()

    print(
        f"\nLetting Nova Carter settle for "
        f"{SETTLE_STEPS} physics steps..."
    )

    if not run_motion_steps(
        controller,
        linear_velocity=0.0,
        angular_velocity=0.0,
        physics_steps=SETTLE_STEPS,
    ):
        return

    # ------------------------------------------------------------------
    # Capture the stationary reference point cloud.
    # ------------------------------------------------------------------

    (
        stationary_points,
        stationary_info,
        latest_scan_key,
    ) = wait_for_new_scan(
        lidar,
        controller,
        previous_key=None,
        linear_velocity=0.0,
        angular_velocity=0.0,
        maximum_steps=MAX_FIRST_SCAN_WAIT_STEPS,
        description="stationary",
    )

    stationary_cloud, stationary_preview = (
        save_point_cloud(
            stationary_points,
            stationary_info,
            "scan_0000_stationary",
        )
    )

    saved_files: list[Path] = [
        stationary_cloud,
        stationary_preview,
    ]

    # ------------------------------------------------------------------
    # Move forward and capture point clouds.
    # ------------------------------------------------------------------

    (
        next_scan_index,
        latest_scan_key,
        moving_files,
    ) = capture_forward_motion(
        lidar,
        controller,
        starting_scan_index=1,
        previous_key=latest_scan_key,
    )

    saved_files.extend(moving_files)

    if not simulation_app.is_running():
        return

    # ------------------------------------------------------------------
    # Stop and capture the final point cloud.
    # ------------------------------------------------------------------

    print(
        f"\nStopping Nova Carter for "
        f"{FINAL_STOP_STEPS} physics steps..."
    )

    if not run_motion_steps(
        controller,
        linear_velocity=0.0,
        angular_velocity=0.0,
        physics_steps=FINAL_STOP_STEPS,
    ):
        return

    (
        final_points,
        final_info,
        latest_scan_key,
    ) = wait_for_new_scan(
        lidar,
        controller,
        previous_key=latest_scan_key,
        linear_velocity=0.0,
        angular_velocity=0.0,
        maximum_steps=MAX_FINAL_SCAN_WAIT_STEPS,
        description="final stopped",
    )

    final_cloud, final_preview = save_point_cloud(
        final_points,
        final_info,
        f"scan_{next_scan_index:04d}_stopped",
    )

    saved_files.extend(
        [final_cloud, final_preview]
    )

    controller.stop()
    app_utils.update_app(steps=5)
    app_utils.stop()

    saved_scan_count = len(saved_files) // 2

    print("\n" + "=" * 70)
    print("LIDAR AND MOTION TEST COMPLETED")
    print("=" * 70)
    print(f"LiDAR prim:        {final_info.lidar_prim_path}")
    print(f"Coordinate type:   {final_info.source_coordinate_type}")
    print(f"Final point count: {final_info.returned_point_count}")
    print(f"Scans saved:       {saved_scan_count}")
    print(f"Files saved:       {len(saved_files)}")
    print(f"Output directory:  {OUTPUT_DIRECTORY}")

    print("\nExpected result:")
    print("  - Each .npy file has shape (N, 3) and dtype float32.")
    print("  - The stationary preview shows the initial surroundings.")
    print("  - Moving previews change as the robot travels forward.")
    print("  - The final preview shows the robot's stopped position.")
    print("  - The red cross is the LiDAR origin.")
    print("  - The green line points forward.")

    print("\nThe GUI will remain open for inspection.")
    print("Close the Isaac Sim window when finished.")

    while simulation_app.is_running():
        simulation_app.update()


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print("\n" + "=" * 70)
        print("ERROR DURING LIDAR AND MOTION TEST")
        print("=" * 70)
        print(error)
        traceback.print_exc()

    finally:
        simulation_app.close()
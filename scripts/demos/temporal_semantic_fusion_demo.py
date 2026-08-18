"""Validate temporal camera--LiDAR semantics without robot motion.

Stage A captures a sequence of fresh LiDAR scans and front-Hawk images while
Nova Carter remains stationary. Every scan is geometrically filtered and
tracked; every paired image is segmented by YOLO and fused through the
authored polynomial-fisheye calibration.

Stage B feeds those single-frame labels into ``SemanticTrackMemory``. A class
must receive repeated evidence on the same LiDAR ``track_id`` before it is
emitted, while conflicting classes remain conservative.

Stage C applies controlled all-unknown observations to the final real tracks.
It verifies that a short visual dropout preserves a recent confirmed label and
that the same evidence later expires back to ``unknown``. No label reaches the
safety controller and no nonzero wheel command is issued.

Run from the project root in the cloned YOLO environment:

    $env:YOLO_MODEL_PATH = "C:\\path\\to\\yolo11n-seg.pt"
    python -u scripts/demos/temporal_semantic_fusion_demo.py

Set ``ISAAC_SIM_HEADLESS=1`` for a headless run or
``TEMPORAL_FUSION_AUTOCLOSE=1`` to close a rendered run after the report.
"""

import math
import os
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from robot.config import (
    EXPECTED_METERS_PER_UNIT,
    FRONT_HAWK_LEFT_CAMERA_PRIM_PATH,
    PANDAR_XT32_LIDAR_PRIM_PATH,
    PHYSICS_DEVICE,
    PHYSICS_DT,
    ROBOT_BASE_PRIM_PATH,
    ROBOT_PRIM_PATH,
    ROBOT_SCENE_PATH,
)
from isaacsim import SimulationApp


def environment_flag(name: str) -> bool:
    """Interpret one conventional boolean environment variable."""

    return os.environ.get(name, "0").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }


SIMULATION_HEADLESS = environment_flag("ISAAC_SIM_HEADLESS")
AUTOCLOSE_AFTER_REPORT = environment_flag("TEMPORAL_FUSION_AUTOCLOSE")
simulation_app = SimulationApp(
    {
        "headless": SIMULATION_HEADLESS,
        "extra_args": ["--/rtx/scenedb/maxHistoryTransformCount=20"],
    }
)


import isaacsim.core.experimental.utils.app as app_utils
import numpy as np
from isaacsim.core.simulation_manager import SimulationManager
from PIL import Image
from pxr import Gf, Usd

from robot.control.differential_drive import NovaCarterController
from robot.perception.camera_lidar_fusion import (
    CameraLidarFusionConfig,
    CameraLidarFusionResult,
    CameraLidarSemanticFusion,
)
from robot.perception.semantic_detection import SemanticDetectionFrame
from robot.perception.semantic_track_memory import (
    SemanticTrackMemory,
    SemanticTrackMemoryConfig,
    SemanticTrackMemoryResult,
)
from robot.perception.visualization import render_camera_lidar_fusion_overlay
from robot.perception.yolo_detector import (
    YoloSegmentationConfig,
    YoloSegmentationDetector,
)
from robot.runtime.simulation_loop import (
    advance_playing_steps,
    wait_until_playing,
)
from robot.safety.obstacle_detection import (
    ObstacleDetectionConfig,
    detect_obstacle_points_3d,
)
from robot.safety.obstacle_tracking import (
    ObstacleTracker,
    ObstacleTrackerConfig,
    ObstacleTrackingResult,
)
from robot.safety.semantic_label_provider import (
    SemanticLabelResult,
    SemanticTrackAssignment,
)
from robot.safety.semantic_obstacle import make_unknown_semantic_obstacle
from robot.scene.camera_calibration import read_fisheye_polynomial_calibration
from robot.scene.loader import load_robot_scene
from robot.scene.transforms import (
    compute_relative_prim_transform,
    get_prim_planar_pose,
    get_prim_world_transform,
)
from robot.sensor.camera import CameraFrameInfo, NovaCarterCamera
from robot.sensor.lidar import LidarFrameInfo, NovaCarterLidar
from robot.sensor.lidar_geometry import transform_points


SETTLE_STEPS = 60
TEMPORAL_SAMPLE_COUNT = 6
MAXIMUM_NEW_SCAN_WAIT_STEPS = 120
MAXIMUM_ALLOWED_SENSOR_AGE_SECONDS = 0.25
MAXIMUM_ALLOWED_POSITION_DRIFT = 0.005
MAXIMUM_ALLOWED_YAW_DRIFT = math.radians(0.10)

DETECTION_HEIGHT_ABOVE_FLOOR = (0.05, 1.50)
DETECTION_MAXIMUM_RANGE = 8.0
DETECTION_ANGULAR_BIN_DEGREES = 0.25
SEMANTIC_CLUSTER_RADIUS = 0.07

YOLO_CONFIDENCE_THRESHOLD = 0.35
YOLO_IOU_THRESHOLD = 0.70
YOLO_IMAGE_SIZE = 640
YOLO_DEVICE = os.environ.get("YOLO_DEVICE", "0").strip() or "0"

FUSION_CONFIG = CameraLidarFusionConfig(
    minimum_visible_points=2,
    minimum_mask_hits=4,
    minimum_overlap_fraction=0.10,
    mask_dilation_radius_pixels=6,
    ambiguity_score_margin=0.05,
    projected_median_box_margin_pixels=12,
)
MEMORY_CONFIG = SemanticTrackMemoryConfig(
    evidence_window_seconds=1.0,
    evidence_half_life_seconds=0.50,
    minimum_confirming_observations=2,
    minimum_observation_confidence=0.05,
    minimum_accumulated_evidence=0.12,
    minimum_class_evidence_margin=0.05,
    track_retention_seconds=2.0,
)

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIRECTORY = (
    PROJECT_ROOT / "outputs" / "temporal_semantic_fusion" / RUN_TIMESTAMP
)

TRANSIENT_SENSOR_ERRORS = (
    "No LiDAR data is available yet",
    "latest LiDAR update contains no points",
    "No RGB frame is available yet",
)


@dataclass(frozen=True)
class TemporalFusionSample:
    """One real paired observation and its temporal semantic result."""

    image_rgb: np.ndarray
    camera_info: CameraFrameInfo
    lidar_info: LidarFrameInfo
    obstacle_point_count: int
    tracking: ObstacleTrackingResult
    detections: SemanticDetectionFrame
    fusion: CameraLidarFusionResult
    temporal: SemanticTrackMemoryResult
    sensor_age_seconds: float


def resolve_model_path() -> Path:
    """Return the explicitly configured local segmentation checkpoint."""

    configured_path = os.environ.get("YOLO_MODEL_PATH", "").strip()
    if not configured_path:
        raise RuntimeError(
            "YOLO_MODEL_PATH is not set. Point it to the existing local "
            "yolo11n-seg.pt checkpoint before running this demo."
        )
    path = Path(configured_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"YOLO model weights were not found: {path}")
    return path


def read_chassis_pose(stage: Usd.Stage) -> np.ndarray:
    """Return the current world-frame ``(x, y, yaw)`` chassis pose."""

    return np.asarray(
        get_prim_planar_pose(stage, ROBOT_BASE_PRIM_PATH),
        dtype=np.float64,
    )


def estimate_floor_height_in_robot_frame(stage: Usd.Stage) -> float:
    """Transform the restaurant's world-Z-zero floor into the chassis frame."""

    robot_to_world = get_prim_world_transform(stage, ROBOT_BASE_PRIM_PATH)
    robot_origin_world = robot_to_world.ExtractTranslation()
    ground_world = Gf.Vec3d(
        robot_origin_world[0],
        robot_origin_world[1],
        0.0,
    )
    return float(robot_to_world.GetInverse().Transform(ground_world)[2])


def make_detector(model_path: Path) -> YoloSegmentationDetector:
    """Initialize the optional Ultralytics adapter."""

    detector = YoloSegmentationDetector(
        YoloSegmentationConfig(
            model_path=model_path,
            confidence_threshold=YOLO_CONFIDENCE_THRESHOLD,
            iou_threshold=YOLO_IOU_THRESHOLD,
            image_size=YOLO_IMAGE_SIZE,
            device=YOLO_DEVICE,
        )
    )
    print("\nLoading YOLO11n segmentation weights...")
    detector.initialize()
    print("YOLO model initialized.")
    return detector


def wait_for_new_sensor_pair(
    *,
    camera: NovaCarterCamera,
    lidar: NovaCarterLidar,
    controller: NovaCarterController,
    previous_lidar_frame_id: int | None,
) -> tuple[np.ndarray, CameraFrameInfo, np.ndarray, LidarFrameInfo, int]:
    """Wait for a completed LiDAR frame and pair the current camera image."""

    last_error: RuntimeError | None = None
    for completed_updates in range(1, MAXIMUM_NEW_SCAN_WAIT_STEPS + 1):
        if not simulation_app.is_running():
            raise RuntimeError("Isaac Sim closed before sensor data arrived.")
        if not app_utils.is_playing():
            simulation_app.update()
            continue
        controller.stop()
        simulation_app.update()
        try:
            points_lidar = lidar.get_point_cloud()
            lidar_info = lidar.get_frame_info()
            if lidar_info.sensor_frame_id == previous_lidar_frame_id:
                continue
            image_rgb = camera.get_rgb()
            camera_info = camera.get_frame_info()
        except RuntimeError as error:
            if any(message in str(error) for message in TRANSIENT_SENSOR_ERRORS):
                last_error = error
                continue
            raise
        return (
            image_rgb,
            camera_info,
            points_lidar,
            lidar_info,
            completed_updates,
        )
    detail = f" Last sensor error: {last_error}" if last_error else ""
    raise TimeoutError(
        "A new camera--LiDAR pair did not arrive after "
        f"{MAXIMUM_NEW_SCAN_WAIT_STEPS} updates.{detail}"
    )


def make_unknown_labels(
    tracking: ObstacleTrackingResult,
) -> SemanticLabelResult:
    """Create camera-dropout observations for current real LiDAR tracks."""

    return SemanticLabelResult(
        assignments=tuple(
            SemanticTrackAssignment(
                make_unknown_semantic_obstacle(track)
            )
            for track in tracking.tracked_obstacles
        )
    )


def wrapped_angle_difference(first: float, second: float) -> float:
    """Return the absolute wrapped difference between two yaw angles."""

    return abs(math.atan2(math.sin(first - second), math.cos(first - second)))


def main() -> None:
    """Run live temporal fusion and controlled memory checks."""

    print("\n" + "=" * 78)
    print("TEMPORAL CAMERA--LIDAR SEMANTIC FUSION VALIDATION")
    print("=" * 78)
    model_path = resolve_model_path()
    loaded_scene = load_robot_scene(
        scene_path=ROBOT_SCENE_PATH,
        simulation_app=simulation_app,
        robot_prim_path=ROBOT_PRIM_PATH,
        expected_meters_per_unit=EXPECTED_METERS_PER_UNIT,
    )
    controller = NovaCarterController(robot_prim_path=ROBOT_PRIM_PATH)
    SimulationManager.setup_simulation(dt=PHYSICS_DT, device=PHYSICS_DEVICE)
    print("\nStarting Isaac Sim playback automatically...")
    app_utils.play()
    if not wait_until_playing(simulation_app):
        return
    app_utils.update_app(steps=10)
    controller.verify_articulation()
    controller.print_configuration()

    camera = NovaCarterCamera()
    lidar = NovaCarterLidar()
    camera.initialize()
    lidar.initialize()
    calibration = read_fisheye_polynomial_calibration(
        loaded_scene.stage,
        camera_prim_path=FRONT_HAWK_LEFT_CAMERA_PRIM_PATH,
        output_resolution=camera.resolution,
    )
    robot_from_lidar = compute_relative_prim_transform(
        loaded_scene.stage,
        source_prim_path=PANDAR_XT32_LIDAR_PRIM_PATH,
        target_prim_path=ROBOT_BASE_PRIM_PATH,
    )
    camera_from_robot = compute_relative_prim_transform(
        loaded_scene.stage,
        source_prim_path=ROBOT_BASE_PRIM_PATH,
        target_prim_path=FRONT_HAWK_LEFT_CAMERA_PRIM_PATH,
    )
    floor_height = estimate_floor_height_in_robot_frame(loaded_scene.stage)
    detection_config = ObstacleDetectionConfig(
        minimum_height=floor_height + DETECTION_HEIGHT_ABOVE_FLOOR[0],
        maximum_height=floor_height + DETECTION_HEIGHT_ABOVE_FLOOR[1],
        maximum_range=DETECTION_MAXIMUM_RANGE,
        forward_only=True,
        angular_bin_size_degrees=DETECTION_ANGULAR_BIN_DEGREES,
    )
    tracker = ObstacleTracker(
        ObstacleTrackerConfig(
            cluster_radius=SEMANTIC_CLUSTER_RADIUS,
            minimum_cluster_points=2,
            maximum_association_distance=0.50,
        )
    )
    fusion_filter = CameraLidarSemanticFusion(FUSION_CONFIG)
    semantic_memory = SemanticTrackMemory(MEMORY_CONFIG)
    detector = make_detector(model_path)

    print(f"\nHolding Nova Carter still for {SETTLE_STEPS} physics steps...")
    if not advance_playing_steps(
        simulation_app,
        physics_steps=SETTLE_STEPS,
        on_step=controller.stop,
    ):
        return
    initial_pose = read_chassis_pose(loaded_scene.stage)
    previous_lidar_frame_id: int | None = None
    samples: list[TemporalFusionSample] = []
    print("\n" + "=" * 78)
    print("STAGE A/B - LIVE MULTI-FRAME TEMPORAL FUSION")
    print("=" * 78)
    for sample_index in range(TEMPORAL_SAMPLE_COUNT):
        (
            image_rgb,
            camera_info,
            points_lidar,
            lidar_info,
            waited_updates,
        ) = wait_for_new_sensor_pair(
            camera=camera,
            lidar=lidar,
            controller=controller,
            previous_lidar_frame_id=previous_lidar_frame_id,
        )
        previous_lidar_frame_id = lidar_info.sensor_frame_id
        points_robot = transform_points(points_lidar, robot_from_lidar)
        obstacle_points = detect_obstacle_points_3d(
            points_robot,
            detection_config,
        )
        pose = read_chassis_pose(loaded_scene.stage)
        tracking = tracker.update(
            obstacle_points[:, :2],
            robot_position_world_xy=pose[:2],
            robot_yaw=float(pose[2]),
            timestamp_seconds=lidar_info.sensor_timestamp_nanoseconds * 1e-9,
        )
        detections = detector.detect(
            image_rgb,
            source_frame_id=camera_info.read_index,
            source_timestamp_seconds=camera_info.simulation_time_seconds,
        )
        fusion = fusion_filter.fuse(
            detection_frame=detections,
            obstacle_points_source_xyz=obstacle_points,
            tracking_result=tracking,
            camera_from_source=camera_from_robot,
            calibration=calibration,
        )
        timestamp = lidar_info.sensor_timestamp_nanoseconds * 1e-9
        temporal = semantic_memory.update(
            fusion.semantic_labels,
            timestamp_seconds=timestamp,
        )
        sensor_age = max(
            0.0,
            camera_info.simulation_time_seconds - timestamp,
        )
        raw_matches = sum(item.is_matched for item in fusion.associations)
        samples.append(
            TemporalFusionSample(
                image_rgb=np.array(image_rgb, copy=True),
                camera_info=camera_info,
                lidar_info=lidar_info,
                obstacle_point_count=obstacle_points.shape[0],
                tracking=tracking,
                detections=detections,
                fusion=fusion,
                temporal=temporal,
                sensor_age_seconds=sensor_age,
            )
        )
        print(
            f"  sample={sample_index + 1:<2} "
            f"lidar_frame={lidar_info.sensor_frame_id:<4} "
            f"wait={waited_updates:<3} "
            f"detections={len(detections.detections):<2} "
            f"tracks={len(tracking.tracked_obstacles):<3} "
            f"raw_matches={raw_matches:<2} "
            f"confirmed={temporal.confirmed_track_count:<2} "
            f"persisted={temporal.persisted_track_count:<2} "
            f"sensor_age={sensor_age:.3f}s"
        )

    final_sample = samples[-1]
    final_timestamp = final_sample.temporal.timestamp_seconds
    unknown_labels = make_unknown_labels(final_sample.tracking)
    dropout_result = semantic_memory.update(
        unknown_labels,
        timestamp_seconds=final_timestamp + 0.10,
    )
    expired_result = semantic_memory.update(
        unknown_labels,
        timestamp_seconds=(
            final_timestamp + MEMORY_CONFIG.evidence_window_seconds + 0.20
        ),
    )

    selected_sample = max(
        samples,
        key=lambda sample: (
            sample.temporal.confirmed_track_count,
            sum(item.is_matched for item in sample.fusion.associations),
            len(sample.detections.detections),
        ),
    )
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    overlay_path = OUTPUT_DIRECTORY / "temporal_fusion_overlay.png"
    overlay = render_camera_lidar_fusion_overlay(
        selected_sample.image_rgb,
        selected_sample.detections,
        selected_sample.fusion,
        point_radius=2,
    )
    Image.fromarray(overlay).save(overlay_path)
    controller.stop()
    app_utils.update_app(steps=5)
    final_pose = read_chassis_pose(loaded_scene.stage)
    app_utils.stop()

    maximum_sensor_age = max(sample.sensor_age_seconds for sample in samples)
    position_drift = float(np.linalg.norm(final_pose[:2] - initial_pose[:2]))
    yaw_drift = wrapped_angle_difference(final_pose[2], initial_pose[2])
    robot_stationary = (
        position_drift <= MAXIMUM_ALLOWED_POSITION_DRIFT
        and yaw_drift <= MAXIMUM_ALLOWED_YAW_DRIFT
    )
    raw_match_seen = any(
        any(item.is_matched for item in sample.fusion.associations)
        for sample in samples
    )
    temporal_confirmation_seen = any(
        sample.temporal.confirmed_track_count > 0 for sample in samples
    )
    dropout_persisted = dropout_result.persisted_track_count > 0
    stale_evidence_expired = expired_result.confirmed_track_count == 0
    sensor_ages_valid = (
        maximum_sensor_age <= MAXIMUM_ALLOWED_SENSOR_AGE_SECONDS
    )
    passed = (
        len(samples) == TEMPORAL_SAMPLE_COUNT
        and raw_match_seen
        and temporal_confirmation_seen
        and dropout_persisted
        and stale_evidence_expired
        and sensor_ages_valid
        and robot_stationary
    )

    print("\n" + "=" * 78)
    print("STAGE C - TEMPORAL MEMORY VALIDATION REPORT")
    print("=" * 78)
    print("\nLive sequence:")
    print(f"  Fresh paired frames:          {len(samples)}")
    print(f"  Raw semantic match observed:  {raw_match_seen}")
    print(f"  Temporal confirmation seen:   {temporal_confirmation_seen}")
    print(f"  Maximum sensor-data age:      {maximum_sensor_age:.4f} s")
    print("\nControlled dropout:")
    print(
        "  Confirmed before dropout:     "
        f"{final_sample.temporal.confirmed_track_count}"
    )
    print(
        "  Persisted after short miss:   "
        f"{dropout_result.persisted_track_count}"
    )
    print(
        "  Confirmed after expiration:   "
        f"{expired_result.confirmed_track_count}"
    )
    print("\nSelected temporal tracks:")
    for item in selected_sample.temporal.tracks:
        if item.is_confirmed:
            print(
                f"  T{item.track_id:<3} class={item.semantic_class:<12} "
                f"status={item.status:<10} confidence={item.confidence:.3f} "
                f"observations={item.supporting_observation_count:<2} "
                f"evidence={item.evidence_score:.3f}"
            )
    print("\nControlled checks:")
    print(f"  Complete sample sequence:     {len(samples) == TEMPORAL_SAMPLE_COUNT}")
    print(f"  Short dropout preserved label:{dropout_persisted}")
    print(f"  Stale evidence became unknown:{stale_evidence_expired}")
    print(f"  Sensor ages accepted:         {sensor_ages_valid}")
    print(f"  Position drift:               {position_drift:.6f} m")
    print(f"  Yaw drift:                    {math.degrees(yaw_drift):.4f} degrees")
    print(f"  Robot remained stationary:    {robot_stationary}")
    print(f"\nSaved temporal overlay:         {overlay_path}")
    print("\nControlled-data boundary:")
    print("  Only real YOLO labels associated with real LiDAR tracks were stored.")
    print("  The controlled dropout removed labels, not obstacle geometry.")
    print("  Expired evidence returned to the conservative unknown class.")
    print("  No result reached Safe-SAGE, CBF, MPC, or the wheels.")
    print("\nInterpretation:")
    if passed:
        print(
            "  [PASS] Persistent LiDAR track IDs stabilized live semantic "
            "labels and expired stale camera evidence conservatively."
        )
    else:
        print(
            "  [WARNING] Temporal fusion did not satisfy every controlled "
            "check. Inspect the live sequence and saved overlay."
        )
    print("=" * 78)
    if SIMULATION_HEADLESS or AUTOCLOSE_AFTER_REPORT:
        print("\nAutomated run complete; closing Isaac Sim.")
        return
    print("\nThe GUI will remain open for inspection.")
    print("Close the Isaac Sim window when finished.")
    while simulation_app.is_running():
        simulation_app.update()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001 - demo must close SimulationApp
        print("\n" + "=" * 78)
        print("ERROR DURING TEMPORAL SEMANTIC FUSION DEMO")
        print("=" * 78)
        print(error)
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()

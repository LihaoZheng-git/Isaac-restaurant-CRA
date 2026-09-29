"""Validate calibrated camera--LiDAR semantic fusion without robot motion.

Stage A - synchronized perception geometry:

1. Load the restaurant scene and preserve Nova Carter's authored sensors.
2. Hold the wheels at zero and capture one PandarXT-32 scan plus one front
   Hawk RGB frame on the same rendered update.
3. Read the Hawk polynomial-fisheye calibration and the USD frame transforms.
4. Preserve obstacle-return heights while clustering them into LiDAR tracks.

Stage B - semantic association:

1. Run YOLO11n instance segmentation on the real RGB frame.
2. Project the tracked LiDAR returns with Isaac Sim's f-theta model.
3. Associate projected points with instance masks and attach labels to the
   existing tracker IDs. Ambiguous or weak evidence stays ``unknown``.

Stage C - controlled validation:

1. Print per-track visibility, overlap, and label diagnostics.
2. Save the raw frame, YOLO overlay, and camera--LiDAR fusion overlay.
3. Verify that the robot remained stationary and no command reached the CBF,
   MPC, or wheel controller.

Run from the project root in the cloned YOLO environment:

    $env:YOLO_MODEL_PATH = "C:\\path\\to\\yolo11n-seg.pt"
    python -u scripts/demos/camera_lidar_semantic_fusion_demo.py

Set ``ISAAC_SIM_HEADLESS=1`` for a headless run or ``FUSION_AUTOCLOSE=1`` to
close a rendered run after the report.
"""

import math
import os
import sys
import traceback
from dataclasses import replace
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
AUTOCLOSE_AFTER_REPORT = environment_flag("FUSION_AUTOCLOSE")
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
from robot.perception.live_physical_evidence import (
    LivePhysicalEvidenceContext,
    build_tracking_physical_attributes,
)
from robot.perception.lidar_fall_surfaces import (
    LidarFallSurfaceResult,
    estimate_lidar_fall_surfaces,
)
from robot.perception.motion_calibration import (
    StationaryMotionCalibration,
    calibrate_stationary_motion_threshold,
)
from robot.perception.physical_attributes import PhysicalAttributes
from robot.perception.semantic_detection import SemanticDetectionFrame
from robot.perception.visualization import (
    render_camera_lidar_fusion_overlay,
    render_semantic_detection_overlay,
)
from robot.perception.visual_mass_runtime import (
    VisualMassRuntimeConfig,
    VisualMassRuntimeEstimator,
)
from robot.perception.visual_mass_live_diagnostic import (
    LiveVisualMassDiagnostic,
    accepted_detection_index,
    evaluate_live_visual_mass_dataflow,
)
from robot.perception.yolo_detector import (
    YoloSegmentationConfig,
    YoloSegmentationDetector,
    rgb_to_bgr_uint8,
)
from robot.risk.physical_profile_provider import (
    PhysicalAttributeCRARiskProfileProvider,
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
from robot.safety.cra_track_policy import (
    CRATrackPolicyResult,
    resolve_cra_track_policy,
)
from robot.safety.cra_safe_sage_policy import CRASafeSAGEPolicy
from robot.scene.camera_calibration import (
    read_fisheye_polynomial_calibration,
)
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
MAXIMUM_SENSOR_WAIT_STEPS = 300
MAXIMUM_ALLOWED_POSITION_DRIFT = 0.005
MAXIMUM_ALLOWED_YAW_DRIFT = math.radians(0.10)
MAXIMUM_SENSOR_DATA_AGE_SECONDS = 0.25

DETECTION_HEIGHT_ABOVE_FLOOR = (0.05, 1.50)
DETECTION_MAXIMUM_RANGE = 8.0
DETECTION_ANGULAR_BIN_DEGREES = 0.25
SEMANTIC_CLUSTER_RADIUS = 0.07
STATIONARY_TRACK_SPEED_THRESHOLD_MPS = float(
    os.environ.get("STATIONARY_TRACK_SPEED_THRESHOLD_MPS", "0.15")
)
STATIONARY_TRACK_MINIMUM_MOTION_UPDATES = int(
    os.environ.get("STATIONARY_TRACK_MINIMUM_MOTION_UPDATES", "3")
)
MOTION_CALIBRATION_FRAME_COUNT = int(
    os.environ.get("MOTION_CALIBRATION_FRAME_COUNT", "12")
)
MOTION_CALIBRATION_MINIMUM_TRACK_POINTS = int(
    os.environ.get("MOTION_CALIBRATION_MINIMUM_TRACK_POINTS", "8")
)
MOTION_CALIBRATION_MINIMUM_OBSERVATIONS = int(
    os.environ.get("MOTION_CALIBRATION_MINIMUM_OBSERVATIONS", "4")
)
FUSION_MINIMUM_MASK_HITS = 4
FUSION_MINIMUM_OVERLAP_FRACTION = 0.10
FUSION_MASK_DILATION_PIXELS = 6
FUSION_PROJECTED_MEDIAN_BOX_MARGIN_PIXELS = 12

YOLO_CONFIDENCE_THRESHOLD = 0.35
YOLO_IOU_THRESHOLD = 0.70
YOLO_IMAGE_SIZE = 640
YOLO_DEVICE = os.environ.get("YOLO_DEVICE", "0").strip() or "0"

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIRECTORY = (
    PROJECT_ROOT / "outputs" / "camera_lidar_semantic_fusion" / RUN_TIMESTAMP
)

TRANSIENT_SENSOR_ERRORS = (
    "No LiDAR data is available yet",
    "latest LiDAR update contains no points",
    "No RGB frame is available yet",
)


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


def make_visual_mass_estimator() -> VisualMassRuntimeEstimator | None:
    """Load an explicitly configured local checkpoint, or leave mass unknown."""
    configured_path = os.environ.get("VISUAL_MASS_CHECKPOINT", "").strip()
    if not configured_path:
        return None
    path = Path(configured_path).expanduser().resolve()
    print(f"Loading bounded ABO visual-mass checkpoint: {path}")
    return VisualMassRuntimeEstimator(
        VisualMassRuntimeConfig(checkpoint_path=path, device=YOLO_DEVICE)
    )


def attach_visual_mass_estimates(
    attributes: tuple[PhysicalAttributes, ...],
    *,
    image_rgb: np.ndarray,
    detections: SemanticDetectionFrame,
    fusion: CameraLidarFusionResult,
    estimator: VisualMassRuntimeEstimator | None,
    timestamp_seconds: float,
) -> tuple[PhysicalAttributes, ...]:
    """Attach supported RGB-plus-LiDAR mass estimates by fused track ID."""
    if estimator is None:
        return attributes
    associations = {item.track_id: item for item in fusion.associations}
    results = []
    for item in attributes:
        association = associations.get(item.track_id)
        detection_index = accepted_detection_index(association)
        if detection_index is None:
            results.append(item)
            continue
        detection = detections.detections[detection_index]
        mass = estimator.estimate(
            image_rgb,
            detection,
            item.geometry,
            timestamp_seconds=timestamp_seconds,
        )
        results.append(replace(item, mass=mass))
    return tuple(results)


def estimate_fall_surfaces_by_track(
    *,
    scene_points_robot_xyz: np.ndarray,
    obstacle_points_robot_xyz: np.ndarray,
    tracking: ObstacleTrackingResult,
    timestamp_seconds: float,
) -> dict[int, LidarFallSurfaceResult]:
    """Infer local support/landing bands without using object height as drop."""
    return {
        track.track_id: estimate_lidar_fall_surfaces(
            scene_points_robot_xyz,
            obstacle_points_robot_xyz[
                tracking.point_track_ids == track.track_id
            ],
            timestamp_seconds=timestamp_seconds,
        )
        for track in tracking.tracked_obstacles
    }


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


def wait_for_sensor_pair(
    *,
    camera: NovaCarterCamera,
    lidar: NovaCarterLidar,
    controller: NovaCarterController,
    minimum_lidar_timestamp_nanoseconds: int | None = None,
) -> tuple[np.ndarray, CameraFrameInfo, np.ndarray, LidarFrameInfo, int]:
    """Read camera and LiDAR data without an intervening rendered update."""

    last_error: RuntimeError | None = None
    for completed_updates in range(1, MAXIMUM_SENSOR_WAIT_STEPS + 1):
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
            image_rgb = camera.get_rgb()
            camera_info = camera.get_frame_info()
        except RuntimeError as error:
            if any(message in str(error) for message in TRANSIENT_SENSOR_ERRORS):
                last_error = error
                continue
            raise
        if (
            minimum_lidar_timestamp_nanoseconds is not None
            and lidar_info.sensor_timestamp_nanoseconds
            <= minimum_lidar_timestamp_nanoseconds
        ):
            continue
        return (
            image_rgb,
            camera_info,
            points_lidar,
            lidar_info,
            completed_updates,
        )
    detail = f" Last sensor error: {last_error}" if last_error else ""
    raise TimeoutError(
        "A camera--LiDAR observation did not arrive after "
        f"{MAXIMUM_SENSOR_WAIT_STEPS} updates.{detail}"
    )


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


def save_images(
    *,
    image_rgb: np.ndarray,
    detections: SemanticDetectionFrame,
    fusion: CameraLidarFusionResult,
) -> tuple[Path, Path, Path]:
    """Save raw, semantic-only, and fused diagnostic images."""

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    raw_rgb = np.ascontiguousarray(rgb_to_bgr_uint8(image_rgb)[:, :, ::-1])
    semantic_rgb = render_semantic_detection_overlay(image_rgb, detections)
    fusion_rgb = render_camera_lidar_fusion_overlay(
        image_rgb,
        detections,
        fusion,
        point_radius=2,
    )
    raw_path = OUTPUT_DIRECTORY / "camera_frame.png"
    semantic_path = OUTPUT_DIRECTORY / "semantic_overlay.png"
    fusion_path = OUTPUT_DIRECTORY / "camera_lidar_fusion_overlay.png"
    Image.fromarray(raw_rgb).save(raw_path)
    Image.fromarray(semantic_rgb).save(semantic_path)
    Image.fromarray(fusion_rgb).save(fusion_path)
    return raw_path, semantic_path, fusion_path


def wrapped_angle_difference(first: float, second: float) -> float:
    """Return the absolute wrapped difference between two yaw angles."""

    return abs(math.atan2(math.sin(first - second), math.cos(first - second)))


def print_report(
    *,
    model_path: Path,
    camera_info: CameraFrameInfo,
    lidar_info: LidarFrameInfo,
    waited_updates: int,
    input_point_count: int,
    obstacle_points_3d: np.ndarray,
    tracking: ObstacleTrackingResult,
    detections: SemanticDetectionFrame,
    fusion: CameraLidarFusionResult,
    physical_attributes: tuple[PhysicalAttributes, ...],
    physical_profile_provider: PhysicalAttributeCRARiskProfileProvider,
    cra_policy_result: CRATrackPolicyResult,
    fall_surface_results: dict[int, LidarFallSurfaceResult],
    motion_calibration: StationaryMotionCalibration,
    published_motion_count: int,
    published_motion_sample_count: int,
    raw_path: Path,
    semantic_path: Path,
    fusion_path: Path,
    visual_mass_checkpoint: Path | None,
    visual_mass_diagnostic: LiveVisualMassDiagnostic | None,
    visual_mass_diagnostic_path: Path | None,
    initial_pose: np.ndarray,
    final_pose: np.ndarray,
) -> bool:
    """Print the validation report and return its acceptance status."""

    position_drift = float(np.linalg.norm(final_pose[:2] - initial_pose[:2]))
    yaw_drift = wrapped_angle_difference(final_pose[2], initial_pose[2])
    lidar_sensor_time = lidar_info.sensor_timestamp_nanoseconds * 1e-9
    sensor_data_age = max(
        0.0,
        camera_info.simulation_time_seconds - lidar_sensor_time,
    )
    projection_inside_image = bool(
        fusion.projection.pixel_xy.shape[0] > 0
        and np.all(fusion.projection.pixel_xy[:, 0] >= 0.0)
        and np.all(fusion.projection.pixel_xy[:, 0] < camera_info.width)
        and np.all(fusion.projection.pixel_xy[:, 1] >= 0.0)
        and np.all(fusion.projection.pixel_xy[:, 1] < camera_info.height)
    )
    matched_count = sum(item.is_matched for item in fusion.associations)
    visible_track_count = sum(
        item.visible_point_count > 0 for item in fusion.associations
    )
    labels_by_track = {
        obstacle.track_id: obstacle
        for obstacle in fusion.semantic_labels.semantic_obstacles
    }
    robot_stationary = (
        position_drift <= MAXIMUM_ALLOWED_POSITION_DRIFT
        and yaw_drift <= MAXIMUM_ALLOWED_YAW_DRIFT
    )
    fixed_lambda_valid = bool(cra_policy_result.assignments) and all(
        math.isclose(
            assignment.guidance_parameters.tangential_strength,
            1.25,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for assignment in cra_policy_result.assignments
    )
    stationary_threshold_valid = (
        STATIONARY_TRACK_SPEED_THRESHOLD_MPS
        >= motion_calibration.recommended_stationary_threshold_mps
    )
    stationary_publication_valid = published_motion_count == 0
    sensor_age_valid = sensor_data_age <= MAXIMUM_SENSOR_DATA_AGE_SECONDS
    visual_mass_valid = (
        visual_mass_diagnostic is None
        or visual_mass_diagnostic.live_dataflow_valid
    )
    passed = (
        projection_inside_image
        and bool(detections.detections)
        and bool(tracking.tracked_obstacles)
        and visible_track_count > 0
        and matched_count > 0
        and sensor_age_valid
        and robot_stationary
        and fixed_lambda_valid
        and stationary_threshold_valid
        and stationary_publication_valid
        and visual_mass_valid
    )

    print("\n" + "=" * 78)
    print("CAMERA--LIDAR SEMANTIC FUSION VALIDATION REPORT")
    print("=" * 78)
    print("\nStage A - calibrated sensor geometry:")
    print(f"  Camera frame:                {camera_info.read_index}")
    print(f"  LiDAR sensor frame:          {lidar_info.sensor_frame_id}")
    print(f"  Waited rendered updates:     {waited_updates}")
    print(f"  Camera resolution:           {camera_info.width} x {camera_info.height}")
    print(f"  Raw LiDAR points:            {input_point_count}")
    print(f"  Filtered 3D obstacle points: {obstacle_points_3d.shape[0]}")
    print(f"  Geometric tracks:            {len(tracking.tracked_obstacles)}")
    print(f"  Projected visible points:    {fusion.projection.pixel_xy.shape[0]}")
    print(f"  LiDAR sensor-data age:       {sensor_data_age:.4f} s")
    print("  Projection model:            authored fisheyePolynomial")
    print("\nStage B - YOLO mask association:")
    print(f"  Model weights:               {model_path}")
    print(f"  Device:                      {YOLO_DEVICE}")
    print(f"  Detected instances:          {len(detections.detections)}")
    print(f"  Detected classes:            {dict(detections.class_counts)}")
    print(f"  Visible LiDAR tracks:        {visible_track_count}")
    print(f"  Semantically matched tracks: {matched_count}")
    print(
        "  Unused camera detections:    "
        f"{fusion.semantic_labels.unmatched_label_count}"
    )
    print("  Track details:")
    for association in fusion.associations:
        obstacle = labels_by_track[association.track_id]
        detection_text = (
            "-"
            if association.detection_index is None
            else str(association.detection_index)
        )
        median_text = (
            "-"
            if association.projected_median_xy is None
            else (
                f"({association.projected_median_xy[0]:.0f},"
                f"{association.projected_median_xy[1]:.0f})"
            )
        )
        box_text = "-"
        if association.detection_index is not None:
            box = detections.detections[
                association.detection_index
            ].bounding_box
            box_text = (
                f"({box.x_min:.0f},{box.y_min:.0f})-"
                f"({box.x_max:.0f},{box.y_max:.0f})"
            )
        print(
            f"    T{association.track_id:<3} "
            f"class={obstacle.semantic_class:<12} "
            f"status={association.status:<28} "
            f"visible={association.visible_point_count:<3} "
            f"detection={detection_text:<3} "
            f"hits={association.mask_hit_count:<3} "
            f"overlap={association.overlap_fraction:.3f} "
            f"center_in_box={str(association.projected_median_inside_box):<5} "
            f"median={median_text:<11} box={box_text:<23} "
            f"score={association.association_score:.3f}"
        )
    print("\nStage C - physical evidence:")
    print(
        "  Visual mass checkpoint:      "
        + (str(visual_mass_checkpoint) if visual_mass_checkpoint else "not configured")
    )
    attributes_by_track = {
        item.track_id: item for item in physical_attributes
    }
    for association in fusion.associations:
        if not association.is_matched:
            continue
        attributes = attributes_by_track[association.track_id]
        geometry = attributes.geometry
        motion = attributes.motion
        print(
            f"  T{attributes.track_id} {attributes.semantic_class}: "
            f"width={geometry.width_m.value:.3f} m, "
            f"depth={geometry.depth_m.value:.3f} m, "
            f"height={geometry.height_m.value:.3f} m"
        )
        print(
            "    sources: geometry="
            f"{geometry.width_m.status.value}; mass="
            f"{attributes.mass.mass_kg.status.value}; motion="
            f"{motion.relative_speed_mps.status.value}; fall_height="
            f"{geometry.fall_height_m.status.value}"
        )
        if attributes.mass.mass_kg.is_available:
            mass = attributes.mass.mass_kg
            print(
                f"    mass={mass.value:.3f} kg "
                f"[{mass.lower_bound:.3f}, {mass.upper_bound:.3f}], "
                f"confidence={mass.confidence:.2f}, status={mass.status.value}"
            )
        surface_result = fall_surface_results[attributes.track_id]
        if geometry.fall_height_m.is_available:
            fall = geometry.fall_height_m
            surfaces = surface_result.evidence
            print(
                f"    fall_height={fall.value:.3f} m "
                f"[{fall.lower_bound:.3f}, {fall.upper_bound:.3f}], "
                f"status={fall.status.value}"
            )
            print(
                f"    support_z={surfaces.support_height_m.value:.3f} m; "
                f"landing_z={surfaces.landing_height_m.value:.3f} m; "
                f"direction={surface_result.fall_direction_robot_xy}"
            )
        else:
            print(
                "    fall-height unavailable: "
                + "; ".join(surface_result.rejection_reasons)
            )
        diagnostic = physical_profile_provider.diagnostics_by_track_id[
            attributes.track_id
        ]
        assignment = cra_policy_result.assignment_for(attributes.track_id)
        print("    CRA factor diagnostics:")
        for factor in (
            diagnostic.potential,
            diagnostic.kinetic,
            diagnostic.contact,
            diagnostic.thermal,
        ):
            estimate = factor.estimate
            if factor.rejection_reasons:
                detail = "unknown: " + "; ".join(factor.rejection_reasons)
            else:
                detail = (
                    f"severity={estimate.severity:.3f}, "
                    f"interaction={estimate.interaction_likelihood:.3f}, "
                    f"confidence={estimate.confidence:.2f}"
                )
            print(f"      {factor.factor:<9} {detail}")
        assessment = assignment.policy_result.assessment
        print(
            f"    aggregate: nominal_phi={assessment.nominal_phi:.3f}, "
            f"conservative_phi={assessment.phi:.3f}, "
            f"flux={assignment.guidance_parameters.boundary_flux:.3f}, "
            f"lambda={assignment.guidance_parameters.tangential_strength:.2f}"
        )
    print("\nStage D - controlled checks:")
    print(f"  Projection inside image:     {projection_inside_image}")
    print(f"  At least one YOLO instance:  {bool(detections.detections)}")
    print(f"  At least one LiDAR track:    {bool(tracking.tracked_obstacles)}")
    print(f"  At least one visible track:  {visible_track_count > 0}")
    print(f"  At least one fused label:    {matched_count > 0}")
    print(f"  Sensor age accepted:         {sensor_age_valid}")
    print(f"  Position drift:              {position_drift:.6f} m")
    print(f"  Yaw drift:                   {math.degrees(yaw_drift):.4f} degrees")
    print(f"  Robot remained stationary:   {robot_stationary}")
    print(f"  Fixed lambda=1.25:            {fixed_lambda_valid}")
    if visual_mass_diagnostic is not None:
        print(
            "  Live visual-mass dataflow:   "
            f"{visual_mass_diagnostic.live_dataflow_valid}"
        )
        print(
            "  Sensor-domain accuracy claim:"
            f" {visual_mass_diagnostic.ready_for_sensor_domain_accuracy_claim}"
        )
        for reason in visual_mass_diagnostic.accuracy_claim_blockers:
            print(f"    accuracy blocker: {reason}")
    print("\nStationary motion calibration:")
    print(f"  Raw matched-track samples:    {motion_calibration.sample_count}")
    print(
        "  Calibration track gate:      "
        f">={MOTION_CALIBRATION_MINIMUM_TRACK_POINTS} points, "
        f">={MOTION_CALIBRATION_MINIMUM_OBSERVATIONS} observations"
    )
    print(
        f"  Median / p95 speed:           "
        f"{motion_calibration.median_speed_mps:.4f} / "
        f"{motion_calibration.p95_speed_mps:.4f} m/s"
    )
    print(
        f"  calibrated quantile / max:    "
        f"{motion_calibration.configured_quantile_speed_mps:.4f} / "
        f"{motion_calibration.maximum_speed_mps:.4f} m/s"
    )
    print(
        f"  Recommended threshold:        "
        f"{motion_calibration.recommended_stationary_threshold_mps:.4f} m/s"
    )
    print(
        f"  Configured threshold:         "
        f"{STATIONARY_TRACK_SPEED_THRESHOLD_MPS:.4f} m/s"
    )
    print(f"  Configured gate accepted:     {stationary_threshold_valid}")
    print(
        f"  Nonzero gated publications:   {published_motion_count} / "
        f"{published_motion_sample_count}"
    )
    print(f"  Stationary publication gate:  {stationary_publication_valid}")
    print("\nSaved validation artifacts:")
    print(f"  Raw camera frame:            {raw_path}")
    print(f"  YOLO overlay:                {semantic_path}")
    print(f"  Fusion overlay:              {fusion_path}")
    if visual_mass_diagnostic_path is not None:
        print(f"  Visual-mass diagnostic:      {visual_mass_diagnostic_path}")
    print("\nControlled-data boundary:")
    print("  LiDAR clustering alone established geometry and track IDs.")
    print("  YOLO masks only attached labels to those existing tracks.")
    print("  Scene prim names and simulator ground truth supplied no label.")
    print("  CRA policy parameters were computed for diagnosis only.")
    print("  No policy parameter reached the field, CBF, MPC, or wheels.")
    print("\nInterpretation:")
    if passed:
        print(
            "  [PASS] Calibrated image projection associated at least one "
            "real YOLO instance with an existing real LiDAR track."
        )
    else:
        print(
            "  [WARNING] Fusion did not satisfy every controlled acceptance "
            "check. Inspect the per-track report and saved overlay."
        )
    print("=" * 78)
    return passed


def main() -> None:
    """Run the complete stationary camera--LiDAR fusion validation."""

    print("\n" + "=" * 78)
    print("CAMERA--LIDAR SEMANTIC FUSION - STATIONARY ISAAC SIM VALIDATION")
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
    print(f"\nHolding Nova Carter still for {SETTLE_STEPS} physics steps...")
    if not advance_playing_steps(
        simulation_app,
        physics_steps=SETTLE_STEPS,
        on_step=controller.stop,
    ):
        return
    initial_pose = read_chassis_pose(loaded_scene.stage)
    print("\nWaiting for a paired camera and LiDAR observation...")
    (
        _,
        _,
        priming_points_lidar,
        priming_lidar_info,
        priming_waited_updates,
    ) = wait_for_sensor_pair(camera=camera, lidar=lidar, controller=controller)
    floor_height = estimate_floor_height_in_robot_frame(loaded_scene.stage)
    priming_points_robot = transform_points(priming_points_lidar, robot_from_lidar)
    priming_obstacle_points_3d = detect_obstacle_points_3d(
        priming_points_robot,
        ObstacleDetectionConfig(
            minimum_height=floor_height + DETECTION_HEIGHT_ABOVE_FLOOR[0],
            maximum_height=floor_height + DETECTION_HEIGHT_ABOVE_FLOOR[1],
            maximum_range=DETECTION_MAXIMUM_RANGE,
            forward_only=True,
            angular_bin_size_degrees=DETECTION_ANGULAR_BIN_DEGREES,
        ),
    )
    if priming_obstacle_points_3d.shape[0] == 0:
        raise RuntimeError("The priming LiDAR scan produced no obstacle points.")
    tracker = ObstacleTracker(
        ObstacleTrackerConfig(
            # The safety demos use 0.35 m to group nearby geometry. Semantic
            # instance association needs finer clusters so neighboring chairs,
            # tables, and walls do not become one mixed single-link component.
            cluster_radius=SEMANTIC_CLUSTER_RADIUS,
            minimum_cluster_points=2,
            maximum_association_distance=0.50,
            stationary_speed_threshold=STATIONARY_TRACK_SPEED_THRESHOLD_MPS,
            minimum_consecutive_motion_updates=(
                STATIONARY_TRACK_MINIMUM_MOTION_UPDATES
            ),
        )
    )
    calibration_tracker = ObstacleTracker(
        ObstacleTrackerConfig(
            cluster_radius=SEMANTIC_CLUSTER_RADIUS,
            minimum_cluster_points=2,
            maximum_association_distance=0.50,
            stationary_speed_threshold=0.0,
            minimum_consecutive_motion_updates=1,
        )
    )
    tracker.update(
        priming_obstacle_points_3d[:, :2],
        robot_position_world_xy=initial_pose[:2],
        robot_yaw=float(initial_pose[2]),
        timestamp_seconds=(
            priming_lidar_info.sensor_timestamp_nanoseconds * 1e-9
        ),
    )
    calibration_tracker.update(
        priming_obstacle_points_3d[:, :2],
        robot_position_world_xy=initial_pose[:2],
        robot_yaw=float(initial_pose[2]),
        timestamp_seconds=(
            priming_lidar_info.sensor_timestamp_nanoseconds * 1e-9
        ),
    )
    print(
        f"Collecting {MOTION_CALIBRATION_FRAME_COUNT} later LiDAR frames "
        "for stationary motion calibration..."
    )
    waited_updates = priming_waited_updates
    previous_lidar_timestamp = priming_lidar_info.sensor_timestamp_nanoseconds
    raw_stationary_speeds = []
    published_motion_count = 0
    published_motion_sample_count = 0
    for _ in range(MOTION_CALIBRATION_FRAME_COUNT):
        (
            image_rgb,
            camera_info,
            points_lidar,
            lidar_info,
            temporal_waited_updates,
        ) = wait_for_sensor_pair(
            camera=camera,
            lidar=lidar,
            controller=controller,
            minimum_lidar_timestamp_nanoseconds=previous_lidar_timestamp,
        )
        waited_updates += temporal_waited_updates
        previous_lidar_timestamp = lidar_info.sensor_timestamp_nanoseconds
        points_robot = transform_points(points_lidar, robot_from_lidar)
        obstacle_points_3d = detect_obstacle_points_3d(
            points_robot,
            ObstacleDetectionConfig(
                minimum_height=floor_height + DETECTION_HEIGHT_ABOVE_FLOOR[0],
                maximum_height=floor_height + DETECTION_HEIGHT_ABOVE_FLOOR[1],
                maximum_range=DETECTION_MAXIMUM_RANGE,
                forward_only=True,
                angular_bin_size_degrees=DETECTION_ANGULAR_BIN_DEGREES,
            ),
        )
        if obstacle_points_3d.shape[0] == 0:
            raise RuntimeError("A calibration LiDAR scan produced no obstacle points.")
        timestamp_seconds = lidar_info.sensor_timestamp_nanoseconds * 1e-9
        tracking = tracker.update(
            obstacle_points_3d[:, :2],
            robot_position_world_xy=initial_pose[:2],
            robot_yaw=float(initial_pose[2]),
            timestamp_seconds=timestamp_seconds,
        )
        raw_tracking = calibration_tracker.update(
            obstacle_points_3d[:, :2],
            robot_position_world_xy=initial_pose[:2],
            robot_yaw=float(initial_pose[2]),
            timestamp_seconds=timestamp_seconds,
        )
        raw_stationary_speeds.extend(
            math.hypot(*track.velocity_world_xy)
            for track in raw_tracking.tracked_obstacles
            if track.matched_previous_track
            and track.point_count >= MOTION_CALIBRATION_MINIMUM_TRACK_POINTS
            and track.observation_count >= MOTION_CALIBRATION_MINIMUM_OBSERVATIONS
        )
        accepted_tracks = tuple(
            track
            for track in tracking.tracked_obstacles
            if track.matched_previous_track
            and track.point_count >= MOTION_CALIBRATION_MINIMUM_TRACK_POINTS
            and track.observation_count >= MOTION_CALIBRATION_MINIMUM_OBSERVATIONS
        )
        published_motion_sample_count += len(accepted_tracks)
        published_motion_count += sum(
            math.hypot(*track.velocity_world_xy) > 1e-9
            for track in accepted_tracks
        )
    motion_calibration = calibrate_stationary_motion_threshold(
        np.asarray(raw_stationary_speeds, dtype=np.float64)
    )

    detector = make_detector(model_path)
    visual_mass_estimator = make_visual_mass_estimator()
    detections = detector.detect(
        image_rgb,
        source_frame_id=camera_info.read_index,
        source_timestamp_seconds=camera_info.simulation_time_seconds,
    )
    fusion = CameraLidarSemanticFusion(
        CameraLidarFusionConfig(
            minimum_visible_points=2,
            # The 32-beam LiDAR primarily hits object silhouettes. Require
            # several independent mask hits while allowing the majority of a
            # track's visible returns to lie outside the segmentation mask.
            minimum_mask_hits=FUSION_MINIMUM_MASK_HITS,
            minimum_overlap_fraction=FUSION_MINIMUM_OVERLAP_FRACTION,
            mask_dilation_radius_pixels=FUSION_MASK_DILATION_PIXELS,
            projected_median_box_margin_pixels=(
                FUSION_PROJECTED_MEDIAN_BOX_MARGIN_PIXELS
            ),
            ambiguity_score_margin=0.05,
        )
    ).fuse(
        detection_frame=detections,
        obstacle_points_source_xyz=obstacle_points_3d,
        tracking_result=tracking,
        camera_from_source=camera_from_robot,
        calibration=calibration,
    )
    waypoint_world_xy = (
        float(initial_pose[0] + math.cos(initial_pose[2])),
        float(initial_pose[1] + math.sin(initial_pose[2])),
    )
    sensor_timestamp_seconds = lidar_info.sensor_timestamp_nanoseconds * 1e-9
    fall_surface_results = estimate_fall_surfaces_by_track(
        scene_points_robot_xyz=points_robot,
        obstacle_points_robot_xyz=obstacle_points_3d,
        tracking=tracking,
        timestamp_seconds=sensor_timestamp_seconds,
    )
    physical_attributes = build_tracking_physical_attributes(
        tracking_result=tracking,
        label_result=fusion.semantic_labels,
        sensor_timestamp_seconds=(
            lidar_info.sensor_timestamp_nanoseconds * 1e-9
        ),
        context=LivePhysicalEvidenceContext(
            obstacle_points_robot_xyz=obstacle_points_3d,
            floor_height_robot_m=floor_height,
            robot_position_world_xy=(
                float(initial_pose[0]),
                float(initial_pose[1]),
            ),
            waypoint_world_xy=waypoint_world_xy,
            measured_robot_velocity_world_xy=(0.0, 0.0),
        ),
        associations=fusion.associations,
        fall_surface_provider=(
            lambda track_id: fall_surface_results[track_id].evidence
        ),
    )
    physical_attributes = attach_visual_mass_estimates(
        physical_attributes,
        image_rgb=image_rgb,
        detections=detections,
        fusion=fusion,
        estimator=visual_mass_estimator,
        timestamp_seconds=lidar_info.sensor_timestamp_nanoseconds * 1e-9,
    )
    visual_mass_diagnostic = None
    visual_mass_diagnostic_path = None
    if visual_mass_estimator is not None:
        visual_mass_diagnostic = evaluate_live_visual_mass_dataflow(
            physical_attributes,
            checkpoint_sha256=visual_mass_estimator.checkpoint_sha256,
        )
        visual_mass_diagnostic_path = OUTPUT_DIRECTORY / "visual_mass_diagnostic.json"
        visual_mass_diagnostic.write_json(visual_mass_diagnostic_path)
    physical_profile_provider = PhysicalAttributeCRARiskProfileProvider(
        {item.track_id: item for item in physical_attributes},
        timestamp_seconds=sensor_timestamp_seconds,
    )
    cra_policy_result = resolve_cra_track_policy(
        fusion.semantic_labels.semantic_obstacles,
        profile_provider=physical_profile_provider,
        policy=CRASafeSAGEPolicy(),
    )
    raw_path, semantic_path, fusion_path = save_images(
        image_rgb=image_rgb,
        detections=detections,
        fusion=fusion,
    )
    controller.stop()
    app_utils.update_app(steps=5)
    final_pose = read_chassis_pose(loaded_scene.stage)
    app_utils.stop()
    print_report(
        model_path=model_path,
        camera_info=camera_info,
        lidar_info=lidar_info,
        waited_updates=waited_updates,
        input_point_count=points_lidar.shape[0],
        obstacle_points_3d=obstacle_points_3d,
        tracking=tracking,
        detections=detections,
        fusion=fusion,
        physical_attributes=physical_attributes,
        physical_profile_provider=physical_profile_provider,
        cra_policy_result=cra_policy_result,
        fall_surface_results=fall_surface_results,
        motion_calibration=motion_calibration,
        published_motion_count=published_motion_count,
        published_motion_sample_count=published_motion_sample_count,
        raw_path=raw_path,
        semantic_path=semantic_path,
        fusion_path=fusion_path,
        visual_mass_checkpoint=(
            visual_mass_estimator.config.checkpoint_path
            if visual_mass_estimator is not None
            else None
        ),
        visual_mass_diagnostic=visual_mass_diagnostic,
        visual_mass_diagnostic_path=visual_mass_diagnostic_path,
        initial_pose=initial_pose,
        final_pose=final_pose,
    )
    print("\nExpected image behavior:")
    print("  - The raw image matches Nova Carter's front Hawk left camera.")
    print("  - The semantic overlay contains YOLO masks and boxes.")
    print("  - The fusion overlay adds projected LiDAR dots and T<number> labels.")
    print("  - Green LiDAR dots belong to a semantically matched track.")
    print("  - White dots are geometric returns with unknown semantics.")
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
        print("ERROR DURING CAMERA--LIDAR SEMANTIC FUSION DEMO")
        print("=" * 78)
        print(error)
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()

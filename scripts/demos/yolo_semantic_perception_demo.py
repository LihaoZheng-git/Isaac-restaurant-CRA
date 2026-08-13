"""Validate YOLO semantic perception on Nova Carter's real RGB camera.

This branch deliberately validates vision before camera--LiDAR fusion or
safety-control integration.

Stage A - stationary camera acquisition:

1. Load the existing restaurant robot scene.
2. Start playback and initialize Nova Carter's articulation and front Hawk
   left camera.
3. Hold both drive wheels at zero velocity.
4. Capture several fresh RGB frames with camera metadata.

Stage B - model-independent semantic output:

1. Load external ``yolo11n-seg.pt`` weights through the optional Ultralytics
   adapter.
2. Run instance segmentation on each captured RGB frame.
3. Convert framework-specific boxes, classes, confidences, and masks into the
   project's immutable ``SemanticDetectionFrame`` representation.
4. Select the frame containing the richest detection result.

Stage C - visual and numerical validation:

1. Verify every mask and bounding box matches its source image.
2. Print class counts and per-instance details.
3. Save the selected raw RGB frame and a deterministic mask/box overlay.
4. Verify that Nova Carter remained stationary throughout the experiment.

No YOLO label reaches the LiDAR tracker, occupancy grid, Safe-SAGE field,
CBF, MPC, or wheel controller in this demo.  Camera--LiDAR association is the
next milestone after this visual-perception boundary is validated.

Run from the project root in the cloned YOLO environment:

    $env:YOLO_MODEL_PATH = "C:\\path\\to\\yolo11n-seg.pt"
    python -u scripts/demos/yolo_semantic_perception_demo.py

Playback starts automatically. Set ``ISAAC_SIM_HEADLESS=1`` for a headless
run, ``YOLO_AUTOCLOSE=1`` to close a rendered run after the report, or
``YOLO_DEVICE=cpu`` to disable CUDA inference.
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
AUTOCLOSE_AFTER_REPORT = environment_flag("YOLO_AUTOCLOSE")
RTX_HISTORY_TRANSFORM_COUNT = 20

# SimulationApp must exist before importing Isaac Sim runtime modules.
simulation_app = SimulationApp(
    {
        "headless": SIMULATION_HEADLESS,
        "extra_args": [
            (
                "--/rtx/scenedb/maxHistoryTransformCount="
                f"{RTX_HISTORY_TRANSFORM_COUNT}"
            ),
        ],
    }
)


import isaacsim.core.experimental.utils.app as app_utils
import numpy as np
from isaacsim.core.simulation_manager import SimulationManager
from PIL import Image

from robot.control.differential_drive import NovaCarterController
from robot.perception.semantic_detection import SemanticDetectionFrame
from robot.perception.visualization import (
    render_semantic_detection_overlay,
)
from robot.perception.yolo_detector import (
    YoloSegmentationConfig,
    YoloSegmentationDetector,
    rgb_to_bgr_uint8,
)
from robot.runtime.simulation_loop import (
    advance_playing_steps,
    wait_until_playing,
)
from robot.scene.loader import load_robot_scene
from robot.scene.transforms import get_prim_planar_pose
from robot.sensor.camera import CameraFrameInfo, NovaCarterCamera


# ---------------------------------------------------------------------------
# Controlled perception configuration
# ---------------------------------------------------------------------------


SETTLE_STEPS = 60
MAXIMUM_FIRST_FRAME_WAIT_STEPS = 300
CAPTURE_SAMPLE_COUNT = 5
CAPTURE_INTERVAL_STEPS = 6
MAXIMUM_ALLOWED_POSITION_DRIFT = 0.005
MAXIMUM_ALLOWED_YAW_DRIFT = math.radians(0.10)

YOLO_CONFIDENCE_THRESHOLD = 0.35
YOLO_IOU_THRESHOLD = 0.70
YOLO_IMAGE_SIZE = 640
YOLO_DEVICE = os.environ.get("YOLO_DEVICE", "0").strip() or "0"

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIRECTORY = (
    PROJECT_ROOT / "outputs" / "yolo_semantic_perception" / RUN_TIMESTAMP
)


@dataclass(frozen=True)
class DetectionSample:
    """One source RGB image and the semantic result derived from it."""

    image_rgb: np.ndarray
    camera_info: CameraFrameInfo
    detections: SemanticDetectionFrame


def resolve_model_path() -> Path:
    """Return an explicitly configured local segmentation checkpoint."""

    configured_path = os.environ.get("YOLO_MODEL_PATH", "").strip()
    if not configured_path:
        raise RuntimeError(
            "YOLO_MODEL_PATH is not set. Point it to the existing local "
            "yolo11n-seg.pt file before running this demo. Keeping model "
            "weights outside the repository prevents accidental commits "
            "and implicit downloads."
        )
    path = Path(configured_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"YOLO model weights were not found: {path}")
    return path


def read_chassis_pose(stage: object) -> np.ndarray:
    """Return the current world-frame ``(x, y, yaw)`` chassis pose."""

    x_position, y_position, yaw = get_prim_planar_pose(
        stage,
        ROBOT_BASE_PRIM_PATH,
    )
    return np.asarray([x_position, y_position, yaw], dtype=np.float64)


def wait_for_first_rgb_frame(
    camera: NovaCarterCamera,
    controller: NovaCarterController,
) -> tuple[np.ndarray, CameraFrameInfo]:
    """Wait for one valid rendered camera frame while commanding zero."""

    print("\nWaiting for the first rendered RGB frame...")
    completed_updates = 0
    last_error: RuntimeError | None = None
    while (
        completed_updates < MAXIMUM_FIRST_FRAME_WAIT_STEPS
        and simulation_app.is_running()
    ):
        if not app_utils.is_playing():
            simulation_app.update()
            continue
        controller.stop()
        simulation_app.update()
        completed_updates += 1
        try:
            image_rgb = camera.get_rgb()
        except RuntimeError as error:
            if "No RGB frame is available yet" not in str(error):
                raise
            last_error = error
            continue
        print(
            "First RGB frame received after "
            f"{completed_updates} rendered update(s)."
        )
        return image_rgb, camera.get_frame_info()
    if not simulation_app.is_running():
        raise RuntimeError(
            "Isaac Sim closed before the first RGB frame arrived."
        )
    detail = f" Last camera error: {last_error}" if last_error else ""
    raise TimeoutError(
        "No RGB frame arrived after "
        f"{MAXIMUM_FIRST_FRAME_WAIT_STEPS} rendered updates.{detail}"
    )


def make_detector(model_path: Path) -> YoloSegmentationDetector:
    """Load the controlled YOLO11n segmentation adapter."""

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


def detect_frame(
    detector: YoloSegmentationDetector,
    image_rgb: np.ndarray,
    camera_info: CameraFrameInfo,
) -> DetectionSample:
    """Run one camera frame through the semantic adapter."""

    detections = detector.detect(
        image_rgb,
        source_frame_id=camera_info.read_index,
        source_timestamp_seconds=camera_info.simulation_time_seconds,
    )
    return DetectionSample(
        image_rgb=np.array(image_rgb, copy=True),
        camera_info=camera_info,
        detections=detections,
    )


def capture_detection_samples(
    *,
    detector: YoloSegmentationDetector,
    camera: NovaCarterCamera,
    controller: NovaCarterController,
    first_image_rgb: np.ndarray,
    first_camera_info: CameraFrameInfo,
) -> tuple[DetectionSample, ...]:
    """Capture fresh stationary frames and run semantic segmentation."""

    samples = [
        detect_frame(
            detector,
            first_image_rgb,
            first_camera_info,
        )
    ]
    print(
        "\nCapturing "
        f"{CAPTURE_SAMPLE_COUNT} stationary semantic sample(s)..."
    )
    print(
        "  sample=1 "
        f"detections={len(samples[0].detections.detections)} "
        "inference="
        f"{samples[0].detections.inference_wall_seconds * 1000.0:.1f} ms"
    )
    for sample_index in range(1, CAPTURE_SAMPLE_COUNT):
        if not advance_playing_steps(
            simulation_app,
            physics_steps=CAPTURE_INTERVAL_STEPS,
            on_step=controller.stop,
        ):
            raise RuntimeError(
                "Isaac Sim closed before semantic capture completed."
            )
        image_rgb = camera.get_rgb()
        sample = detect_frame(
            detector,
            image_rgb,
            camera.get_frame_info(),
        )
        samples.append(sample)
        print(
            f"  sample={sample_index + 1} "
            f"detections={len(sample.detections.detections)} "
            "inference="
            f"{sample.detections.inference_wall_seconds * 1000.0:.1f} ms"
        )
    return tuple(samples)


def select_richest_sample(
    samples: tuple[DetectionSample, ...],
) -> DetectionSample:
    """Select deterministically by count, then summed confidence."""

    if not samples:
        raise ValueError("samples cannot be empty.")
    return max(
        samples,
        key=lambda sample: (
            len(sample.detections.detections),
            sum(
                detection.confidence
                for detection in sample.detections.detections
            ),
            sample.camera_info.read_index,
        ),
    )


def save_validation_images(sample: DetectionSample) -> tuple[Path, Path]:
    """Save one source frame and its project-rendered semantic overlay."""

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    image_rgb_uint8 = np.ascontiguousarray(
        rgb_to_bgr_uint8(sample.image_rgb)[:, :, ::-1]
    )
    overlay_rgb = render_semantic_detection_overlay(
        sample.image_rgb,
        sample.detections,
    )
    raw_path = OUTPUT_DIRECTORY / "selected_camera_frame.png"
    overlay_path = OUTPUT_DIRECTORY / "selected_semantic_overlay.png"
    Image.fromarray(image_rgb_uint8).save(raw_path)
    Image.fromarray(overlay_rgb).save(overlay_path)
    return raw_path, overlay_path


def validate_detection_geometry(frame: SemanticDetectionFrame) -> bool:
    """Recheck the model-independent pixel geometry for the report."""

    expected_shape = (frame.image_height, frame.image_width)
    return all(
        detection.mask.shape == expected_shape
        and 0.0 <= detection.bounding_box.x_min
        < detection.bounding_box.x_max <= frame.image_width
        and 0.0 <= detection.bounding_box.y_min
        < detection.bounding_box.y_max <= frame.image_height
        for detection in frame.detections
    )


def wrapped_angle_difference(first: float, second: float) -> float:
    """Return the absolute wrapped difference between two yaw angles."""

    return abs(math.atan2(math.sin(first - second), math.cos(first - second)))


def print_report(
    *,
    model_path: Path,
    camera: NovaCarterCamera,
    samples: tuple[DetectionSample, ...],
    selected: DetectionSample,
    raw_path: Path,
    overlay_path: Path,
    initial_pose: np.ndarray,
    final_pose: np.ndarray,
) -> bool:
    """Print controlled perception results and return acceptance status."""

    inference_times = np.asarray(
        [sample.detections.inference_wall_seconds for sample in samples],
        dtype=np.float64,
    )
    position_drift = float(np.linalg.norm(final_pose[:2] - initial_pose[:2]))
    yaw_drift = wrapped_angle_difference(final_pose[2], initial_pose[2])
    camera_shape_valid = all(
        sample.image_rgb.ndim == 3
        and sample.image_rgb.shape[2] in (3, 4)
        for sample in samples
    )
    geometry_valid = all(
        validate_detection_geometry(sample.detections) for sample in samples
    )
    independent_output_valid = all(
        not detection.mask.flags.writeable
        for sample in samples
        for detection in sample.detections.detections
    )
    robot_stationary = (
        position_drift <= MAXIMUM_ALLOWED_POSITION_DRIFT
        and yaw_drift <= MAXIMUM_ALLOWED_YAW_DRIFT
    )
    detection_observed = any(
        sample.detections.detections for sample in samples
    )
    passed = (
        camera_shape_valid
        and geometry_valid
        and independent_output_valid
        and robot_stationary
        and detection_observed
    )

    print("\n" + "=" * 78)
    print("YOLO SEMANTIC PERCEPTION VALIDATION REPORT")
    print("=" * 78)
    print("\nModel configuration:")
    print(f"  Weights:                    {model_path}")
    print(f"  Model name:                 {selected.detections.model_name}")
    print(f"  Device:                     {YOLO_DEVICE}")
    print(f"  Confidence threshold:       {YOLO_CONFIDENCE_THRESHOLD:.2f}")
    print(f"  IoU threshold:              {YOLO_IOU_THRESHOLD:.2f}")
    print(f"  Inference image size:       {YOLO_IMAGE_SIZE}")
    print("\nCamera and sampling:")
    print(f"  Camera:                     {camera.camera_name}")
    print(f"  Camera prim:                {camera.camera_prim_path}")
    print(
        "  Source image shape:         "
        f"{selected.image_rgb.shape}"
    )
    print(f"  Captured semantic samples:  {len(samples)}")
    print(
        "  Selected camera read index: "
        f"{selected.camera_info.read_index}"
    )
    print(
        "  Mean inference wall time:   "
        f"{float(np.mean(inference_times)) * 1000.0:.1f} ms"
    )
    print(
        "  Maximum inference time:     "
        f"{float(np.max(inference_times)) * 1000.0:.1f} ms"
    )
    print("\nSelected semantic result:")
    print(
        "  Detected instances:         "
        f"{len(selected.detections.detections)}"
    )
    print(
        "  Class counts:               "
        f"{dict(selected.detections.class_counts)}"
    )
    if selected.detections.detections:
        print("  Instance details:")
    for index, detection in enumerate(selected.detections.detections):
        box = detection.bounding_box
        print(
            f"    [{index}] {detection.semantic_class:<16} "
            f"confidence={detection.confidence:.3f} "
            "box=("
            f"{box.x_min:.1f}, {box.y_min:.1f}, "
            f"{box.x_max:.1f}, {box.y_max:.1f}) "
            f"mask_pixels={detection.mask_pixel_count}"
        )
    print("\nStationary-control audit:")
    print(f"  Position drift:             {position_drift:.6f} m")
    print(f"  Yaw drift:                  {math.degrees(yaw_drift):.4f} degrees")
    print("  Commands sent to wheels:    zero only")
    print("\nControlled checks:")
    print(f"  Camera RGB shape valid:      {camera_shape_valid}")
    print(f"  Detection geometry valid:   {geometry_valid}")
    print(f"  Project owns immutable masks:{independent_output_valid}")
    print(f"  At least one instance seen: {detection_observed}")
    print(f"  Robot remained stationary:  {robot_stationary}")
    print("\nSaved validation artifacts:")
    print(f"  Raw camera frame:           {raw_path}")
    print(f"  Semantic overlay:           {overlay_path}")
    print("\nControlled-data boundary:")
    print("  YOLO used only the real front Hawk RGB image.")
    print("  No scene prim name or simulator ground truth supplied a label.")
    print("  No label reached LiDAR tracking, Safe-SAGE, CBF, MPC, or wheels.")
    print("\nInterpretation:")
    if passed:
        print(
            "  [PASS] Real camera images produced valid project-owned "
            "semantic instance masks while Nova Carter remained stationary."
        )
    else:
        print(
            "  [WARNING] The semantic perception demo did not satisfy every "
            "acceptance check. Inspect the checks and saved images before "
            "camera--LiDAR fusion."
        )
    print("=" * 78)
    return passed


def main() -> None:
    """Run the complete stationary visual-perception validation."""

    print("\n" + "=" * 78)
    print("YOLO SEMANTIC PERCEPTION - STATIONARY ISAAC SIM VALIDATION")
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
    print("Play detected. Initializing physics...")
    app_utils.update_app(steps=10)
    controller.verify_articulation()
    controller.print_configuration()

    camera = NovaCarterCamera()
    camera.initialize()
    print(
        "\nHolding Nova Carter still for "
        f"{SETTLE_STEPS} physics steps..."
    )
    if not advance_playing_steps(
        simulation_app,
        physics_steps=SETTLE_STEPS,
        on_step=controller.stop,
    ):
        return
    initial_pose = read_chassis_pose(loaded_scene.stage)
    first_image, first_info = wait_for_first_rgb_frame(camera, controller)
    detector = make_detector(model_path)
    samples = capture_detection_samples(
        detector=detector,
        camera=camera,
        controller=controller,
        first_image_rgb=first_image,
        first_camera_info=first_info,
    )
    selected = select_richest_sample(samples)
    raw_path, overlay_path = save_validation_images(selected)
    controller.stop()
    app_utils.update_app(steps=5)
    final_pose = read_chassis_pose(loaded_scene.stage)
    app_utils.stop()
    print_report(
        model_path=model_path,
        camera=camera,
        samples=samples,
        selected=selected,
        raw_path=raw_path,
        overlay_path=overlay_path,
        initial_pose=initial_pose,
        final_pose=final_pose,
    )
    print("\nExpected scene and image behavior:")
    print("  - Nova Carter remains stationary in the restaurant.")
    print("  - The raw PNG matches the front Hawk left-camera view.")
    print("  - The overlay PNG shows colored masks and boxes on detections.")
    print("  - False or missed classes are model observations, not ground truth.")
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
        print("ERROR DURING YOLO SEMANTIC PERCEPTION DEMO")
        print("=" * 78)
        print(error)
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()

"""Observe a sphere, compute its CRA policy, then optionally drive past it.

The default uses actual camera/LiDAR evidence and the existing estimators,
including conservative CRA uncertainty handling. Unknown factors remain unknown;
the resulting phi is used unchanged by the controller, even if it saturates.
This validates uncertainty-aware navigation, not complete perception accuracy.
The older supplied-profile comparison requires --oracle-comparison explicitly.
No restaurant asset is saved or enriched by this session-layer fixture.

Configure YOLO_MODEL_PATH, VISUAL_MASS_CHECKPOINT and
CONTACT_SEGMENTATION_CHECKPOINT with the existing local weights. Run stationary
first, then add --motion to execute the assessed policy on the simulated wheels.
"""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--motion", action="store_true")
parser.add_argument("--motion-maximum-seconds", type=float, default=30.0)
parser.add_argument("--oracle-comparison", action="store_true")
parser.add_argument("--object-plume-policy", action="store_true",
                    help="validate a stationary object-level plume window using this controlled clean fixture")
parser.add_argument("--plume-control", choices=("none", "intermittent", "persistent"), default="none",
                    help="rendered RGB fixture control, never supplied as a thermal score")
parser.add_argument("--contact-control", choices=("round", "protrusion"), default="round",
                    help="rendered/collidable geometry control; no supplied contact score")
parser.add_argument("--seed-color-outline", action="store_true",
                    help="explicit uniform-color separated-object RGB refinement domain")
parser.add_argument("--spherical-body-prior", action="store_true",
                    help="allow a detector-segmented spherical body; density must cover the small-appendage assembly")
parser.add_argument("--ground-continuity-prior", action="store_true",
                    help="explicit bounded flat-floor continuity assumption; result remains PRIOR")
parser.add_argument("--ground-prior-maximum-extrapolation-m", type=float, default=1.5)
parser.add_argument("--require-four-factor-readiness", action="store_true",
                    help="require all factors in the stated sensor/prior-assisted mode before accepting the test")
parser.add_argument("--grid-resolution", type=float, default=0.10)
parser.add_argument("--yaw-slices", type=int, default=16)
parser.add_argument("--object-radius", type=float, default=0.25,
                    help="authored fixture radius, never supplied to estimators")
parser.add_argument("--object-appearance", choices=("glossy", "matte"), default="glossy",
                    help="rendered fixture appearance control, never supplied to segmentation or mass inference")
parser.add_argument("--support-drop-m", type=float, default=0.0,
                    help="authored platform top above floor; estimator never receives this height")
parser.add_argument("--support-width-m", type=float, default=0.7)
parser.add_argument("--instance-mask-geometry", action="store_true",
                    help="estimate the camera-matched instance separately; retain remaining LiDAR occupancy with unknown risk")
parser.add_argument("--instance-mask-margin-pixels", type=int, default=0,
                    help="geometry selection margin; association retains its separate six-pixel tolerance")
parser.add_argument("--chromatic-instance-proposal", action="store_true",
                    help="explicit separated uniformly chromatic object/contrasting-background RGB domain; class stays unknown")
parser.add_argument("--sam-instance-proposal", action="store_true",
                    help="LiDAR-prompted SAM 2.1 RGB mask; unknown class; SAM2_CHECKPOINT must be an audited local Tiny checkpoint")
parser.add_argument("--category-free-visual-mass", action="store_true",
                    help="explicit SAM-associated RGB/LiDAR network extrapolation without class or density input")
parser.add_argument("--worst-case-fall-interaction-prior", action="store_true",
                    help="explicit harmful-fall interaction prior, upper bound one, independent of route overlap")
parser.add_argument("--fall-landing-maximum-offset-m", type=float, default=2.0)
parser.add_argument("--sphere-density-prior-kg-m3", type=float, nargs=3,
                    metavar=("NOMINAL", "LOWER", "UPPER"),
                    help="explicit effective bulk-density prior for accepted sphere fits")
args = parser.parse_args()
if not math.isfinite(args.motion_maximum_seconds) or not 0 < args.motion_maximum_seconds <= 60:
    parser.error("motion duration must lie in (0, 60] seconds")
if not math.isfinite(args.object_radius) or not 0.08 <= args.object_radius <= 0.5:
    parser.error("--object-radius must be within [0.08, 0.5] m")
if (not math.isfinite(args.support_drop_m) or not 0 <= args.support_drop_m <= .4
        or not math.isfinite(args.support_width_m) or not .5 <= args.support_width_m <= 1.0
        or not math.isfinite(args.fall_landing_maximum_offset_m)
        or not .12 < args.fall_landing_maximum_offset_m <= 3.0):
    parser.error("invalid bounded support/drop/landing geometry")
if args.support_drop_m and (not args.instance_mask_geometry or args.oracle_comparison
        or not args.worst_case_fall_interaction_prior or args.object_radius > args.support_width_m / 2):
    parser.error("raised test requires instance geometry, explicit fall-interaction prior, and a contained ball")
if args.support_drop_m and (args.contact_control != "round" or args.plume_control != "none"):
    parser.error("raised potential fixture requires unchanged round/no-plume controls")
if args.instance_mask_geometry and args.oracle_comparison:
    parser.error("instance geometry requires observed mode")
if args.chromatic_instance_proposal and not args.instance_mask_geometry:
    parser.error("chromatic proposal requires instance-mask geometry")
if args.sam_instance_proposal and (not args.instance_mask_geometry or args.chromatic_instance_proposal or args.seed_color_outline):
    parser.error("SAM proposal requires instance-mask geometry and excludes chromatic/seed-color outline modes")
if args.category_free_visual_mass and (not args.sam_instance_proposal or args.sphere_density_prior_kg_m3 is not None):
    parser.error("category-free visual mass requires SAM and excludes density-prior mass fallback")
if not 0 <= args.instance_mask_margin_pixels <= 6:
    parser.error("instance mask margin must lie within [0, 6] pixels")
if (not math.isfinite(args.ground_prior_maximum_extrapolation_m)
        or not 0 < args.ground_prior_maximum_extrapolation_m <= 2.0):
    parser.error("ground prior extrapolation must lie in (0, 2.0] m")
if args.oracle_comparison and (args.require_four_factor_readiness or args.ground_continuity_prior):
    parser.error("ground/four-factor observed readiness is separate from --oracle-comparison")
if args.plume_control != "none" and (not args.object_plume_policy or args.oracle_comparison):
    parser.error("plume controls require the observed --object-plume-policy path")
if args.contact_control != "round" and (args.oracle_comparison or args.plume_control != "none"):
    parser.error("contact controls require observed mode and no simultaneous plume intervention")
if args.spherical_body_prior and (args.sphere_density_prior_kg_m3 is None or not args.seed_color_outline):
    parser.error("spherical body prior requires density and seed-color outline opt-ins")
if args.oracle_comparison and args.seed_color_outline:
    parser.error("RGB outline refinement requires observed mode")
if args.sphere_density_prior_kg_m3 is not None:
    nominal, lower, upper = args.sphere_density_prior_kg_m3
    if not all(math.isfinite(value) for value in (nominal, lower, upper)) or not 0 < lower <= nominal <= upper:
        parser.error("density prior must satisfy 0 < LOWER <= NOMINAL <= UPPER")
args.output.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("WARP_CACHE_PATH", str(args.output / "warp"))

if not args.oracle_comparison:
    # Resolve the existing vision environment's torch/torchvision pair before
    # SimulationApp adds its bundled packages to sys.path. No install or
    # dependency replacement occurs; mixing those two environments is invalid.
    from robot.perception.visual_mass_runtime import (
        VisualMassRuntimeConfig, VisualMassRuntimeEstimator,
    )

# Reuse the validated scene, footprint, PDE and controller configurations.
# This creates SimulationApp once; imported main functions do not execute.
from scripts.demos import safe_sage_unicycle_mpc_motion_demo as existing

import isaacsim.core.experimental.utils.app as app_utils
from isaacsim.core.simulation_manager import SimulationManager
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import numpy as np
import omni.timeline
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from robot.control.differential_drive import NovaCarterController
from robot.evaluation.factor_specific_cra import diagnose_factor_specific_cra
from robot.evaluation.cra_experiments import EvidenceMode, RestaurantScenario
from robot.evaluation.perception_readiness import evaluate_perception_readiness, runtime_factor_evidence_mode
from robot.evaluation.simple_object_test import (
    ControlledObjectInputs,
    build_assessed_object_field,
    controlled_object_adapter,
    disk_footprint_clearance,
    run_bounded_pass_by,
)
from robot.perception.fall_height import estimate_fall_height
from robot.perception.camera_lidar_fusion import (
    CameraLidarFusionConfig, CameraLidarSemanticFusion,
    project_fisheye_polynomial_points,
)
from robot.perception.contact_detection_adapter import associate_contact_parts_to_objects
from robot.perception.lidar_fall_surfaces import LidarFallSurfaceConfig, estimate_lidar_fall_surfaces
from robot.perception.instance_lidar_geometry import masked_track_return_selection, propose_chromatic_instance
from robot.perception.sam_instance_segmentation import SamInstanceConfig, SamInstanceSegmenter, SAM_INSTANCE_SOURCE
from robot.perception.ground_support import GroundSupportConfig, estimate_spherical_ground_support
from robot.perception.motion_attributes import RouteMotionContext, estimate_track_motion
from robot.perception.physical_attributes import EstimateStatus, PhysicalAttributes, ScalarEstimate
from robot.perception.physical_geometry import estimate_track_geometry
from robot.perception.specialist_track_evidence import (
    SpecialistAssociationFrame, build_track_shape_contact_proxies, build_track_visual_contact_proxies,
)
from robot.perception.spherical_mass import estimate_spherical_geometry, estimate_spherical_mass
from robot.perception.seed_color_outline import refine_seed_color_outline
from robot.perception.object_plume_evidence import (
    OBJECT_PLUME_BASIS, ObjectPlumeObservation, estimate_object_plume_evidence,
)
from robot.perception.thermal_interaction import estimate_thermal_route_interaction
from robot.perception.visual_shape_contact import SHAPE_CONTACT_SOURCE, SHAPE_CONTACT_BASIS
from robot.perception.visualization import render_camera_lidar_fusion_overlay
from robot.perception.yolo_detector import YoloSegmentationConfig, YoloSegmentationDetector
from robot.risk.physical_factor_adapter import PhysicalFactorAdapterConfig, adapt_physical_factors
from robot.safety.cra_safe_sage_policy import CRASafeSAGEPolicy
from robot.safety.class_safety_policy import ClassSafetyParameters
from robot.safety.local_occupancy_grid import build_local_occupancy_grid
from robot.safety.obstacle_detection import (
    ObstacleDetectionConfig,
    detect_obstacle_points_3d,
)
from robot.safety.obstacle_tracking import ObstacleTracker, ObstacleTrackerConfig
from robot.safety.safe_sage_configuration_space import RectangularRobotFootprint
from robot.safety.safe_sage_field_pipeline import build_safe_sage_field_pipeline
from robot.safety.safe_sage_unicycle_mpc import SafeSAGEUnicycleMPC
from robot.safety.safe_sage_guidance_policy import (
    SafeSAGEGuidanceParameters,
    PassingSide,
)
from robot.safety.semantic_obstacle import SemanticObstacle
from robot.scene.loader import load_robot_scene
from robot.scene.camera_calibration import read_fisheye_polynomial_calibration
from robot.scene.transforms import compute_relative_prim_transform
from robot.sensor.camera import NovaCarterCamera
from robot.sensor.lidar import NovaCarterLidar
from robot.sensor.lidar_geometry import transform_points

stationary = existing.stationary
simulation_app = existing.simulation_app
RADIUS_M = args.object_radius
CENTER_WORLD_XY = np.array(
    [
        stationary.ROBOT_START_POSITION_WORLD[0] + 1.30,
        stationary.ROBOT_START_POSITION_WORLD[1],
    ]
)
FIXTURE_PATH = "/World/CRASimpleObject/Sphere"
GOAL_XY = (2.50, 0.0)
PROFILE_INPUTS = {
    "controlled_low": ControlledObjectInputs(),
    "controlled_medium": ControlledObjectInputs(contact_proxy=0.40),
    "controlled_high": ControlledObjectInputs(contact_proxy=0.80),
}


def save_json(name, payload):
    (args.output / name).write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def local_state(stage, anchor):
    pose = stationary.read_robot_pose(stage)
    xy, yaw = existing.motion.current_pose_in_field_frame(pose, anchor_pose=anchor)
    return np.array([*xy, yaw])


def refine_frame_target(rgb, frame, index):
    result = refine_seed_color_outline(rgb, frame.detections[index],
        uniform_color_domain_validated=True)
    updated = list(frame.detections)
    if result.detection is not None:
        updated[index] = result.detection
    return replace(frame, detections=tuple(updated),
        model_name=f"{frame.model_name}+{result.source}"), result


def append_chromatic_proposal(rgb, frame, projection, ids, track_id):
    result = propose_chromatic_instance(rgb, ids, projection, track_id=track_id,
        separated_chromatic_domain_validated=True)
    if result.detection is None:
        raise RuntimeError(f"chromatic instance proposal rejected: {result.rejection_reasons}")
    mask = result.detection.mask
    kept = tuple(item for item in frame.detections
        if np.count_nonzero(item.mask & mask) / max(1, min(item.mask.sum(), mask.sum())) < .5)
    return replace(frame, detections=(result.detection, *kept),
        model_name=f"{frame.model_name}+{result.source}")


def append_sam_proposal(segmenter, rgb, frame, projection, ids, track_id):
    segmenter.set_image(rgb)
    result = segmenter.propose(ids, projection, track_id=track_id)
    save_json(f"sam-mask-frame-{frame.source_frame_id}.json", {
        "source": result.source, "prompt_xy": result.prompt_xy,
        "model_quality": result.model_quality,
        "projected_track_points": result.projected_track_points,
        "mask_track_hits": result.mask_track_hits,
        "rejection_reasons": result.rejection_reasons,
        "image_encoder_wall_seconds": result.image_encoder_wall_seconds,
        "decoder_wall_seconds": result.inference_wall_seconds,
        "checkpoint_sha256": segmenter.checkpoint_sha256,
        "class_or_density_input": False, "authored_prompt_or_mask": False,
        "prior_detection_mask_overlaps": [
            {"class": item.semantic_class, "confidence": item.confidence,
             "overlap_fraction": float(np.count_nonzero(item.mask & result.detection.mask)
                / max(1, min(item.mask.sum(), result.detection.mask.sum())))}
            for item in frame.detections] if result.detection is not None else [],
    })
    if result.detection is None:
        raise RuntimeError(f"SAM proposal rejected: {result.rejection_reasons}")
    mask = result.detection.mask
    np.save(args.output / f"sam-mask-frame-{frame.source_frame_id}.npy", mask, allow_pickle=False)
    kept = tuple(item for item in frame.detections
        if np.count_nonzero(item.mask & mask) / max(1, min(item.mask.sum(), mask.sum())) < .5)
    return replace(frame, detections=(result.detection, *kept),
        model_name=f"{frame.model_name}+{result.source}")


def main():
    detector = contact_detector = mass_estimator = None
    sam_segmenter = None
    model_artifacts = {}
    if not args.oracle_comparison:
        # Explicit existing paths prevent automatic weight downloads. Keep the
        # same bounded model/domain contracts as the stationary fusion demo.
        paths = {}
        for name, variable in (
            ("objects", "YOLO_MODEL_PATH"),
            ("mass", "VISUAL_MASS_CHECKPOINT"),
            ("contact", "CONTACT_SEGMENTATION_CHECKPOINT"),
        ):
            configured = os.environ.get(variable, "").strip()
            if not configured or not Path(configured).is_file():
                raise FileNotFoundError(f"{variable} must identify existing local weights")
            paths[name] = Path(configured)
            model_artifacts[name] = {
                "path": str(paths[name].resolve()),
                "sha256": hashlib.sha256(paths[name].read_bytes()).hexdigest(),
            }
        device = os.environ.get("YOLO_DEVICE", "0")
        detector = YoloSegmentationDetector(YoloSegmentationConfig(
            model_path=paths["objects"], device=device,
        ))
        contact_detector = YoloSegmentationDetector(YoloSegmentationConfig(
            model_path=paths["contact"], device=device, confidence_threshold=0.25,
            allowed_classes=("exposed_edge",),
        ))
        detector.initialize()
        contact_detector.initialize()
        mass_estimator = VisualMassRuntimeEstimator(VisualMassRuntimeConfig(
            checkpoint_path=paths["mass"], device=device,
            allow_category_free_instances=args.category_free_visual_mass,
        ))
        if args.sam_instance_proposal:
            sam_path = Path(os.environ.get("SAM2_CHECKPOINT", ""))
            sam_segmenter = SamInstanceSegmenter(SamInstanceConfig(sam_path, device=device))
            model_artifacts["sam"] = {"path": str(sam_path.resolve()),
                                      "sha256": sam_segmenter.checkpoint_sha256}
    class_policy, guidance_policy, grid_config, field_config = (
        existing.create_fixed_field_configuration()
    )
    # Finer discretization avoids the coarse demo grid's false unsafe cells
    # during the sphere turn. The measured footprint/margin are unchanged.
    grid_config = replace(grid_config, resolution=args.grid_resolution)
    field_config = replace(
        field_config,
        configuration_space_config=replace(
            field_config.configuration_space_config, yaw_slice_count=args.yaw_slices
        ),
        h_full_config=replace(
            field_config.h_full_config, yaw_slice_count=args.yaw_slices
        ),
        guidance_field_config=replace(field_config.guidance_field_config,
            allow_dense_scene_arbitration=(args.instance_mask_geometry
                or field_config.guidance_field_config.allow_dense_scene_arbitration)),
    )
    loaded = load_robot_scene(
        scene_path=stationary.ROBOT_SCENE_PATH,
        simulation_app=simulation_app,
        robot_prim_path=stationary.ROBOT_PRIM_PATH,
        expected_meters_per_unit=stationary.EXPECTED_METERS_PER_UNIT,
    )
    stage = loaded.stage
    root_before = stage.GetRootLayer().ExportToString()
    scene_hash_before = hashlib.sha256(
        Path(stationary.ROBOT_SCENE_PATH).read_bytes()
    ).hexdigest()
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        sphere = UsdGeom.Sphere.Define(stage, FIXTURE_PATH)
        sphere.CreateRadiusAttr(RADIUS_M)
        sphere.AddTranslateOp().Set(Gf.Vec3d(*CENTER_WORLD_XY, RADIUS_M + args.support_drop_m))
        sphere.CreateDisplayColorAttr([Gf.Vec3f(0.15, 0.45, 0.95)])
        UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
        if args.chromatic_instance_proposal or args.object_appearance == "matte":
            # Matched matte appearance defines this bounded RGB test domain.
            # The estimator still discovers the color/outline from real RGB.
            material = UsdShade.Material.Define(stage, "/World/CRASimpleObject/MatteMaterial")
            shader = UsdShade.Shader.Define(stage, "/World/CRASimpleObject/MatteMaterial/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.15, .45, .95))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
            shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0))
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(material)
        if args.support_drop_m:
            support = UsdGeom.Cube.Define(stage, "/World/CRASimpleObject/Support")
            support.CreateSizeAttr(1.0)
            support.AddTranslateOp().Set(Gf.Vec3d(*CENTER_WORLD_XY, args.support_drop_m / 2))
            support.AddScaleOp().Set(Gf.Vec3f(args.support_width_m, args.support_width_m, args.support_drop_m))
            support.CreateDisplayColorAttr([Gf.Vec3f(.35, .35, .35)])
            UsdPhysics.CollisionAPI.Apply(support.GetPrim())
        if args.contact_control == "protrusion":
            tip = UsdGeom.Cone.Define(stage, "/World/CRASimpleObject/Tip")
            tip.CreateRadiusAttr(.055)
            tip.CreateHeightAttr(.22)
            tip.CreateAxisAttr("Y")
            tip.AddTranslateOp().Set(Gf.Vec3d(*(
                CENTER_WORLD_XY + np.array([0, -.32])), RADIUS_M))
            tip.AddRotateXOp().Set(180)
            tip.CreateDisplayColorAttr([Gf.Vec3f(.15, .45, .95)])
            UsdPhysics.CollisionAPI.Apply(tip.GetPrim())
        # Bounded visual puffs, like the accepted bowl fixture: no temperature,
        # aerosol physics or collision geometry. Begin hidden for the reference.
        puffs = []
        for index in range(4):
            puff = UsdGeom.Sphere.Define(stage, f"/World/CRASimpleObject/Plume/Puff{index}")
            puff.CreateRadiusAttr(.035 + .009 * index)
            puff.AddTranslateOp().Set(Gf.Vec3d(
                CENTER_WORLD_XY[0], CENTER_WORLD_XY[1] + .012 * (-1)**index,
                args.support_drop_m + 2 * RADIUS_M + .07 + .085 * index))
            puff.AddScaleOp().Set(Gf.Vec3f(1, 1, 1.4))
            puff.CreateDisplayColorAttr([Gf.Vec3f(.8, .8, .8)])
            puff.MakeInvisible()
            puffs.append(puff)
    root_unchanged_by_fixture = stage.GetRootLayer().ExportToString() == root_before
    controller = NovaCarterController(robot_prim_path=stationary.ROBOT_PRIM_PATH)
    SimulationManager.setup_simulation(
        dt=stationary.PHYSICS_DT, device=stationary.PHYSICS_DEVICE
    )
    omni.timeline.get_timeline_interface().play()
    app_utils.update_app(steps=10)
    controller.verify_articulation()
    stationary.position_robot_for_experiment(controller)
    for _ in range(stationary.SETTLE_STEPS):
        controller.stop()
        simulation_app.update()
    anchor = stationary.read_robot_pose(stage)
    robot_from_lidar = compute_relative_prim_transform(
        stage,
        source_prim_path=stationary.PANDAR_XT32_LIDAR_PRIM_PATH,
        target_prim_path=stationary.ROBOT_BASE_PRIM_PATH,
    )
    floor = stationary.estimate_floor_height_in_robot_frame(stage)
    detection = ObstacleDetectionConfig(
        minimum_height=floor + 0.05,
        maximum_height=floor + 1.0,
        minimum_range=0.15,
        maximum_range=5.0,
        forward_only=False,
        angular_bin_size_degrees=None,
    )
    lidar = NovaCarterLidar(lidar_prim_path=stationary.PANDAR_XT32_LIDAR_PRIM_PATH)
    lidar.initialize()
    camera = NovaCarterCamera()
    camera.initialize()
    tracker = ObstacleTracker(
        ObstacleTrackerConfig(
            cluster_radius=0.20,
            minimum_cluster_points=4,
            maximum_association_distance=0.50,
        )
    )
    last_timestamp = None
    samples = []
    for _ in range(400):
        controller.stop()
        simulation_app.update()
        try:
            raw = lidar.get_point_cloud()
            info = lidar.get_frame_info()
        except RuntimeError:
            continue
        if info.sensor_timestamp_nanoseconds == last_timestamp:
            continue
        last_timestamp = info.sensor_timestamp_nanoseconds
        scene_xyz = transform_points(raw, robot_from_lidar)
        obstacle_xyz = detect_obstacle_points_3d(scene_xyz, detection)
        pose = stationary.read_robot_pose(stage)
        tracking = tracker.update(
            obstacle_xyz[:, :2],
            robot_position_world_xy=np.array([pose.x, pose.y]),
            robot_yaw=pose.yaw,
            timestamp_seconds=last_timestamp * 1e-9,
        )
        target, error = existing.motion.find_controlled_proxy_track_world(
            tracking.tracked_obstacles, CENTER_WORLD_XY
        )
        if target is None or error > 0.40:
            continue
        samples.append((scene_xyz, obstacle_xyz, tracking, target, info, pose, error))
        if len(samples) == 12:
            break
    if len(samples) < 12:
        raise RuntimeError(
            f"sphere capture insufficient: {len(samples)} matched frames"
        )
    scene_xyz, obstacle_xyz, tracking, target, info, final_pose, association_error = (
        samples[-1]
    )
    timestamp = info.sensor_timestamp_nanoseconds * 1e-9
    image_rgb = camera.get_rgb()
    plt.imsave(args.output / "camera.png", image_rgb)
    camera_info = camera.get_frame_info()
    camera_lidar_age = abs(camera_info.simulation_time_seconds - timestamp)
    evidence_timestamp = max(timestamp, camera_info.simulation_time_seconds)
    np.savez_compressed(args.output / "preselection-capture.npz",
        scene_xyz=scene_xyz, obstacle_xyz=obstacle_xyz,
        point_track_ids=tracking.point_track_ids, target_track_id=target.track_id)
    fusion = detections = contact_association = target_association = None
    contact_proxies = {}
    observed_semantic = None
    model_invocations = {}
    sensor_dataflow_passed = True
    body_detection = outline_result = None
    if detector is not None:
        detections = detector.detect(
            image_rgb, source_frame_id=camera_info.read_index,
            source_timestamp_seconds=camera_info.simulation_time_seconds,
        )
        calibration = read_fisheye_polynomial_calibration(
            stage, camera_prim_path=camera.camera_prim_path,
            output_resolution=camera.resolution,
        )
        camera_from_robot = compute_relative_prim_transform(
            stage, source_prim_path=stationary.ROBOT_BASE_PRIM_PATH,
            target_prim_path=camera.camera_prim_path,
        )
        if args.chromatic_instance_proposal or args.sam_instance_proposal:
            proposal_projection = project_fisheye_polynomial_points(obstacle_xyz,
                camera_from_source=camera_from_robot, calibration=calibration)
            if args.sam_instance_proposal:
                save_json("yolo-before-sam.json", {"detections": [
                    {"class": item.semantic_class, "confidence": item.confidence,
                     "bounding_box": asdict(item.bounding_box)} for item in detections.detections]})
                detections = append_sam_proposal(sam_segmenter, image_rgb, detections,
                    proposal_projection, tracking.point_track_ids, target.track_id)
            else:
                detections = append_chromatic_proposal(image_rgb, detections,
                    proposal_projection, tracking.point_track_ids, target.track_id)
        fusion = CameraLidarSemanticFusion(CameraLidarFusionConfig(
            minimum_mask_hits=4, minimum_overlap_fraction=0.10,
            mask_dilation_radius_pixels=6, projected_median_box_margin_pixels=12,
            require_projected_median_in_box=not args.instance_mask_geometry,
        )).fuse(
            detection_frame=detections, obstacle_points_source_xyz=obstacle_xyz,
            tracking_result=tracking, camera_from_source=camera_from_robot,
            calibration=calibration,
        )
        target_association = next(
            item for item in fusion.associations if item.track_id == target.track_id
        )
        if target_association.is_matched:
            body_detection = detections.detections[target_association.detection_index]
            if args.seed_color_outline:
                detections, outline_result = refine_frame_target(image_rgb, detections,
                    target_association.detection_index)
                fusion = CameraLidarSemanticFusion(CameraLidarFusionConfig(
                    minimum_mask_hits=4, minimum_overlap_fraction=.10,
                    mask_dilation_radius_pixels=6, projected_median_box_margin_pixels=12,
                    require_projected_median_in_box=not args.instance_mask_geometry,
                )).fuse(detection_frame=detections, obstacle_points_source_xyz=obstacle_xyz,
                    tracking_result=tracking, camera_from_source=camera_from_robot, calibration=calibration)
                target_association = next(item for item in fusion.associations if item.track_id == target.track_id)
        observed_semantic = next(
            item for item in fusion.semantic_labels.semantic_obstacles
            if item.track_id == target.track_id
        )
        if args.chromatic_instance_proposal and target_association.is_matched:
            observed_semantic = replace(observed_semantic,
                label_source="lidar_seeded_chromatic_rgb_component")
        if args.sam_instance_proposal and target_association.is_matched:
            observed_semantic = replace(observed_semantic, label_source=SAM_INSTANCE_SOURCE)
        part_frame = contact_detector.detect(
            image_rgb, source_frame_id=camera_info.read_index,
            source_timestamp_seconds=camera_info.simulation_time_seconds,
        )
        contact_association = associate_contact_parts_to_objects(detections, part_frame)
        contact_proxies = build_track_visual_contact_proxies(
            SpecialistAssociationFrame(
                camera_info.read_index, camera_info.simulation_time_seconds,
                camera_info.height, camera_info.width, fusion.associations,
            ), contact_association.segmentations_by_object_detection_index,
        )
        part_contact_proxies = contact_proxies
        contact_proxies = build_track_shape_contact_proxies(
            detections, SpecialistAssociationFrame(
                camera_info.read_index, camera_info.simulation_time_seconds,
                camera_info.height, camera_info.width, fusion.associations,
            ), part_proxies=contact_proxies,
        )
        if outline_result is not None:
            contact_proxies = dict(contact_proxies)
            save_json("outline-refinement.json", {
                "source": outline_result.source, "rejection_reasons": outline_result.rejection_reasons,
                "added_pixels": outline_result.added_pixel_count,
                "seed_color_fraction": outline_result.seed_color_fraction,
                "domain": "explicit uniformly colored separated controlled fixture; touching similar colors unsupported",
                "authored_mask_or_color_used": False,
            })
            if outline_result.detection is None:
                missing_outline = ScalarEstimate.unavailable(unit="1",
                    source=SHAPE_CONTACT_SOURCE, timestamp_seconds=evidence_timestamp,
                    evidence=outline_result.rejection_reasons)
                contact_proxies[target.track_id] = part_contact_proxies.get(target.track_id, missing_outline)
            elif target.track_id in contact_proxies:
                proxy = contact_proxies[target.track_id]
                contact_proxies[target.track_id] = replace(proxy, evidence=(*proxy.evidence,
                    f"outline source={outline_result.source}; bounded RGB domain opt-in; added_pixels={outline_result.added_pixel_count}"))
        model_invocations = {
            "objects": {"invoked": True, "detection_count": len(detections.detections),
                        "backend": detections.model_name},
            "contact": {
                "invoked": True, "part_detection_count": len(part_frame.detections),
                "accepted_part_count": contact_association.diagnostics.accepted_part_count,
                "target_proxy_available": target.track_id in contact_proxies,
                "absence_is_negative_evidence": False,
                "whole_silhouette_proxy_enabled": True,
            },
            "thermal": {
                "invoked": False, "evidence_status": "unavailable",
                "reason": "bounded RGB plume estimator requires a clean reference and supported open-container source evidence",
            },
        }
        sensor_dataflow_passed = target_association.is_matched and camera_lidar_age <= 0.25
        plt.imsave(args.output / "fusion-overlay.png", render_camera_lidar_fusion_overlay(
            image_rgb, detections, fusion,
        ))
        save_json("vision.json", {
            "camera_info": asdict(camera_info),
            "camera_lidar_age_seconds": camera_lidar_age,
            "target_association": asdict(target_association),
            "associations": [asdict(item) for item in fusion.associations],
            "calibration": asdict(calibration),
            "camera_from_robot": np.asarray(camera_from_robot).tolist(),
            "models": model_artifacts, "invocations": model_invocations,
            "detections": [{
                "semantic_class": item.semantic_class, "confidence": item.confidence,
                "bounding_box": asdict(item.bounding_box),
            } for item in detections.detections],
            "contact_association": asdict(contact_association.diagnostics),
        })
        np.savez_compressed(
            args.output / "vision-masks.npz",
            object_masks=np.array([item.mask for item in detections.detections]),
            contact_masks=np.array([item.mask for item in part_frame.detections]),
        )
    selected = tracking.point_track_ids == target.track_id
    full_track_xyz = obstacle_xyz[selected]
    if args.instance_mask_geometry:
        if target_association is None or not target_association.is_matched:
            raise RuntimeError("instance geometry requires an unambiguous matched camera mask")
        selected = masked_track_return_selection(tracking.point_track_ids, fusion.projection,
            detections.detections[target_association.detection_index].mask,
            track_id=target.track_id, margin_pixels=args.instance_mask_margin_pixels)
    object_xyz = obstacle_xyz[selected]
    unassessed_points_anchor = (existing.motion.robot_points_in_anchor_frame(
        obstacle_xyz[~selected, :2], robot_pose=final_pose, anchor_pose=anchor)
        if args.instance_mask_geometry else None)
    spherical = estimate_spherical_geometry(object_xyz, timestamp_seconds=timestamp)
    full_spherical = spherical
    spherical_fit_mode = "camera_mask_selected_instance" if args.instance_mask_geometry else "full_associated_track"
    body_xyz = np.empty((0, 3))
    if (args.spherical_body_prior and body_detection is not None
            and outline_result is not None and outline_result.detection is not None
            and not spherical.radius_m.is_available):
        projection = fusion.projection
        xy = projection.pixel_indices_xy
        # Match the existing camera/LiDAR association's six-pixel tolerance.
        # Exact mask clipping loses curved boundary returns under calibration
        # and segmentation errors; the unchanged sphere residual gate remains.
        body_mask = cv2.dilate(body_detection.mask.astype(np.uint8),
            np.ones((13, 13), np.uint8)).astype(bool)
        np.savez_compressed(args.output / "body-selection-mask.npz",
            detector_mask=body_detection.mask, association_mask=body_mask)
        accepted = ((fusion.projected_track_ids == target.track_id)
                    & body_mask[xy[:, 1], xy[:, 0]])
        body_xyz = obstacle_xyz[projection.source_indices[accepted]]
        # A bounded body hypothesis, with full occupancy retained for control.
        if len(body_xyz) >= .75 * len(object_xyz):
            candidate = estimate_spherical_geometry(body_xyz, timestamp_seconds=timestamp)
            if candidate.radius_m.is_available:
                spherical = candidate
                spherical_fit_mode = "detector_body_with_assembly_density_prior"
        save_json("body-geometry-selection.json", {
            "full_track_fit": asdict(full_spherical), "selected_fit": asdict(spherical),
            "selected_mode": spherical_fit_mode, "camera_mask_margin_pixels": 6,
            "maximum_excluded_return_fraction": .25,
            "body_return_count": len(body_xyz), "full_track_return_count": len(object_xyz),
            "excluded_fraction_is_mass_bound": False,
            "whole_assembly_density_is_prior": True,
            "all_track_returns_retained_for_controller": True,
        })
    points_anchor = existing.motion.robot_points_in_anchor_frame(
        object_xyz[:, :2], robot_pose=final_pose, anchor_pose=anchor
    )
    geometry = estimate_track_geometry(
        obstacle_xyz,
        tracking_result=(replace(tracking, point_track_ids=np.where(selected, target.track_id, -1))
                         if args.instance_mask_geometry else tracking),
        track_id=target.track_id,
        floor_height_robot_m=floor,
        association=target_association,
    )
    surface_config = (LidarFallSurfaceConfig(support_xy_margin_m=.30,
        support_maximum_gap_m=.20, minimum_support_overhang_m=.05,
        landing_maximum_offset_m=args.fall_landing_maximum_offset_m)
        if args.instance_mask_geometry else None)
    surfaces = estimate_lidar_fall_surfaces(scene_xyz, object_xyz,
        timestamp_seconds=timestamp, config=surface_config)
    support_consistency = None
    if args.instance_mask_geometry and surfaces.is_available:
        support = surfaces.evidence.support_height_m
        support_consistency = bool(spherical.radius_m.is_available
            and spherical.center_xyz is not None
            and support.lower_bound <= spherical.center_xyz[2] - spherical.radius_m.lower_bound
            and support.upper_bound >= spherical.center_xyz[2] - spherical.radius_m.upper_bound)
    save_json("fall-surface-evidence.json", {
        "result": asdict(surfaces), "configuration": asdict(surface_config) if surface_config else None,
        "fitted_bottom_consistency_gate": support_consistency,
        "authored_drop_used_as_estimator_input": False,
        "scope": "observed local support and candidate landing bands; exact impact site unestablished",
    })
    if surfaces.is_available:
        geometry = replace(
            geometry,
            fall_height_m=estimate_fall_height(
                surfaces.evidence, timestamp_seconds=timestamp
            ),
        )
    motion = estimate_track_motion(
        target,
        timestamp_seconds=timestamp,
        route_context=RouteMotionContext(
            (final_pose.x, final_pose.y),
            (
                anchor.x + math.cos(anchor.yaw) * GOAL_XY[0] - math.sin(anchor.yaw) * GOAL_XY[1],
                anchor.y + math.sin(anchor.yaw) * GOAL_XY[0] + math.cos(anchor.yaw) * GOAL_XY[1],
            ),
            0.20,
        ),
    )
    base = PhysicalAttributes.unknown(
        track_id=target.track_id,
        timestamp_seconds=evidence_timestamp,
        semantic_class=observed_semantic.semantic_class if observed_semantic else "object",
        semantic_source=observed_semantic.label_source if observed_semantic else "controlled_track_selection_prior",
        semantic_confidence=observed_semantic.semantic_confidence if observed_semantic else 1.0,
    )
    sensor_attributes = replace(base, geometry=geometry, motion=motion)
    if mass_estimator is not None and target_association.is_matched:
        if args.category_free_visual_mass:
            mass = mass_estimator.estimate_instance(image_rgb, detections,
                target_association, geometry, track_id=target.track_id,
                timestamp_seconds=evidence_timestamp)
        else:
            mass = mass_estimator.estimate(
                image_rgb, detections.detections[target_association.detection_index],
                geometry, timestamp_seconds=evidence_timestamp,
            )
        learned_mass_available = mass.mass_kg.is_available
        if args.sphere_density_prior_kg_m3 is not None:
            nominal, lower, upper = args.sphere_density_prior_kg_m3
            density = ScalarEstimate(nominal, lower, upper, .35, "kg/m^3",
                "configured_effective_sphere_density_prior", evidence_timestamp,
                EstimateStatus.PRIOR, (
                    "explicit demonstration assumption; material, hollow fraction and contents are not measured",
                "same effective density interval for size comparison; not a universal ball prior",
                "detector-body mode uses whole-assembly mass per body-volume PRIOR, including unmodeled small appendages",
                ))
            spherical_mass = estimate_spherical_mass(spherical,
                effective_density_kg_m3=density, timestamp_seconds=evidence_timestamp)
            # Explicit bounded fallback only when the learned class model cannot
            # supply mass. Do not silently replace an existing learned estimate.
            if not mass.mass_kg.is_available:
                mass = spherical_mass
            save_json("sphere-size-evidence.json", {
                "geometry": asdict(spherical), "density": asdict(density),
                "mass": asdict(spherical_mass),
                "selected_as_runtime_mass": mass is spherical_mass,
                "authored_radius_used_as_estimator_input": False,
                "spherical_fit_mode": spherical_fit_mode,
                "body_return_count": len(body_xyz), "full_track_return_count": len(object_xyz),
            })
        sensor_attributes = replace(
            sensor_attributes, mass=mass,
            thermal=replace(base.thermal, heat_source_likelihood=ScalarEstimate.unavailable(
                unit="1", source="bounded_plume_prerequisites_unavailable",
                timestamp_seconds=evidence_timestamp,
                evidence=(model_invocations["thermal"]["reason"],),
            )),
        )
        model_invocations["mass"] = {
            "adapter_invoked": True,
            "prediction_available": learned_mass_available,
            "runtime_mass_available": mass.mass_kg.is_available,
            "runtime_mass_status": mass.mass_kg.status.value,
            "evidence": mass.mass_kg.evidence,
        }
        if args.category_free_visual_mass:
            from robot.perception.visual_mass_live_diagnostic import evaluate_live_visual_mass_dataflow
            evaluate_live_visual_mass_dataflow((sensor_attributes,),
                checkpoint_sha256=mass_estimator.checkpoint_sha256,
                instance_associations=(target_association,)).write_json(args.output / "live-mass-dataflow.json")
        save_json("vision-model-invocations.json", model_invocations)
    potential_interaction = None
    if args.worst_case_fall_interaction_prior:
        potential_interaction = ScalarEstimate(1, 0, 1, .5, "1",
            "controlled_worst_case_harmful_fall_interaction_prior", evidence_timestamp,
            EstimateStatus.PRIOR, ("worst-case consequence assumption; not inferred from route crossing",))
    ground_support = None
    ground_config = None
    if args.ground_continuity_prior:
        ground_config = GroundSupportConfig(maximum_extrapolation_m=args.ground_prior_maximum_extrapolation_m)
        continuity = ScalarEstimate(1, 1, 1, .5, "1",
            "controlled_flat_floor_continuity_prior", timestamp, EstimateStatus.PRIOR,
            ("controlled local flat-floor domain; unseen support/landing continuity is an assumption",))
        ground_support = estimate_spherical_ground_support(scene_xyz, spherical,
            timestamp_seconds=evidence_timestamp, scene_timestamp_seconds=timestamp,
            ground_continuity_prior=continuity,
            prior_scope="controlled simple-object flat-floor fixture; no unseen edge in bounded local neighborhood",
            config=ground_config)
        selected_ground = not surfaces.is_available and ground_support.fall_height_m.is_available
        if selected_ground:
            sensor_attributes = replace(sensor_attributes,
                geometry=replace(sensor_attributes.geometry, fall_height_m=ground_support.fall_height_m))
            potential_interaction = ground_support.potential_interaction_likelihood
        save_json("ground-support-evidence.json", {
            "evidence": asdict(ground_support), "selected_as_runtime_fall_evidence": selected_ground,
            "strict_lidar_surface_rejections": surfaces.rejection_reasons,
            "authored_height_used_as_estimator_input": False,
            "ground_continuity_prior": asdict(continuity),
            "configuration": asdict(ground_config),
        })
    thermal_interaction = None
    if args.object_plume_policy and detector is not None:
        reference_time = camera_info.simulation_time_seconds
        plume_observations = []
        visible_counts = {"none": (0, 0, 0), "intermittent": (3, 0, 3),
                          "persistent": (4, 4, 4)}[args.plume_control]
        for frame_index, count in enumerate(visible_counts):
            with Usd.EditContext(stage, stage.GetSessionLayer()):
                for index, puff in enumerate(puffs):
                    puff.MakeVisible() if index < count else puff.MakeInvisible()
            for _ in range(6):
                controller.stop()
                simulation_app.update()
            frame_rgb = camera.get_rgb()
            plt.imsave(args.output / f"plume-frame-{frame_index}.png", frame_rgb)
            frame_info = camera.get_frame_info()
            frame_detections = detector.detect(frame_rgb, source_frame_id=frame_info.read_index,
                source_timestamp_seconds=frame_info.simulation_time_seconds)
            if args.chromatic_instance_proposal:
                frame_detections = append_chromatic_proposal(frame_rgb, frame_detections,
                    proposal_projection, tracking.point_track_ids, target.track_id)
            if args.sam_instance_proposal:
                frame_detections = append_sam_proposal(sam_segmenter, frame_rgb, frame_detections,
                    proposal_projection, tracking.point_track_ids, target.track_id)
            frame_outline_ok = True
            if args.seed_color_outline:
                preliminary = CameraLidarSemanticFusion(CameraLidarFusionConfig(
                    minimum_mask_hits=4, minimum_overlap_fraction=.10,
                    mask_dilation_radius_pixels=6, projected_median_box_margin_pixels=12,
                )).fuse(detection_frame=frame_detections, obstacle_points_source_xyz=obstacle_xyz,
                    tracking_result=tracking, camera_from_source=camera_from_robot, calibration=calibration)
                link = next(item for item in preliminary.associations if item.track_id == target.track_id)
                frame_outline_ok = link.is_matched
                if link.is_matched:
                    frame_detections, frame_outline = refine_frame_target(frame_rgb, frame_detections, link.detection_index)
                    frame_outline_ok = frame_outline.detection is not None
            frame_fusion = CameraLidarSemanticFusion(CameraLidarFusionConfig(
                minimum_mask_hits=4, minimum_overlap_fraction=.10,
                mask_dilation_radius_pixels=6, projected_median_box_margin_pixels=12,
                require_projected_median_in_box=not args.instance_mask_geometry,
            )).fuse(detection_frame=frame_detections, obstacle_points_source_xyz=obstacle_xyz,
                tracking_result=tracking, camera_from_source=camera_from_robot, calibration=calibration)
            if not frame_outline_ok:
                frame_fusion = replace(frame_fusion, associations=tuple(
                    replace(item, status="outline_refinement_unavailable") if item.track_id == target.track_id else item
                    for item in frame_fusion.associations))
            current_pose = stationary.read_robot_pose(stage)
            camera_stationary = (math.hypot(current_pose.x-final_pose.x, current_pose.y-final_pose.y) < .005
                                 and abs(current_pose.yaw-final_pose.yaw) < .005)
            plume_observations.append(ObjectPlumeObservation(frame_rgb, frame_detections,
                SpecialistAssociationFrame(frame_info.read_index, frame_info.simulation_time_seconds,
                    frame_info.height, frame_info.width, frame_fusion.associations),
                timestamp, camera_stationary))
        evidence_timestamp = plume_observations[-1].detections.source_timestamp_seconds
        object_plumes = estimate_object_plume_evidence(image_rgb,
            reference_timestamp_seconds=reference_time, observations=tuple(plume_observations),
            timestamp_seconds=evidence_timestamp, clean_reference_validated=True,
            negative_validation_scope="controlled fixed-camera rendered-puff fixture; initially clean reference, same observed object; no temperature or aerosol-identity claim")
        save_json("object-plume-evidence.json", {
            "records": {str(track): asdict(value) for track, value in object_plumes.items()},
            "reference_timestamp": reference_time,
            "camera_stationary": all(item.camera_stationary for item in plume_observations),
            "tracking_geometry_frozen": True,
            "rendered_control": args.plume_control,
            "visible_puff_counts": visible_counts,
            "control_used_as_estimator_score": False,
            "maximum_camera_track_age_seconds": max(item.detections.source_timestamp_seconds-timestamp
                                                      for item in plume_observations),
        })
        np.savez_compressed(args.output / "plume-window.npz",
            rgb=np.stack([item.rgb for item in plume_observations]),
            timestamps=[item.detections.source_timestamp_seconds for item in plume_observations])
        record = object_plumes.get(target.track_id)
        if record is not None:
            sensor_attributes = replace(sensor_attributes, timestamp_seconds=evidence_timestamp,
                                         thermal=record.thermal)
            thermal_interaction = estimate_thermal_route_interaction(record.thermal,
                motion.route_crossing_likelihood, timestamp_seconds=evidence_timestamp)
        model_invocations["thermal"] = {
            "invoked": True, "backend": "fixed-camera reference RGB object plume proxy",
            "object_records": len(object_plumes),
            "target_evidence_available": record is not None and record.thermal.heat_source_likelihood.is_available,
            "temperature_available": False, "moving_camera_supported": False,
        }
        save_json("vision-model-invocations.json", model_invocations)
    visual_contact_proxy = contact_proxies.get(target.track_id)
    sensor_adapter = adapt_physical_factors(
        sensor_attributes, timestamp_seconds=evidence_timestamp,
        potential_interaction_likelihood=potential_interaction,
        visual_contact_proxy=visual_contact_proxy,
        visual_contact_basis=SHAPE_CONTACT_BASIS
        if visual_contact_proxy is not None and visual_contact_proxy.source == SHAPE_CONTACT_SOURCE
        else "visual_exposed_edge_proxy",
        contact_interaction_likelihood=motion.route_crossing_likelihood
        if visual_contact_proxy is not None else None,
        thermal_interaction_likelihood=thermal_interaction,
        thermal_basis=OBJECT_PLUME_BASIS if thermal_interaction is not None else "physical",
        config=PhysicalFactorAdapterConfig(allow_visual_contact_proxy=True,
                                           allow_visual_thermal_proxy=args.object_plume_policy),
    )
    sensor_policy = CRASafeSAGEPolicy().evaluate(sensor_adapter.profile)
    np.savez_compressed(
        args.output / "capture.npz",
        scene_xyz=scene_xyz,
        object_xyz=object_xyz,
        full_track_xyz=full_track_xyz,
        all_obstacle_xyz=obstacle_xyz,
        instance_selection=selected,
        body_xyz=body_xyz,
        points_anchor=points_anchor,
        timestamps=[
            sample[4].sensor_timestamp_nanoseconds * 1e-9 for sample in samples
        ],
        object_velocities=[sample[3].velocity_world_xy for sample in samples],
    )
    plt.imsave(args.output / "camera.png", image_rgb)
    drift = math.hypot(final_pose.x - anchor.x, final_pose.y - anchor.y)
    yaw_drift = abs(final_pose.yaw - anchor.yaw)
    stationary_passed = (
        drift <= 0.02
        and yaw_drift <= math.radians(1)
        and len(object_xyz) >= 4
        and association_error <= 0.40
        and root_unchanged_by_fixture
        and sensor_dataflow_passed
        and support_consistency is not False
    )
    # Known position is only a controlled track-selection and independent
    # collision-scoring input; no authored radius/center enters the field.
    delta = CENTER_WORLD_XY - np.array([anchor.x, anchor.y])
    c, s = math.cos(anchor.yaw), math.sin(anchor.yaw)
    sphere_center = np.array(
        [c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1]]
    )
    footprint = RectangularRobotFootprint(
        stationary.ROBOT_FRONT_EXTENT,
        stationary.ROBOT_REAR_EXTENT,
        stationary.ROBOT_LEFT_EXTENT,
        stationary.ROBOT_RIGHT_EXTENT,
        stationary.ROBOT_SAFETY_MARGIN,
    )
    scoring_disks = [(sphere_center, RADIUS_M)]
    if args.contact_control == "protrusion":
        # Contains the entire cone's XY projection, including its base corners.
        # Independent oracle monitor only; occupancy still comes from LiDAR.
        scoring_disks.append((sphere_center + np.array([-.32 * s, -.32 * c]), .125))
    if args.support_drop_m:
        # Circumscribes the complete authored platform footprint. Conservative
        # independent scoring only; the controller receives observed returns.
        scoring_disks.append((sphere_center, args.support_width_m / math.sqrt(2)))

    def fixture_clearance(state):
        return min(disk_footprint_clearance(state, center_xy=center,
            radius_m=radius, footprint=footprint) for center, radius in scoring_disks)
    semantic = observed_semantic or SemanticObstacle(
        target, "object", 1.0, "controlled_track_selection_prior"
    )
    fields, policies, factors, predictions, readiness = {}, {}, {}, {}, {}
    field_risk_cells = {}
    prior_readiness = evaluate_perception_readiness(
        RestaurantScenario("simple_object", "potential", "explicit prior-assisted physical cues",
                           EvidenceMode.PRIOR, 0), sensor_adapter)
    runtime_evidence_mode = runtime_factor_evidence_mode(sensor_adapter)
    runtime_readiness = evaluate_perception_readiness(
        RestaurantScenario("simple_object", "potential", "actual factor input provenance",
                           runtime_evidence_mode, 0), sensor_adapter)
    mpc, backup = existing.create_controller_stack(controller)
    # Match the geometry checks to the actual physics-step CBF rechecks.
    # The old frozen-field demo only checked MPC interval endpoints.
    mpc = SafeSAGEUnicycleMPC(
        mpc.paper_filter,
        replace(
            mpc.config,
            safety_substeps_per_interval=round(
                mpc.config.time_step / stationary.PHYSICS_DT
            ),
            integration_substeps_per_interval=round(
                mpc.config.time_step / stationary.PHYSICS_DT
            ),
            enable_progress_seeds=True,
            progress_seed_linear_fraction=0.5,
            # Prefix envelopes across 1,056 preceding wheel intervals cover
            # lookahead-position/yaw errors, including command reversals.
            configuration_position_padding_m=0.025,
            configuration_yaw_padding_radians=0.21,
            configuration_padding_initial_fraction=0.1,
        ),
    )
    # Stationary initial predictions never command the wheels.
    from robot.navigation.waypoint_controller import (
        RobotPose2D,
        Waypoint,
        WaypointController,
        WaypointControllerConfig,
    )

    waypoint = WaypointController(
        [Waypoint(*GOAL_XY)],
        config=WaypointControllerConfig(
            maximum_linear_velocity=0.20,
            maximum_angular_velocity=0.60,
            rotate_in_place_threshold_radians=math.radians(100),
            minimum_alignment_scale=0.25,
        ),
    )
    adapters = {"observed": sensor_adapter}
    if args.oracle_comparison:
        adapters = {
            name: controlled_object_adapter(
                inputs, track_id=target.track_id, timestamp_seconds=timestamp,
            ) for name, inputs in PROFILE_INPUTS.items()
        }
    for name, adapted in adapters.items():
        policy, field = build_assessed_object_field(
            points_anchor, adapted=adapted, semantic=semantic,
            class_policy=class_policy, guidance_policy=guidance_policy,
            grid_config=grid_config, field_config=field_config,
            unassessed_points_xy=unassessed_points_anchor,
        )
        readiness[name] = asdict(
            evaluate_perception_readiness(
                RestaurantScenario(
                    "simple_object",
                    "contact",
                    "oracle comparison" if args.oracle_comparison else "observed sphere, conservative unknown handling",
                    EvidenceMode.ORACLE if args.oracle_comparison else EvidenceMode.SENSOR,
                    0,
                ),
                adapted,
            )
        )
        fields[name] = field
        field_risk_cells[name] = {
            "assessed_phi_cells_per_yaw": [int(np.count_nonzero(grid.occupied_mask
                & np.isclose(grid.cell_risks, policy.assessment.phi)))
                for grid in field.configuration_space.occupancy_grids],
            "unknown_phi_one_cells_per_yaw": [int(np.count_nonzero(grid.occupied_mask
                & np.isclose(grid.cell_risks, 1.0)))
                for grid in field.configuration_space.occupancy_grids],
            "note": "overlapping cells use conservative arbitration; these counts are not object identity accuracy",
        }
        policies[name] = policy.as_dict()
        factors[name] = diagnose_factor_specific_cra(adapted).to_dict()
        np.savez_compressed(
            args.output / f"{name}-field.npz", h_full=field.h_full.values
        )
        nominal = waypoint.predict_commands(
            RobotPose2D(0.0, 0.0, 0.0),
            horizon_steps=mpc.config.horizon_steps,
            time_step=mpc.config.time_step,
        )
        planned = mpc.plan(
            np.zeros(3),
            np.array(
                [[item.linear_velocity, item.angular_velocity] for item in nominal]
            ),
            configuration_space=field.configuration_space,
            h_full=field.h_full,
            guidance_fields=field.guidance_fields,
            temporal_safety_derivative=0.0,
            preferred_lateral_direction_grid_y=-1.0,
        )
        predictions[name] = {
            "success": existing.predictive_result_is_valid(
                planned, constraint_tolerance=mpc.config.constraint_tolerance
            ),
            "status": planned.status,
            "minimum_margin": float(np.min(planned.dtcbf_margins)),
            "predicted_chassis_states": planned.predicted_chassis_states.tolist(),
        }
    if args.oracle_comparison:
        # A constant policy baseline shares CRA's flux range and lambda, so a
        # different legacy flux range or class-specific lambda cannot confound it.
        fixed_risk = 0.40
        fixed_guidance = SafeSAGEGuidanceParameters(
            ClassSafetyParameters(
                fixed_risk,
                CRASafeSAGEPolicy().config.normal_flux_magnitude_for_risk(fixed_risk),
            ),
            1.25,
            PassingSide.RIGHT,
        )
        fixed_grid = build_local_occupancy_grid(
            points_anchor,
            point_track_ids=np.full(len(points_anchor), target.track_id),
            semantic_obstacles=(semantic,),
            class_safety_policy=class_policy,
            config=grid_config,
            track_safety_parameters={target.track_id: fixed_guidance.safety_parameters},
        )
        fields["fixed_parameter_baseline"] = build_safe_sage_field_pipeline(
            fixed_grid,
            guidance_policy=guidance_policy,
            config=field_config,
            semantic_obstacles=(semantic,),
            track_guidance_parameters={target.track_id: fixed_guidance},
        )
        policies["fixed_parameter_baseline"] = {
            "cra": {
                "phi": fixed_risk,
                "source": "constant experimental baseline, not physical assessment",
            },
            "safe_sage": {
                "boundary_flux_magnitude": fixed_guidance.boundary_flux_magnitude,
                "tangential_strength": 1.25,
            },
        }
        fixed_field = fields["fixed_parameter_baseline"]
        fixed_prediction = mpc.plan(
            np.zeros(3),
            np.array([[item.linear_velocity, item.angular_velocity] for item in nominal]),
            configuration_space=fixed_field.configuration_space,
            h_full=fixed_field.h_full,
            guidance_fields=fixed_field.guidance_fields,
            temporal_safety_derivative=0.0,
            preferred_lateral_direction_grid_y=-1.0,
        )
        predictions["fixed_parameter_baseline"] = {
            "success": existing.predictive_result_is_valid(
                fixed_prediction, constraint_tolerance=mpc.config.constraint_tolerance
            ),
            "status": fixed_prediction.status,
            "minimum_margin": float(np.min(fixed_prediction.dtcbf_margins)),
            "predicted_chassis_states": fixed_prediction.predicted_chassis_states.tolist(),
        }
    phi = [policies[name]["cra"]["phi"] for name in adapters]
    flux = [
        policies[name]["safe_sage"]["boundary_flux_magnitude"]
        for name in adapters
    ]
    source_mapping_passed = (
        all(a < b for a, b in zip(phi, phi[1:]))
        and all(a < b for a, b in zip(flux, flux[1:]))
        if args.oracle_comparison else (
            policies["observed"] == sensor_policy.as_dict()
            and policies["observed"]["cra"]["phi"] == sensor_policy.assessment.phi
        )
    )
    mapping_passed = (
        source_mapping_passed
        and all(np.isfinite(field.h_full.values).all() for field in fields.values())
        and all(
            item["safe_sage"]["tangential_strength"] == 1.25
            for item in policies.values()
        )
    )
    gate = (
        stationary_passed
        and mapping_passed
        and all(item["success"] for item in predictions.values())
        and (not args.oracle_comparison or all(item["ready"] for item in readiness.values()))
        and (not (args.require_four_factor_readiness or args.ground_continuity_prior) or
             runtime_readiness.ready)
    )
    report = {
        "stationary_passed": stationary_passed,
        "mapping_passed": mapping_passed,
        "motion_gate_passed": gate,
        "evidence_mode": ("oracle profiles with measured LiDAR geometry" if args.oracle_comparison
                          else "prior-assisted camera/LiDAR with conservative uncertainty" if runtime_evidence_mode is EvidenceMode.PRIOR
                          else "observed camera/LiDAR with conservative uncertainty"),
        "acceptance_scope": "oracle controller comparison" if args.oracle_comparison else "uncertainty-aware observed-object navigation; complete four-factor perception is assessed separately",
        "aggregate_sensor_acceptance": False if args.oracle_comparison else readiness["observed"]["aggregate_phi_reportable"],
        "aggregate_prior_assisted_acceptance": prior_readiness.aggregate_phi_reportable,
        "prior_assisted_perception_readiness": asdict(prior_readiness),
        "runtime_perception_readiness": asdict(runtime_readiness),
        "four_factor_readiness_required": args.require_four_factor_readiness or args.ground_continuity_prior,
        "ground_continuity_prior_enabled": args.ground_continuity_prior,
        "plume_control": args.plume_control,
        "contact_control": args.contact_control,
        "seed_color_outline_enabled": args.seed_color_outline,
        "spherical_fit_mode": spherical_fit_mode,
        "instance_mask_geometry_enabled": args.instance_mask_geometry,
        "instance_mask_margin_pixels": args.instance_mask_margin_pixels,
        "chromatic_instance_proposal_enabled": args.chromatic_instance_proposal,
        "sam_instance_proposal_enabled": args.sam_instance_proposal,
        "category_free_visual_mass_enabled": args.category_free_visual_mass,
        "full_target_track_return_count": len(full_track_xyz),
        "unassessed_occupancy_return_count": len(unassessed_points_anchor) if unassessed_points_anchor is not None else 0,
        "unassessed_occupancy_phi": 1.0 if args.instance_mask_geometry else None,
        "authored_support_drop_m": args.support_drop_m,
        "authored_support_width_m": args.support_width_m if args.support_drop_m else None,
        "fall_interaction_prior_enabled": args.worst_case_fall_interaction_prior,
        "fall_surface_configuration": asdict(surface_config) if surface_config else None,
        "fitted_bottom_support_consistency": support_consistency,
        "spherical_body_assembly_prior_enabled": args.spherical_body_prior,
        "independent_scoring_disks": [{"center_xy": center.tolist(), "radius_m": radius}
                                      for center, radius in scoring_disks],
        "ground_support_configuration": asdict(ground_config) if ground_config is not None else None,
        "sensor_dataflow_passed": sensor_dataflow_passed,
        "camera_lidar_age_seconds": camera_lidar_age,
        "model_artifacts": model_artifacts,
        "model_invocations": model_invocations,
        "sensor_phi": sensor_policy.assessment.phi,
        "sensor_factors": diagnose_factor_specific_cra(sensor_adapter).to_dict(),
        "sensor_attributes": asdict(sensor_attributes),
        "visual_contact_proxy": asdict(visual_contact_proxy) if visual_contact_proxy is not None else None,
        "authored_fixture_radius_m": RADIUS_M,
        "authored_fixture_appearance": args.object_appearance,
        "sphere_density_prior_kg_m3": args.sphere_density_prior_kg_m3,
        "root_layer_unchanged_by_fixture": root_unchanged_by_fixture,
        "position_drift_m": drift,
        "yaw_drift_degrees": math.degrees(yaw_drift),
        "target_point_count": len(object_xyz),
        "target_track_id": target.track_id,
        "association_error_m": association_error,
        "distinct_matched_frames": len(samples),
        "fall_surface_rejections": surfaces.rejection_reasons,
        "profiles": {name: asdict(value) for name, value in PROFILE_INPUTS.items()} if args.oracle_comparison else {},
        "policies": policies,
        "field_risk_cells": field_risk_cells,
        "factor_diagnostics": factors,
        "stationary_predictions": predictions,
        "perception_readiness": readiness,
        "fixed_lambda": 1.25,
        "grid_configuration": asdict(grid_config),
        "field_configuration": asdict(field_config),
        "mpc_configuration": asdict(mpc.config),
        "capture_sha256": hashlib.sha256(
            (args.output / "capture.npz").read_bytes()
        ).hexdigest(),
        "camera_sha256": hashlib.sha256((args.output / "camera.png").read_bytes()).hexdigest(),
        "assumptions": [
            "frozen field and CRA assessment during each pass-by; static sphere; h_t=0",
            "robot ego pose and LiDAR extrinsics are simulation-oracle USD references; world-Z-zero floor is a configured prior",
            "oracle comparison only: controlled factor values are supplied and separately labeled",
            "observed mode uses the computed conservative sensor policy unchanged, including phi saturation",
            "unmatched part masks are not negative evidence; a separately accepted visible silhouette may support a continuous contact proxy",
            "silhouette shape is a 2D visible-outline proxy; hidden features and physical sharpness/hardness remain unknown",
            "optional protrusion is actual visible/collidable geometry; spherical mass extrapolation must still pass its unchanged residual gate and is only a rough body-volume hypothesis",
            "optional fitted-sphere mass depends on an explicitly configured effective-density prior; size alone is not measured mass",
            "optional ground support uses a LiDAR floor plane plus an explicit bounded continuity PRIOR; fitted bottom only gates consistency, never supplies fall height",
            "ground-support harmful-fall interaction uses a separately labeled worst-case PRIOR upper bound 1, not route probability or absence of risk",
            "no plume is not proof of a cold object; absent thermal prerequisites stay unknown",
            "optional object plume policy accepts low visible cue only with validated stationary clean-reference observations; all track records include rejection reasons",
            "plume controls alter rendered RGB only; puffs are opaque stylized proxies, not simulated steam or temperature; LiDAR geometry is frozen before puffs appear",
            "kinetic uses nonzero 0.20 m/s relative approach speed even though the object is stationary",
            "authored location selects the target track; authored radius only scores independent clearance",
            "unsupported estimator outputs remain unavailable with per-factor rejection reasons",
            "optional chromatic RGB proposals require a validated separated uniformly colored object against contrasting background; no authored color/mask; heuristic confidence .5; class stays unknown",
            "instance mode permits a merged support/object track median outside the ball box; unique mask overlap and unchanged sphere quality gates remain required",
            "remaining detected LiDAR returns retain conservative unknown-risk occupancy; existing dense guidance arbitration is an implementation adaptation",
            "complete four-factor readiness is reported separately from uncertainty-aware controller acceptance",
            "fixed geometry, waypoint, controller limits and lambda across comparisons",
        ],
        "motion": {},
    }
    save_json("stationary.json", report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "stationary_passed",
                    "mapping_passed",
                    "motion_gate_passed",
                    "sensor_phi",
                    "target_point_count",
                    "position_drift_m",
                )
            }
        ),
        flush=True,
    )
    if args.motion and not gate:
        raise RuntimeError("stationary dataflow/field gate rejected wheel motion")
    if args.motion:
        initial_dof_positions = controller.robot.get_dof_positions().numpy().copy()
        motion_names = ("controlled_low", "fixed_parameter_baseline", "controlled_high") if args.oracle_comparison else ("observed",)
        for name in motion_names:
            controller.stop()
            controller.robot.set_velocities(
                linear_velocities=[[0.0, 0.0, 0.0]],
                angular_velocities=[[0.0, 0.0, 0.0]],
            )
            controller.robot.set_dof_velocities(np.zeros_like(initial_dof_positions))
            controller.robot.set_dof_positions(initial_dof_positions)
            stationary.position_robot_for_experiment(controller)
            for _ in range(stationary.SETTLE_STEPS):
                controller.stop()
                simulation_app.update()
            state_before = local_state(stage, anchor)
            if np.linalg.norm(state_before[:2]) > 0.02 or abs(
                state_before[2]
            ) > math.radians(1):
                raise RuntimeError(
                    f"matched trial reset failed: {state_before.tolist()}"
                )

            def advance(v, omega):
                if not simulation_app.is_running() or not app_utils.is_playing():
                    raise RuntimeError("simulation stopped during bounded motion")
                controller.apply_command(v, omega)
                simulation_app.update()

            result = run_bounded_pass_by(
                field=fields[name],
                mpc=mpc,
                backup=backup,
                read_state=lambda: local_state(stage, anchor),
                advance=advance,
                stop=controller.stop,
                clearance=fixture_clearance,
                goal_xy=GOAL_XY,
                physics_dt=stationary.PHYSICS_DT,
                maximum_seconds=args.motion_maximum_seconds,
                progress=lambda row: print(
                    json.dumps({"profile": name, "progress": row}), flush=True
                ),
            )
            result["motion_mode"] = "Isaac Nova Carter physical wheel commands"
            result["initial_state"] = state_before.tolist()
            stopping_states = []
            stopping_clearances = []
            for _ in range(30):
                advance(0.0, 0.0)
                state = local_state(stage, anchor)
                stopping_states.append(state.tolist())
                stopping_clearances.append(fixture_clearance(state))
            stopping_passed = bool(
                np.isfinite(stopping_states).all()
                and min(stopping_clearances) > 0.02
            )
            result["stopping"] = {
                "simulated_seconds": 30 * stationary.PHYSICS_DT,
                "minimum_clearance_m": min(stopping_clearances),
                "trajectory": stopping_states,
                "passed": bool(stopping_passed),
            }
            result["passed"] = result["passed"] and stopping_passed
            report["motion"][name] = result
            save_json("report.json", report)
            print(
                json.dumps(
                    {
                        "profile": name,
                        **{
                            key: value
                            for key, value in result.items()
                            if key not in {"trajectory", "commands", "nominal_commands"}
                        },
                    }
                ),
                flush=True,
            )
    report["passed"] = gate and (
        not args.motion or all(item["passed"] for item in report["motion"].values())
    )
    report["scene_file_unchanged"] = (
        hashlib.sha256(Path(stationary.ROBOT_SCENE_PATH).read_bytes()).hexdigest()
        == scene_hash_before
    )
    report["passed"] = report["passed"] and report["scene_file_unchanged"]
    save_json("report.json", report)
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(
        points_anchor[:, 0],
        points_anchor[:, 1],
        s=8,
        c="black",
        label="Pandar target returns",
    )
    axis.add_patch(
        plt.Circle(
            sphere_center,
            RADIUS_M,
            color="steelblue",
            alpha=0.3,
            label="oracle scoring outline",
        )
    )
    axis.plot([0, GOAL_XY[0]], [0, GOAL_XY[1]], "k--", label="nominal waypoint line")
    for name, result in report["motion"].items():
        states = np.asarray(result["trajectory"])
        axis.plot(
            states[:, 0],
            states[:, 1],
            label=f"{name}, phi={policies[name]['cra']['phi']:.3f}",
        )
    if not report["motion"]:
        for name, result in predictions.items():
            states = np.asarray(result["predicted_chassis_states"])
            axis.plot(states[:, 0], states[:, 1], label=f"{name} stationary prediction")
    axis.scatter(*GOAL_XY, marker="*", s=100, c="green", label="goal")
    axis.set(
        xlabel="Trial-frame x (m)",
        ylabel="Trial-frame y (m)",
        title="Simple object: " + ("oracle comparison" if args.oracle_comparison else "observed CRA with uncertainty") + ", lambda=1.25",
        xlim=(-0.3, 2.8),
        ylim=(-1.3, 1.0),
    )
    axis.set_aspect("equal")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output / "paths.png", dpi=160)
    plt.close(fig)
    controller.stop()
    omni.timeline.get_timeline_interface().stop()
    if not report["passed"]:
        raise RuntimeError(
            "simple-object test did not meet every acceptance criterion; see report.json"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        traceback.print_exc()
        save_json(
            "failure.json",
            {
                "error": str(error),
                "traceback": traceback.format_exc(),
                "motion_requested": args.motion,
            },
        )
        raise
    finally:
        simulation_app.close()

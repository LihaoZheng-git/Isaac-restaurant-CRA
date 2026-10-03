"""Validate positive LiDAR fall-height evidence with a stationary Isaac fixture.

The estimator receives only live PandarXT-32 points. Authored support and floor
heights are read separately after inference for offline scoring. The configured
fixture location is used only to select its LiDAR cluster from the restaurant;
object height is never used as fall height. No wheel motion is commanded.
"""

from __future__ import annotations

from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
import traceback


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault(
    "WARP_CACHE_PATH", str(ROOT / "outputs" / "lidar_fall_height_isaac" / "warp")
)

from robot.config import (
    EXPECTED_METERS_PER_UNIT,
    PANDAR_XT32_LIDAR_PRIM_PATH,
    PHYSICS_DEVICE,
    PHYSICS_DT,
    ROBOT_BASE_PRIM_PATH,
    ROBOT_PRIM_PATH,
    ROBOT_SCENE_PATH,
)
from isaacsim import SimulationApp


HEADLESS = os.environ.get("ISAAC_SIM_HEADLESS", "0") != "0"
simulation_app = SimulationApp({"headless": HEADLESS})


import isaacsim.core.experimental.utils.app as app_utils
from isaacsim.core.simulation_manager import SimulationManager
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

from robot.control.differential_drive import NovaCarterController
from robot.perception.fall_height import estimate_fall_height
from robot.perception.fall_height_diagnostic import evaluate_fall_height_diagnostic
from robot.perception.lidar_fall_surfaces import (
    LidarFallSurfaceConfig,
    estimate_lidar_fall_surfaces,
)
from robot.runtime.simulation_loop import advance_playing_steps, wait_until_playing
from robot.safety.obstacle_detection import (
    ObstacleDetectionConfig,
    detect_obstacle_points_3d,
)
from robot.safety.obstacle_tracking import ObstacleTracker, ObstacleTrackerConfig
from robot.scene.loader import load_robot_scene
from robot.scene.controlled_semantic_obstacle import read_world_bounds
from robot.scene.transforms import (
    compute_relative_prim_transform,
    get_prim_planar_pose,
    get_prim_world_transform,
)
from robot.sensor.lidar import NovaCarterLidar
from robot.sensor.lidar_geometry import transform_points


OUTPUT_DIRECTORY = Path(os.environ.get(
    "CRA_LIDAR_DIAGNOSTIC_DIR", str(ROOT / "outputs" / "lidar_fall_height_isaac")
))
FIXTURE_ROOT = "/World/CRA_FallHeightValidation"
ISOLATED_SCENE = os.environ.get("CRA_FALL_ISOLATED_SCENE", "0") == "1"
FIXTURE_CENTER_ROBOT_XY = np.asarray((2.0, 0.50), dtype=np.float64)
SUPPORT_DROP_M = float(os.environ.get("CRA_FALL_SUPPORT_DROP_M", "0.80"))
if not math.isfinite(SUPPORT_DROP_M) or SUPPORT_DROP_M <= 0:
    raise ValueError("CRA_FALL_SUPPORT_DROP_M must be positive and finite")
SUPPORT_SIZE_M = (1.40, 1.40, 0.10)
OBJECT_SIZE_M = (0.14, 0.14, 0.34)
CAPTURE_FRAME_COUNT = 12
MAXIMUM_SENSOR_WAIT_STEPS = 400
FIXTURE_SURFACE_CONFIG = LidarFallSurfaceConfig(
    support_xy_margin_m=0.30,
    support_maximum_gap_m=0.20,
    minimum_support_overhang_m=0.05,
    # At the mounted scanner's elevation, the first visible floor returns
    # beyond the platform can lie outside the default 0.90 m neighborhood.
    landing_maximum_offset_m=1.50,
)


def _material(stage, path: str, color: tuple[float, float, float]):
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _cube(stage, path: str, center, size, material) -> None:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(*center))
    # A USD cube with size=1 has unit full extent, not half extent.
    cube.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)


def author_fixture(stage, floor_height_robot_m: float) -> None:
    """Author a fixed world fixture using robot-frame coordinates at setup."""

    with Usd.EditContext(stage, stage.GetSessionLayer()):
        _author_fixture(stage, floor_height_robot_m)


def _author_fixture(stage, floor_height_robot_m: float) -> None:
    if ISOLATED_SCENE:
        UsdGeom.Imageable(stage.GetPrimAtPath("/World/Environment")).MakeInvisible()
    root = UsdGeom.Xform.Define(stage, FIXTURE_ROOT)
    root.AddTransformOp().Set(get_prim_world_transform(stage, ROBOT_BASE_PRIM_PATH))
    support = _material(stage, f"{FIXTURE_ROOT}/SupportMaterial", (0.18, 0.42, 0.72))
    target = _material(stage, f"{FIXTURE_ROOT}/TargetMaterial", (0.90, 0.28, 0.10))
    if ISOLATED_SCENE:
        _cube(stage, f"{FIXTURE_ROOT}/Landing", (2.0, 0.0, floor_height_robot_m - 0.05),
              (20.0, 20.0, 0.10), support)
    support_center_z = floor_height_robot_m + SUPPORT_DROP_M - SUPPORT_SIZE_M[2] / 2
    _cube(
        stage,
        f"{FIXTURE_ROOT}/Support",
        (2.0, 0.0, support_center_z),
        SUPPORT_SIZE_M,
        support,
    )
    object_center_z = floor_height_robot_m + SUPPORT_DROP_M + OBJECT_SIZE_M[2] / 2
    _cube(
        stage,
        f"{FIXTURE_ROOT}/Target",
        (*FIXTURE_CENTER_ROBOT_XY, object_center_z),
        OBJECT_SIZE_M,
        target,
    )


def floor_height_robot(stage) -> float:
    robot_to_world = get_prim_world_transform(stage, ROBOT_BASE_PRIM_PATH)
    origin = robot_to_world.ExtractTranslation()
    ground_world = Gf.Vec3d(origin[0], origin[1], 0.0)
    return float(robot_to_world.GetInverse().Transform(ground_world)[2])


def wait_for_lidar(lidar: NovaCarterLidar, controller: NovaCarterController):
    last_timestamp = None
    frames = []
    for _ in range(MAXIMUM_SENSOR_WAIT_STEPS):
        controller.stop()
        simulation_app.update()
        try:
            points = lidar.get_point_cloud()
            info = lidar.get_frame_info()
        except RuntimeError:
            continue
        if info.sensor_timestamp_nanoseconds == last_timestamp:
            continue
        last_timestamp = info.sensor_timestamp_nanoseconds
        frames.append((points, info))
        if len(frames) >= CAPTURE_FRAME_COUNT:
            return frames
    raise TimeoutError("LiDAR did not provide enough distinct stationary frames")


def main() -> None:
    loaded = load_robot_scene(
        scene_path=ROBOT_SCENE_PATH,
        simulation_app=simulation_app,
        robot_prim_path=ROBOT_PRIM_PATH,
        expected_meters_per_unit=EXPECTED_METERS_PER_UNIT,
    )
    floor_z = floor_height_robot(loaded.stage)
    author_fixture(loaded.stage, floor_z)
    support_bounds = read_world_bounds(loaded.stage, f"{FIXTURE_ROOT}/Support")
    object_bounds = read_world_bounds(loaded.stage, f"{FIXTURE_ROOT}/Target")
    # Independent authored geometry is checked offline, never given to the
    # surface estimator. Setup/cluster selection remain controlled fixtures.
    fixture_geometry_valid = (
        abs(support_bounds.maximum[2] - SUPPORT_DROP_M) < 0.005
        and abs(object_bounds.minimum[2] - SUPPORT_DROP_M) < 0.005
        and abs(object_bounds.size[2] - OBJECT_SIZE_M[2]) < 0.005
    )
    if not fixture_geometry_valid:
        raise RuntimeError("rendered fixture does not match declared metric geometry")
    controller = NovaCarterController(robot_prim_path=ROBOT_PRIM_PATH)
    SimulationManager.setup_simulation(dt=PHYSICS_DT, device=PHYSICS_DEVICE)
    app_utils.play()
    if not wait_until_playing(simulation_app):
        raise RuntimeError("Isaac playback did not start")
    if not advance_playing_steps(simulation_app, physics_steps=60, on_step=controller.stop):
        raise RuntimeError("Isaac closed while the stationary fixture settled")
    controller.verify_articulation()
    lidar = NovaCarterLidar()
    lidar.initialize()
    robot_from_lidar = compute_relative_prim_transform(
        loaded.stage,
        source_prim_path=PANDAR_XT32_LIDAR_PRIM_PATH,
        target_prim_path=ROBOT_BASE_PRIM_PATH,
    )
    initial_pose = np.asarray(get_prim_planar_pose(loaded.stage, ROBOT_BASE_PRIM_PATH))
    frames = wait_for_lidar(lidar, controller)
    # Use one internally accumulated Pandar scan so the aligned tracker returns
    # can be removed exactly from the same scene cloud. Stacking asynchronous
    # scans would leave near-duplicate target returns that resemble support.
    scene_points = transform_points(frames[-1][0], robot_from_lidar)
    # The chassis settles from its authored pose. Re-express the independently
    # known floor in the current sensor frame for fixture selection/scoring.
    floor_z = floor_height_robot(loaded.stage)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT_DIRECTORY / "scene_points.npz",
                        scene_points_robot_xyz=scene_points,
                        robot_from_lidar=robot_from_lidar,
                        floor_height_robot_m=floor_z)
    support_height = floor_z + SUPPORT_DROP_M
    obstacle_points = detect_obstacle_points_3d(
        scene_points,
        ObstacleDetectionConfig(
            minimum_height=support_height + 0.07,
            maximum_height=support_height + OBJECT_SIZE_M[2] + 0.08,
            maximum_range=5.0,
            forward_only=True,
            angular_bin_size_degrees=0.15,
        ),
    )
    timestamp = frames[-1][1].sensor_timestamp_nanoseconds * 1e-9
    tracking = ObstacleTracker(
        ObstacleTrackerConfig(
            cluster_radius=0.10,
            minimum_cluster_points=5,
            maximum_association_distance=0.4,
        )
    ).update(
        obstacle_points[:, :2],
        robot_position_world_xy=tuple(initial_pose[:2]),
        robot_yaw=float(initial_pose[2]),
        timestamp_seconds=timestamp,
    )
    if not tracking.tracked_obstacles:
        raise RuntimeError("no elevated-object LiDAR track was found")
    target_track = min(
        tracking.tracked_obstacles,
        key=lambda item: float(
            np.linalg.norm(np.asarray(item.centroid_robot_xy) - FIXTURE_CENTER_ROBOT_XY)
        ),
    )
    association_distance = float(
        np.linalg.norm(
            np.asarray(target_track.centroid_robot_xy) - FIXTURE_CENTER_ROBOT_XY
        )
    )
    if association_distance > 0.35:
        raise RuntimeError(
            f"nearest LiDAR track is not the validation target: {association_distance:.3f} m"
        )
    object_points = obstacle_points[tracking.point_track_ids == target_track.track_id]
    surface_result = estimate_lidar_fall_surfaces(
        scene_points,
        object_points,
        timestamp_seconds=timestamp,
        config=FIXTURE_SURFACE_CONFIG,
    )
    fall = estimate_fall_height(surface_result.evidence, timestamp_seconds=timestamp)
    report = evaluate_fall_height_diagnostic(
        fall,
        authored_support_height_m=support_height,
        authored_landing_height_m=floor_z,
    )
    controller.stop()
    simulation_app.update()
    final_pose = np.asarray(get_prim_planar_pose(loaded.stage, ROBOT_BASE_PRIM_PATH))
    drift = float(np.linalg.norm(final_pose[:2] - initial_pose[:2]))
    yaw_drift = abs(math.atan2(math.sin(final_pose[2] - initial_pose[2]), math.cos(final_pose[2] - initial_pose[2])))
    blockers = list(report.blockers)
    if drift > 0.005 or yaw_drift > math.radians(0.1):
        blockers.append("robot did not remain stationary")
    payload = {
        **asdict(report),
        "blockers": blockers,
        "passed": not blockers,
        "evidence_source": fall.source,
        "evidence_status": fall.status.value,
        "object_height_used_as_fall_height": False,
        "fixture_track_id": target_track.track_id,
        "fixture_track_association_distance_m": association_distance,
        "object_point_count": int(object_points.shape[0]),
        "scene_point_count": int(scene_points.shape[0]),
        "fall_direction_robot_xy": surface_result.fall_direction_robot_xy,
        "surface_rejection_reasons": surface_result.rejection_reasons,
        "experimental_surface_config": asdict(FIXTURE_SURFACE_CONFIG),
        "runtime_default_config_changed": False,
        "robot_position_drift_m": drift,
        "robot_yaw_drift_degrees": math.degrees(yaw_drift),
        "robot_motion_command": "zero",
        "oracle_role": "offline_scoring_only",
        "fixture_geometry_valid": fixture_geometry_valid,
        "rendered_support_top_world_m": support_bounds.maximum[2],
        "rendered_object_bottom_world_m": object_bounds.minimum[2],
        "declared_object_height_m": OBJECT_SIZE_M[2],
        "rendered_object_height_m": object_bounds.size[2],
        "selection_boundary": "authored target location and height filter; inference uses live LiDAR surfaces",
        "scene_scope": "isolated_visible_fixture" if ISOLATED_SCENE else "restaurant_fixture",
    }
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIRECTORY / "report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        OUTPUT_DIRECTORY / "points.npz",
        scene_points_robot_xyz=scene_points,
        object_points_robot_xyz=object_points,
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    if blockers:
        raise RuntimeError("stationary LiDAR fall-height diagnostic did not pass")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()

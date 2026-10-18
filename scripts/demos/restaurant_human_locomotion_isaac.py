"""Native idle/walk-or-fast-locomotion/stop; Nova Carter receives zero commands."""

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scene", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--layout-survey", type=Path, required=True)
parser.add_argument("--speed", type=float, default=0.7)
parser.add_argument("--goal-y", type=float, default=12.5)
parser.add_argument("--goal-x", type=float, default=29.8)
parser.add_argument("--sensor-diagnostic", action="store_true")
parser.add_argument("--sensor-simulation-rate", type=float, default=0.1)
parser.add_argument("--robot-view", choices=("approach", "close"), default="approach")
parser.add_argument("--camera-upward-degrees", type=float, default=0.)
parser.add_argument("--camera-raise-metres", type=float, default=0.)
parser.add_argument(
    "--human-mass-prior",
    nargs=3,
    type=float,
    default=(70.0, 45.0, 100.0),
    metavar=("KG", "LOWER", "UPPER"),
)
args = parser.parse_args()
if not 0.5 <= args.speed <= 2:
    parser.error("bounded native human speed is .5..2 m/s")
if not 0 <= args.camera_upward_degrees <= 25 or not 0 <= args.camera_raise_metres <= .6 or (
    (args.robot_view != "approach" or args.camera_upward_degrees or args.camera_raise_metres) and not args.sensor_diagnostic
):
    parser.error("experimental camera/close view requires sensor diagnostic; pitch 0..25")
crossing_goal = args.goal_x == 27.8 and args.goal_y == 8.9
if (not crossing_goal and (args.goal_x != 29.8 or not 9.8 <= args.goal_y <= 12.5)
    or not 0.05 <= args.sensor_simulation_rate <= 1):
    parser.error("use original x=29.8/y=9.8..12.5 or crossing goal (27.8,8.9); pacing .05..1")
args.scene, args.output, args.layout_survey = (
    args.scene.resolve(),
    args.output.resolve(),
    args.layout_survey.resolve(),
)
if args.output.exists():
    parser.error("use a fresh output directory")
args.output.mkdir(parents=True)
if args.sensor_diagnostic:
    # Resolve the existing vision torch/torchvision pair before Kit adds paths.
    importlib.import_module("robot.perception.visual_mass_runtime")

from isaacsim import SimulationApp

# Late extension activation breaks RGB capture in this installed Kit build.
# Load the already installed native animation stack before renderer startup.
app = SimulationApp(
    {
        "headless": True,
        "enable_motion_bvh": True,
        "extra_args": [
            "--enable",
            "omni.anim.behavior.core",
            "--/rtx/scenedb/maxHistoryTransformCount=128",
        ],
    }
)
import carb
import carb.settings
import cv2
import numpy as np
import omni.anim.behavior.core as bh
import omni.anim.navigation.core as nav
import omni.timeline
import omni.usd
from isaacsim.core.experimental.prims import RigidPrim
from isaacsim.core.experimental.utils.backend import use_backend
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera
from pxr import Gf, Sdf, Usd, UsdGeom, UsdSkel
from robot.config import (
    PHYSICS_DEVICE,
    PHYSICS_DT,
    ROBOT_PRIM_PATH,
    ROBOT_BASE_PRIM_PATH,
)
from robot.control.differential_drive import NovaCarterController
from robot.evaluation.human_locomotion import summarize_human_locomotion
from robot.evaluation.restaurant_service_validation import (
    authored_raster_clearance,
    load_authored_layout_monitor,
)
from robot.safety.safe_sage_configuration_space import RectangularRobotFootprint
from robot.scene.controlled_semantic_obstacle import read_world_bounds
from robot.scene.restaurant_human_motion import (
    prepare_native_restaurant_human,
)
from robot.scene.restaurant_service_build import (
    SERVICE_ROOT,
    SHORT_START_WORLD_XYZ,
    SHORT_START_YAW,
    service_props,
)
from robot.scene.transforms import get_prim_planar_pose


def save(name, data):
    (args.output / name).write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def main():
    save("run-configuration.json", {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    })
    goal = (args.goal_x, args.goal_y, 0.0)
    robot_start = (28.8, 7.9, SHORT_START_WORLD_XYZ[2]) if args.robot_view == "close" else SHORT_START_WORLD_XYZ
    robot_yaw = math.pi / 2 if args.robot_view == "close" else SHORT_START_YAW
    sources = {
        ROOT / "scenes/generated/restaurant_base.usda",
        ROOT / "scenes/generated/restaurant_robot.usda",
    }
    pending = [args.scene]
    while pending:
        p = pending.pop()
        if p in sources:
            continue
        sources.add(p)
        layer = Sdf.Layer.FindOrOpen(str(p))
        pending.extend(
            Path(Sdf.ComputeAssetPathRelativeToLayer(layer, s)).resolve()
            for s in layer.subLayerPaths
        )
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    if not omni.usd.get_context().open_stage(str(args.scene)):
        raise RuntimeError("scene unavailable")
    for _ in range(100):
        app.update()
    stage = omni.usd.get_context().get_stage()
    additions = []
    for prop in service_props():
        b = read_world_bounds(stage, SERVICE_ROOT + "/" + prop.name)
        additions.append({"bounds_minimum": b.minimum, "bounds_maximum": b.maximum})
    points, pixel_allowance, _ = load_authored_layout_monitor(
        args.layout_survey, additions
    )
    # Existing measured Nova Carter footprint; independent abort information.
    footprint = RectangularRobotFootprint(0.14, 0.587351, 0.2466, 0.2466, 0.05)
    setup_clearance = authored_raster_clearance(
        points,
        (*robot_start[:2], robot_yaw),
        footprint,
        pixel_allowance,
    )
    if setup_clearance <= 0.02:
        raise RuntimeError("configured parked robot overlaps authored occupancy")
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        setup = prepare_native_restaurant_human(stage)
    save("native-setup.json", setup)
    for _ in range(20):
        app.update()
    print("Baking bounded restaurant human NavMesh...", flush=True)
    if not nav.acquire_interface().start_navmesh_baking_and_wait():
        raise RuntimeError("native restaurant NavMesh bake failed")
    controller = NovaCarterController(robot_prim_path=ROBOT_PRIM_PATH)
    SimulationManager.setup_simulation(dt=PHYSICS_DT, device=PHYSICS_DEVICE)
    carb.settings.get_settings().set("/app/player/useFixedTimeStepping", True)
    timeline = omni.timeline.get_timeline_interface()
    timeline.set_play_every_frame(True)
    timeline.set_time_codes_per_second(60)
    timeline.play()
    for _ in range(10):
        app.update()
    controller.robot.set_world_poses(
        positions=[robot_start],
        orientations=[
            [math.cos(robot_yaw / 2), 0, 0, math.sin(robot_yaw / 2)]
        ],
    )
    for _ in range(60):
        controller.stop()
        app.update()
    settled_clearance = authored_raster_clearance(
        points,
        get_prim_planar_pose(stage, ROBOT_BASE_PRIM_PATH),
        footprint,
        pixel_allowance,
    )
    save(
        "robot-setup-validation.json",
        {
            "configured_clearance_m": setup_clearance,
            "settled_clearance_m": settled_clearance,
            "scope": "authored independent setup/abort only; never perception or control inputs",
        },
    )
    if settled_clearance <= 0.02:
        raise RuntimeError("settled parked robot overlaps authored occupancy")
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        camera_path = "/World/HumanMotionDiagnosticCamera"
        rtx_camera = RtxCamera.create(camera_path, tick_rate=10)
        camera = UsdGeom.Camera.Get(stage, camera_path)
        camera.MakeMatrixXform().Set(
            Gf.Matrix4d()
            .SetLookAt(
                Gf.Vec3d(25.9, 6.7, 3.2) if crossing_goal else Gf.Vec3d(27.7, 10.6, 2.0),
                Gf.Vec3d(29.0, 8.9, .8) if crossing_goal else Gf.Vec3d(29.8, 10.7, 0.9),
                Gf.Vec3d(0, 0, 1),
            )
            .GetInverse()
        )
        camera.CreateFocalLengthAttr(18)
        camera.CreateHorizontalApertureAttr(36)
        camera.CreateVerticalApertureAttr(27)
    camera_sensor = CameraSensor(rtx_camera, resolution=(600, 800), annotators=["rgb"])
    for _ in range(30):
        controller.stop()
        app.update()
    agent = bh.acquire_interface().get_agent(setup["rig_path"])
    if agent is None:
        raise RuntimeError("native retargeted agent not initialized")
    agent.set_auto_avoidance_enabled(False)
    body_view = RigidPrim(setup["collider_path"])
    if not body_view.is_physics_tensor_entity_valid():
        raise RuntimeError("native actor collider has no physics readback")
    skeleton = next(
        p
        for p in Usd.PrimRange(stage.GetPrimAtPath(setup["rig_path"]))
        if p.IsA(UsdSkel.Skeleton)
    )
    names = list(UsdSkel.Skeleton(skeleton).GetJointsAttr().Get())
    feet = {
        side: next(
            i for i, n in enumerate(names) if n.rsplit("/", 1)[-1] == side + "_Foot"
        )
        for side in ("L", "R")
    }
    agent.idle()
    # Let the native idle transition and its one-physics-step collider settle
    # before starting the recorded idle/walk/stop episode.
    for _ in range(60):
        controller.stop()
        app.update()
    sensor = None
    if args.sensor_diagnostic:
        from robot.config import FRONT_HAWK_LEFT_CAMERA_PRIM_PATH
        from robot.scene.camera_mount import apply_experimental_camera_pitch
        from robot.evaluation.restaurant_human_sensor import (
            RestaurantHumanSensorDiagnostic,
        )

        save("camera-mount-audit.json", apply_experimental_camera_pitch(
            stage, FRONT_HAWK_LEFT_CAMERA_PRIM_PATH, args.camera_upward_degrees,
            ROBOT_BASE_PRIM_PATH,
            raise_metres=args.camera_raise_metres,
        ))

        sensor = RestaurantHumanSensorDiagnostic(
            stage,
            ROOT,
            args.output,
            get_prim_planar_pose(stage, ROBOT_BASE_PRIM_PATH),
            mass_interval=args.human_mass_prior,
        )
    phase = "idle_before"
    phase_start = timeline.get_current_time()
    task = None
    task_succeeded = False
    records = []
    frame_epochs = []
    frame_hashes = set()
    last_frame_time = -math.inf
    writer = cv2.VideoWriter(
        str(args.output / "human-motion.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10,
        (800, 600),
    )
    if not writer.isOpened():
        raise RuntimeError("local video encoder unavailable")
    wall_start = time.perf_counter()
    try:
        for index in range(1800):
            step_started = time.perf_counter()
            wheel = controller.stop()
            if np.any(np.asarray(wheel) != 0):
                raise RuntimeError("nonzero wheel command in actor-only test")
            app.update()
            now = timeline.get_current_time()
            xyz = agent.get_world_translation()
            xyz = [xyz.x, xyz.y, xyz.z]
            if sensor is not None:
                sensor.tick(now, xyz, phase)
            if (
                phase == "idle_before"
                and sensor is not None
                and not sensor.stationary_ready
                and now - phase_start > 8
            ):
                raise RuntimeError(
                    "stationary fresh human-person/mass prior gate failed before actor movement"
                )
            if (
                phase == "idle_before"
                and now - phase_start >= (3 if sensor else 1)
                and (sensor is None or sensor.stationary_ready)
            ):
                agent.set_speed(args.speed / UsdGeom.GetStageMetersPerUnit(stage))
                task = agent.move_to(goal, auto_brake=True)
                if task == bh.BEHAVIOR_TASK_ID_INVALID:
                    raise RuntimeError("native move task rejected")
                phase = "move"
                phase_start = now
            elif phase == "move":
                status = agent.get_task_status(task)
                if status == bh.BehaviorTaskStatus.FAILED:
                    raise RuntimeError("native move task failed")
                if status == bh.BehaviorTaskStatus.SUCCEEDED:
                    task_succeeded = True
                    agent.idle()
                    phase = "stopped_after"
                    phase_start = now
                elif now - phase_start > 20:
                    raise RuntimeError("native human endpoint timeout")
            foot_values = {}
            for side, joint in feet.items():
                t, q = carb.Float3(), carb.Float4()
                if not agent.get_joint_world_transform(joint, t, q):
                    raise RuntimeError("native foot transform unavailable")
                foot_values[side] = [t.x, t.y, t.z]
            # Native animation uses Fabric; ordinary USD transforms may be stale.
            # Audit the actual physics body rather than its authored transform.
            with use_backend("tensor", raise_on_fallback=True):
                body = body_view.get_world_poses()[0].numpy()[0]
            error = float(
                np.linalg.norm(
                    np.array(body)
                    - np.array(xyz)
                    - [0, 0, setup["collision_height_m"] / 2]
                )
            )
            clearance = float(
                np.linalg.norm(points - np.array(xyz[:2]), axis=1).min()
                - setup["collision_radius_m"]
                - pixel_allowance
            )
            if clearance <= 0.02:
                raise RuntimeError("independent human layout-clearance guard")
            robot = get_prim_planar_pose(stage, ROBOT_BASE_PRIM_PATH)
            records.append(
                {
                    "time": now,
                    "phase": phase,
                    "actor_xyz": xyz,
                    "left_foot_xyz": foot_values["L"],
                    "right_foot_xyz": foot_values["R"],
                    "collider_alignment_error_m": error,
                    "physics_collider_center_xyz": body.tolist(),
                    "actor_layout_clearance_m": clearance,
                    "robot_xy": robot[:2],
                    "nonzero_wheel_command_count": 0,
                }
            )
            if now - last_frame_time >= 0.1 - 1e-6:
                data, _ = camera_sensor.get_data("rgb")
                data = np.asarray([]) if data is None else data.numpy()
                if index == 0:
                    print("RGB diagnostic shape:", data.shape, flush=True)
                if data.shape[:2] == (600, 800):
                    frame = data[:, :, :3].astype(np.uint8)
                    writer.write(frame[:, :, ::-1])
                    frame_epochs.append(now)
                    frame_hashes.add(hashlib.sha256(frame.tobytes()).hexdigest())
                    if index % 30 == 0:
                        cv2.imwrite(
                            str(args.output / f"frame-{index:04d}.png"),
                            frame[:, :, ::-1],
                        )
                last_frame_time = now
            if index % 60 == 0:
                print(
                    json.dumps(
                        {
                            "phase": phase,
                            "time": now,
                            "xyz": xyz,
                            "collider_error": error,
                            "layout_clearance": clearance,
                        }
                    ),
                    flush=True,
                )
            if index >= 60 and not frame_epochs:
                raise RuntimeError("no RTX RGB frames during the initial idle window")
            if phase == "stopped_after" and now - phase_start >= 1.5:
                break
            if sensor is not None:
                time.sleep(
                    max(
                        0,
                        PHYSICS_DT / args.sensor_simulation_rate
                        - (time.perf_counter() - step_started),
                    )
                )
    finally:
        controller.stop()
        writer.release()
        camera_sensor.detach_annotators("rgb")
        save("actor-records.json", records)
        save("video-frame-epochs.json", frame_epochs)
        sensor_report = (
            sensor.close(timeline.get_current_time()) if sensor is not None else None
        )
    video = cv2.VideoCapture(str(args.output / "human-motion.mp4"))
    decoded_frames = 0
    while video.read()[0]:
        decoded_frames += 1
    video.release()
    report = summarize_human_locomotion(
        records,
        goal_world_xyz=goal,
        requested_speed_mps=args.speed,
        task_succeeded=task_succeeded,
    )
    report.update(
        {
            "requested_speed_mps": args.speed,
            "goal_world_xyz": goal,
            "human_sensor_gate": sensor_report,
            "source_files_unchanged": all(
                hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p)]
                for p in sources
            ),
            "source_sha256": hashes,
            "wall_seconds": time.perf_counter() - wall_start,
            "nonzero_wheel_commands": 0,
            "collider_readback": "PhysX tensor world pose, not ordinary USD transforms; update-phase lag is monitored",
            "render_scope": "10 Hz RTX camera diagnostic; no real-time claim",
            "video_frames_written": len(frame_epochs),
            "video_frames_decoded": decoded_frames,
            "unique_rgb_frame_count": len(frame_hashes),
        }
    )
    report["checks"]["nonempty_video"] = (
        decoded_frames >= 20
        and decoded_frames == len(frame_epochs)
        and len(frame_hashes) >= 10
    )
    report["passed"] &= (
        report["source_files_unchanged"] and report["checks"]["nonempty_video"]
    )
    if sensor_report is not None:
        report["passed"] &= sensor_report["passed"]
    save("validation.json", report)
    print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise RuntimeError("native human locomotion gate failed")


try:
    main()
except Exception:
    traceback.print_exc()
    raise
finally:
    app.close()

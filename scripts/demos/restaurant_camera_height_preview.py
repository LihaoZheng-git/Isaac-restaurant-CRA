"""Paused RGB-only camera-height survey in the unchanged restaurant v2 scene.

No physics, wheel commands, detector, LiDAR association or CRA acceptance. The
authored front Owl lens and runtime resolution are preserved. Poses select
views for inspection only; they are never navigation references.
"""

import argparse
from dataclasses import asdict
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raises", type=float, nargs="+", default=[0, .2, .4, .6])
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new; preserve earlier surveys")
    if any(not math.isfinite(v) or not 0 <= v <= .6 for v in args.raises):
        parser.error("raise must be finite and within 0..0.6 m")
    if len(set(args.raises)) != len(args.raises):
        parser.error("duplicate heights")
    return args


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def survey(args, app):
    import numpy as np
    import omni.replicator.core as rep
    import omni.timeline
    import omni.usd
    from PIL import Image
    from pxr import Gf, Usd, UsdGeom

    from robot.config import ROBOT_BASE_PRIM_PATH, ROBOT_PRIM_PATH
    from robot.scene.camera_calibration import read_fisheye_polynomial_calibration
    from robot.scene.camera_mount import apply_experimental_camera_pitch
    from robot.scene.restaurant_demo_build import DEMO_V2_OBSERVATIONS, demo_prop_records
    from robot.scene.restaurant_service_build import SERVICE_ROOT
    from robot.scene.controlled_semantic_obstacle import read_world_bounds
    from robot.sensor.camera import FRONT_OWL_CAMERA_PATH

    args.output.mkdir(parents=True)
    if not omni.usd.get_context().open_stage(str(args.scene.resolve())):
        raise RuntimeError("scene did not open")
    for _ in range(100):
        app.update()
    timeline = omni.timeline.get_timeline_interface()
    timeline.stop()
    stage = omni.usd.get_context().get_stage()
    source_paths = {Path(layer.realPath) for layer in stage.GetUsedLayers()
                    if layer.realPath and Path(layer.realPath).is_file()}
    source_paths.update([Path(__file__).resolve(), ROOT / "robot/scene/camera_mount.py"])
    hashes = {str(p): digest(p) for p in source_paths}
    props_before = demo_prop_records(stage)
    pot_before = read_world_bounds(stage, SERVICE_ROOT + "/SecondPot")
    calibration = asdict(read_fisheye_polynomial_calibration(
        stage, camera_prim_path=FRONT_OWL_CAMERA_PATH, output_resolution=(480, 640)))
    base = UsdGeom.Xformable(stage.GetPrimAtPath(ROBOT_BASE_PRIM_PATH))
    robot = UsdGeom.Xformable(stage.GetPrimAtPath(ROBOT_PRIM_PATH))
    camera = UsdGeom.Xformable(stage.GetPrimAtPath(FRONT_OWL_CAMERA_PATH))
    original_camera = camera.GetLocalTransformation()
    original_robot_world = robot.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    original_base_world = base.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    original_z = float(original_base_world.ExtractTranslation()[2])
    parent_world = UsdGeom.Xformable(robot.GetPrim().GetParent()).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default())
    base_from_robot = original_robot_world * original_base_world.GetInverse()
    views = []
    for name, key in (("utensils-facing", "demo-utensils"), ("pot-facing", "pot")):
        _, xyz, yaw = DEMO_V2_OBSERVATIONS[key]
        views.append((name, xyz[:2], yaw, "existing stationary prop-view XY/yaw"))
    # Exact points on the authored route chords, with their forward headings.
    views.extend([
        ("utensils-route", (26.8, 5.725), math.atan2(3.35, 1.2), "midpoint of wide-corner route segment"),
        ("pot-route", (28.8, 11.7), math.pi / 2, "point on northbound v2 route segment"),
    ])
    rows = []
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        product = rep.create.render_product(FRONT_OWL_CAMERA_PATH, (640, 480))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb.attach(product)
        try:
            for name, xy, yaw, provenance in views:
                target_base = Gf.Matrix4d(1)
                target_base.SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), math.degrees(yaw)))
                target_base.SetTranslateOnly(Gf.Vec3d(*xy, original_z))
                robot.MakeMatrixXform().Set(base_from_robot * target_base * parent_world.GetInverse())
                for raise_m in args.raises:
                    # Absolute comparison from the same original mount, not cumulative.
                    camera.MakeMatrixXform().Set(original_camera)
                    audit = apply_experimental_camera_pitch(
                        stage, FRONT_OWL_CAMERA_PATH, 0., ROBOT_BASE_PRIM_PATH,
                        raise_metres=raise_m)
                    for _ in range(8):
                        app.update()
                    frame = None
                    for _ in range(4):
                        rep.orchestrator.step(delta_time=0., rt_subframes=16,
                                              pause_timeline=True, wait_for_render=True)
                        app.update()
                        data = np.asarray(rgb.get_data())
                        if data.shape[:2] == (480, 640) and data.max() > 5:
                            frame = data[:, :, :3].astype(np.uint8)
                    if frame is None:
                        raise RuntimeError(f"missing RGB at {name}/{raise_m}")
                    actual_base = np.asarray(base.ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
                    actual_camera = np.asarray(camera.ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
                    if not np.allclose(actual_base[:3, 3], [*xy, original_z], atol=1e-7):
                        raise RuntimeError("view transform does not match requested chassis pose")
                    current_calibration = asdict(read_fisheye_polynomial_calibration(
                        stage, camera_prim_path=FRONT_OWL_CAMERA_PATH, output_resolution=(480, 640)))
                    if calibration != current_calibration or timeline.get_current_time() != 0:
                        raise RuntimeError("lens or paused timeline changed")
                    filename = f"{name}-raise-{round(raise_m * 100):02d}cm.png"
                    Image.fromarray(frame).save(args.output / filename)
                    rows.append(dict(view=name, raise_m=raise_m, pitch_degrees=0,
                                     image=filename, image_sha256=digest(args.output / filename),
                                     pose_provenance=provenance, robot_world_xy_yaw=[*xy, yaw],
                                     camera_world_xyz=actual_camera[:3, 3].tolist(),
                                     camera_forward_world=(-actual_camera[:3, 2]).tolist(), mount=audit))
                    print(f"saved {filename}; camera world z={actual_camera[2, 3]:.4f} m", flush=True)
        finally:
            rgb.detach(product)
            product.destroy()
    report = dict(
        scope="paused actual front Owl RGB visibility survey only; no detector/association/CRA/navigation acceptance",
        scene=str(args.scene.resolve()), source_sha256=hashes,
        source_files_unchanged=all(digest(p) == value for p, value in hashes.items()),
        props_unchanged=demo_prop_records(stage) == props_before,
        pot_unchanged=read_world_bounds(stage, SERVICE_ROOT + "/SecondPot") == pot_before,
        calibration=calibration, resolution_height_width=[480, 640],
        original_chassis_world_z=original_z, timeline_seconds=timeline.get_current_time(),
        wheel_commands=0, physics_enabled=False, rows=rows)
    report["capture_integrity_passed"] = bool(
        report["source_files_unchanged"] and report["props_unchanged"]
        and report["pot_unchanged"] and report["timeline_seconds"] == 0)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not report["capture_integrity_passed"]:
        raise RuntimeError("capture integrity failed")


if __name__ == "__main__":
    parsed = arguments()
    # Match the native demos: resolve the installed vision package pair before
    # Kit prepends its bundled packages. No installation or environment change.
    importlib.import_module("robot.perception.camera_lidar_fusion")
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        survey(parsed, simulation_app)
    except Exception:
        failure = traceback.format_exc()
        print(failure, flush=True)
        parsed.output.mkdir(parents=True, exist_ok=True)
        (parsed.output / "failure.txt").write_text(failure, encoding="utf-8")
        raise
    finally:
        simulation_app.close()

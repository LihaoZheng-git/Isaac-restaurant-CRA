"""Stationary camera/LiDAR epoch audit; no nonzero wheel commands or scene edits."""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
os.environ.setdefault("ISAAC_SIM_HEADLESS", "1")

from scripts.demos import safe_sage_unicycle_mpc_motion_demo as existing
import numpy as np
from scipy.spatial.transform import Rotation
import omni.timeline
from isaacsim.core.simulation_manager import SimulationManager
import isaacsim.core.experimental.utils.app as app_utils
from robot.control.differential_drive import NovaCarterController
from robot.perception.physical_attributes import EstimateStatus
from robot.perception.sensor_epochs import (
    EgoPoseHistory,
    EgoPoseSample,
    align_camera_lidar_epoch,
)
from robot.scene.loader import load_robot_scene
from robot.scene.transforms import compute_relative_prim_transform
from robot.sensor.camera import NovaCarterCamera
from robot.sensor.lidar import NovaCarterLidar


def main():
    stationary = existing.stationary
    args.output.mkdir(parents=True, exist_ok=True)
    original_hash = hashlib.sha256(
        Path(stationary.ROBOT_SCENE_PATH).read_bytes()
    ).hexdigest()
    loaded = load_robot_scene(
        simulation_app=existing.simulation_app,
        scene_path=stationary.ROBOT_SCENE_PATH,
        robot_prim_path=stationary.ROBOT_PRIM_PATH,
    )
    stage = loaded.stage
    root_before = stage.GetRootLayer().ExportToString()
    controller = NovaCarterController(robot_prim_path=stationary.ROBOT_PRIM_PATH)
    SimulationManager.setup_simulation(
        dt=stationary.PHYSICS_DT, device=stationary.PHYSICS_DEVICE
    )
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    app_utils.update_app(steps=10)
    controller.verify_articulation()
    stationary.position_robot_for_experiment(controller)
    for _ in range(stationary.SETTLE_STEPS):
        controller.stop()
        existing.simulation_app.update()
    lidar = NovaCarterLidar(lidar_prim_path=stationary.PANDAR_XT32_LIDAR_PRIM_PATH)
    lidar.initialize()
    camera = NovaCarterCamera()
    camera.initialize()
    history = EgoPoseHistory()
    camera_from_robot = compute_relative_prim_transform(
        stage,
        source_prim_path=stationary.ROBOT_BASE_PRIM_PATH,
        target_prim_path=camera.camera_prim_path,
    )
    records, rejected = [], []
    last_frame = -1
    for _ in range(400):
        controller.stop()
        existing.simulation_app.update()
        now = float(timeline.get_current_time())
        pose = compute_relative_prim_transform(
            stage,
            source_prim_path=stationary.ROBOT_BASE_PRIM_PATH,
            target_prim_path="/World",
        )
        history.append(
            EgoPoseSample(now, pose, EstimateStatus.ORACLE, "isaac_usd_chassis_pose")
        )
        try:
            points = lidar.get_point_cloud()
            info = lidar.get_frame_info()
            if info.sensor_frame_id == last_frame:
                continue
            camera.get_rgb()
            rgb_info = camera.get_frame_info()
            scan_time = info.sensor_timestamp_nanoseconds * 1e-9
            epoch = align_camera_lidar_epoch(
                history,
                lidar_timestamp_seconds=scan_time,
                camera_timestamp_seconds=rgb_info.simulation_time_seconds,
                camera_from_robot=camera_from_robot,
            )
        except (RuntimeError, ValueError) as error:
            rejected.append(str(error))
            continue
        last_frame = info.sensor_frame_id
        metadata = camera.get_annotator_info()
        records.append(
            {
                "lidar": asdict(info),
                "camera": asdict(rgb_info),
                "point_count": len(points),
                "sensor_age_seconds": epoch.sensor_age_seconds,
                "camera_from_scan_robot": epoch.camera_from_scan_robot.tolist(),
                "pose_source": epoch.lidar_pose.source,
                "pose_status": epoch.lidar_pose.status.value,
                "annotator_metadata": {
                    key: value
                    if isinstance(value, (str, int, float, bool, type(None)))
                    else type(value).__name__
                    for key, value in metadata.items()
                },
            }
        )
        if len(records) >= 8:
            break
    controller.stop()
    timeline.stop()
    scene_unchanged = (
        original_hash
        == hashlib.sha256(Path(stationary.ROBOT_SCENE_PATH).read_bytes()).hexdigest()
    )
    first, last = (
        history.samples[0].world_from_robot,
        history.samples[-1].world_from_robot,
    )
    drift = float(np.linalg.norm(last[:3, 3] - first[:3, 3]))
    rotation_drift = float(
        np.degrees(Rotation.from_matrix(last[:3, :3] @ first[:3, :3].T).magnitude())
    )
    passed = (
        len(records) >= 8
        and scene_unchanged
        and all(item["sensor_age_seconds"] <= 0.15 for item in records)
        and drift <= 0.02
        and rotation_drift <= 1.0
    )
    report = {
        "passed": passed,
        "records": records,
        "rejections": sorted(set(rejected)),
        "scene_unchanged": scene_unchanged,
        "nonzero_wheel_command_count": 0,
        "position_drift_m": drift,
        "rotation_drift_degrees": rotation_drift,
        "root_serialization_changed_during_runtime": root_before
        != stage.GetRootLayer().ExportToString(),
        "root_boundary": "transient sensor/physics runtime changes are not persisted; no layer save is called",
        "scope": "stationary simulation sensor-epoch alignment only; no CRA accuracy or live wheel claim",
        "camera_timestamp_boundary": "wrapper read time; an independent acquisition timestamp is unavailable unless the recorded annotator metadata exposes one",
        "pose_boundary": "oracle USD pose history and extrinsics; no pose extrapolation",
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("passed", "scene_unchanged", "nonzero_wheel_command_count")
            }
        )
    )
    if not passed:
        raise RuntimeError("stationary sensor-epoch gate failed")


try:
    main()
except Exception:
    traceback.print_exc()
    raise
finally:
    existing.simulation_app.close()

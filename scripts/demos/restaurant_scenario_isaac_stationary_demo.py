"""Author the restaurant risk catalog in Isaac Sim without robot motion.

This diagnostic validates scene composition, robot-frame placement, rendered
bounds, and the session-layer boundary.  All catalog roles, dimensions,
velocities, and bounds remain oracle setup/scoring information.  The combined
scene is not a perception result because factor fixtures share one location.
"""

from __future__ import annotations

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
    "WARP_CACHE_PATH",
    str(ROOT / "outputs" / "restaurant_scenario_authoring" / "warp"),
)

from robot.config import (
    EXPECTED_METERS_PER_UNIT,
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

from robot.control.differential_drive import NovaCarterController
from robot.runtime.simulation_loop import advance_playing_steps, wait_until_playing
from robot.scene.loader import load_robot_scene
from robot.scene.restaurant_risk_scenarios import (
    primary_restaurant_risk_scenarios,
)
from robot.scene.restaurant_scenario_authoring import (
    author_restaurant_risk_scenario,
)
from robot.scene.transforms import get_prim_planar_pose


OUTPUT_DIRECTORY = ROOT / "outputs" / "restaurant_scenario_authoring"
ROBOT_START_POSITION_WORLD = (26.4955, 5.00175, 0.07382)
ROBOT_START_YAW_RADIANS = 0.0
MAXIMUM_POSITION_DRIFT_M = 0.02
MAXIMUM_YAW_DRIFT_RADIANS = math.radians(1.0)


def _yaw_quaternion_wxyz(yaw_radians: float) -> list[float]:
    half_yaw = 0.5 * yaw_radians
    return [math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)]


def _wrapped_angle_difference(first: float, second: float) -> float:
    return abs(math.atan2(math.sin(first - second), math.cos(first - second)))


def _actor_payload(actor) -> dict[str, object]:
    return {
        "actor_id": actor.actor_id,
        "prim_path": actor.prim_path,
        "geometry": actor.geometry.value,
        "placement_anchor": actor.placement_anchor.value,
        "target_anchor_world_xyz_m": actor.target_anchor_world_xyz_m,
        "placed_anchor_world_xyz_m": actor.placed_anchor_world_xyz_m,
        "bounds_world_minimum_m": actor.bounds_world.minimum,
        "bounds_world_maximum_m": actor.bounds_world.maximum,
        "configured_velocity_world_xy_mps": (
            actor.configured_velocity_world_xy_mps
        ),
        "actor_motion_enabled": actor.motion_enabled,
        "evidence_provenance": "oracle_fixture_only",
    }


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    loaded = load_robot_scene(
        scene_path=ROBOT_SCENE_PATH,
        simulation_app=simulation_app,
        robot_prim_path=ROBOT_PRIM_PATH,
        expected_meters_per_unit=EXPECTED_METERS_PER_UNIT,
    )
    controller = NovaCarterController(robot_prim_path=ROBOT_PRIM_PATH)
    SimulationManager.setup_simulation(dt=PHYSICS_DT, device=PHYSICS_DEVICE)
    app_utils.play()
    if not wait_until_playing(simulation_app):
        raise RuntimeError("Isaac playback did not start")
    app_utils.update_app(steps=10)
    controller.verify_articulation()
    controller.robot.set_world_poses(
        positions=[ROBOT_START_POSITION_WORLD],
        orientations=[_yaw_quaternion_wxyz(ROBOT_START_YAW_RADIANS)],
    )
    if not advance_playing_steps(
        simulation_app,
        physics_steps=60,
        on_step=controller.stop,
    ):
        raise RuntimeError("Isaac closed during the stationary settle period")
    controller.stop()
    app_utils.pause()
    simulation_app.update()
    initial_pose = get_prim_planar_pose(loaded.stage, ROBOT_BASE_PRIM_PATH)
    # Isaac may author runtime articulation state while physics initializes.
    # Snapshot only after setup so this check isolates scenario authoring.
    root_before_authoring = loaded.stage.GetRootLayer().ExportToString()

    reports = []
    for scenario in primary_restaurant_risk_scenarios():
        result = author_restaurant_risk_scenario(
            loaded.stage,
            scenario,
            robot_pose_world_xy_yaw=initial_pose,
            root_prim_path=(
                f"/World/CRARestaurantScenario_{scenario.scenario_id}"
            ),
        )
        reports.append(
            {
                "scenario_id": result.scenario_id,
                "factor": result.factor.value,
                "runtime_sensor_claims": result.runtime_sensor_claims,
                "robot_motion_enabled": result.robot_motion_enabled,
                "actor_motion_enabled": result.actor_motion_enabled,
                "actors": [_actor_payload(actor) for actor in result.actors],
            }
        )

    final_pose = get_prim_planar_pose(loaded.stage, ROBOT_BASE_PRIM_PATH)
    position_drift = math.hypot(
        final_pose[0] - initial_pose[0],
        final_pose[1] - initial_pose[1],
    )
    yaw_drift = _wrapped_angle_difference(final_pose[2], initial_pose[2])
    blockers = []
    if position_drift > MAXIMUM_POSITION_DRIFT_M:
        blockers.append("robot position drift exceeded stationary tolerance")
    if yaw_drift > MAXIMUM_YAW_DRIFT_RADIANS:
        blockers.append("robot yaw drift exceeded stationary tolerance")
    if loaded.stage.GetRootLayer().ExportToString() != root_before_authoring:
        blockers.append("root layer changed during session-only authoring")

    payload = {
        "passed": not blockers,
        "blockers": blockers,
        "scenario_count": len(reports),
        "robot_initial_pose_world_xy_yaw": initial_pose,
        "robot_final_pose_world_xy_yaw": final_pose,
        "robot_position_drift_m": position_drift,
        "robot_yaw_drift_degrees": math.degrees(yaw_drift),
        "wheel_command": "zero_only",
        "runtime_sensor_evidence": "not_evaluated",
        "oracle_boundary": (
            "fixtures are coauthored for placement validation only; roles, "
            "bounds, and configured velocities are unavailable to runtime CRA"
        ),
        "perception_evaluation_valid": False,
        "scenarios": reports,
    }
    (OUTPUT_DIRECTORY / "report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    loaded.stage.GetSessionLayer().Export(
        str(OUTPUT_DIRECTORY / "authored_session.usda")
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    if blockers:
        raise RuntimeError("stationary restaurant scenario authoring failed")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()

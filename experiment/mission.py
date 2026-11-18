"""Run the selected restaurant service phases in one live Isaac session."""

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import sys
import traceback


ROOT = Path(__file__).resolve().parents[1]
PHASE_ORDER = ("left", "return", "right")
PHASE_RUNTIME = "experiment/runtime/service_phase.py"
COMMON_PICKUP = "--left-outbound-common-pickup"
PRELOADED_WEST_PICKUP = "--left-outbound-preloaded-west"
LIDAR_GUIDED_PROP_MASKS = "--prop-masks-lidar-guided-only"
ANIMATED_RESTAURANT = "--restaurant-background-motion"
RIGHT_TABLE_FRONT = "--right-table-front"
PHASE_SELECTORS = {
    "left": ("--left-outbound-lidar-full-delivery",),
    "return": ("--return-to-pickup",),
    "right": ("--right-crossing-trial", "delivery"),
}


@dataclass
class LiveMission:
    output: Path
    requested_phases: tuple = PHASE_ORDER
    case: str | None = None
    shared: dict = field(default_factory=dict)
    phases: list = field(default_factory=list)
    meal_events: list = field(default_factory=list)


def save(path, record):
    data = json.dumps(record, indent=2, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(data, encoding="utf-8")
    temporary.replace(path)


def read_recipe(path, scene_root):
    data = json.loads(path.read_text(encoding="utf-8"))
    requested = data.get("requested_phases", list(PHASE_ORDER))
    recipes = data.get("phase_recipes", {})
    left_start = data.get("left_start")
    left_prop_mask_search = data.get("left_prop_mask_search", "full")
    background_motion = data.get("background_motion", False)
    if (data.get("schema_version") != 1 or requested != list(PHASE_ORDER)
            or set(recipes) != set(requested)):
        raise ValueError("Mission recipe must declare left, return and right")
    if left_start not in (None, "preloaded_west"):
        raise ValueError("Unknown left pickup setup")
    if left_prop_mask_search not in ("full", "lidar_guided_only"):
        raise ValueError("Unknown left prop mask search")
    if not isinstance(background_motion, bool):
        raise ValueError("Continuous background motion requires the three-leg restaurant mission")
    phase_commands, hashes = {}, {}
    scene = None
    for name in requested:
        recipe = recipes[name]
        recipe_path = (ROOT / recipe["path"]).resolve()
        if not recipe_path.is_relative_to(ROOT / "experiment" / "configs"):
            raise ValueError(f"The {name} recipe must be a repository config")
        contents = recipe_path.read_bytes()
        hashes[recipe_path.relative_to(ROOT).as_posix()] = hashlib.sha256(contents).hexdigest()
        command = json.loads(contents)["commands"][recipe["case"]]
        if (not command or command[0] != PHASE_RUNTIME or command.count("--restaurant-scene") != 1
            or {"--output", "--motion", COMMON_PICKUP, PRELOADED_WEST_PICKUP,
                LIDAR_GUIDED_PROP_MASKS, ANIMATED_RESTAURANT} & set(command)):
            raise ValueError(f"Invalid runtime command in the {name} recipe")
        selected = PHASE_SELECTORS[name]
        index = command.index(selected[0]) if selected[0] in command else -1
        if index < 0 or tuple(command[index:index + len(selected)]) != selected:
            raise ValueError(f"The {name} recipe selects the wrong phase")
        # Each path in a phase recipe is relative to the prepared scene bundle.
        for flag in ("--restaurant-scene", "--restaurant-layout-survey",
                     "--left-outbound-cast-manifest", "--right-crossing-cast-manifest"):
            if flag in command:
                index = command.index(flag) + 1
                target = (scene_root / command[index]).resolve()
                if not target.is_relative_to(scene_root):
                    raise ValueError(f"{name} {flag} is outside the scene bundle")
                command[index] = str(target)
        phase_scene = Path(command[command.index("--restaurant-scene") + 1])
        if scene is None:
            scene = phase_scene
        elif phase_scene != scene:
            raise ValueError(f"The {name} recipe references a different scene")
        if name == "left" and left_start == "preloaded_west":
            command = [*command, PRELOADED_WEST_PICKUP]
        if name == "left" and left_prop_mask_search == "lidar_guided_only":
            if "--stationary-image-objects" not in command:
                raise ValueError("LiDAR-guided prop masks require the left image-object experiment")
            command = [*command, LIDAR_GUIDED_PROP_MASKS]
        if RIGHT_TABLE_FRONT in command and (name != "right" or not background_motion):
            raise ValueError("The continuous table-front finish requires the animated right delivery")
        if background_motion:
            command = [*command, ANIMATED_RESTAURANT]
        phase_commands[name] = command
    return phase_commands, hashes


def verify_launcher_receipt(output, mission_config, hashes):
    path = output / "invocation.json"
    if not path.is_file():
        return
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if (receipt.get("referenced_config_sha256") != hashes
        or receipt.get("config_sha256") != hashlib.sha256(mission_config.read_bytes()).hexdigest()):
        raise RuntimeError("Mission and launcher configuration receipts differ")


def completed_phase(mission, name, phase_output):
    """Check native phase output and the separately recorded continuity receipt."""
    expected = PHASE_ORDER.index(name) + 1
    if len(mission.phases) != expected:
        raise RuntimeError(f"The {name} phase must append exactly one native receipt")
    receipt = mission.phases[-1]
    if receipt.get("case") != name or receipt.get("passed") is not True:
        raise RuntimeError(f"The {name} phase did not return a passing receipt")
    keys = ("start_simulation_seconds", "navigation_start_seconds", "end_simulation_seconds")
    times = [receipt.get(key) for key in keys]
    if (not all(type(t) in (int, float) and math.isfinite(t) for t in times)
        or not 0. <= times[0] <= times[1] <= times[2]):
        raise RuntimeError(f"The {name} phase did not preserve ordered native clocks")
    if expected > 1 and (receipt.get("tracker_reset") is not True
                         or times[0] < mission.phases[-2]["end_simulation_seconds"] - 1e-8):
        raise RuntimeError(f"The {name} phase did not preserve a fresh forward boundary")
    start_pose = receipt["start_world_xy_yaw"]
    end_pose = receipt["end_world_xy_yaw"]
    if any(len(pose) != 3 or not all(math.isfinite(value) for value in pose)
           for pose in (start_pose, end_pose)):
        raise RuntimeError(f"The {name} phase did not record valid native chassis poses")
    if expected > 1:
        previous_end = mission.phases[-2]["end_world_xy_yaw"]
        translation = math.dist(start_pose[:2], previous_end[:2])
        yaw = abs(math.remainder(start_pose[2] - previous_end[2], math.tau))
        receipt["boundary_drift_m"] = translation
        receipt["boundary_yaw_drift_degrees"] = math.degrees(yaw)
        if translation > .01 or yaw > math.radians(1.):
            raise RuntimeError(f"Carter moved during the zero-command transition into {name}")
    report = json.loads((phase_output / "report.json").read_text(encoding="utf-8"))
    if report.get("passed") is not True or report.get("reason") != "goal_reached":
        raise RuntimeError(f"The {name} native report did not reach its verified goal")


def restore_process_state(argv, path, cwd):
    sys.argv[:] = argv
    sys.path[:] = path
    os.chdir(cwd)


def run_mission(output, commands, hashes):
    requested = tuple(commands)
    mission = LiveMission(output, requested_phases=requested)
    argv, path, cwd = sys.argv.copy(), sys.path.copy(), Path.cwd()
    runtime = ROOT / PHASE_RUNTIME
    failure = None
    try:
        for name in requested:
            command = commands[name]
            mission.case = name
            phase_output = output / name
            if phase_output.exists():
                raise RuntimeError(f"Preserve the existing {name} phase output")
            sys.argv[:] = [str(runtime), *command[1:], "--motion", "--output", str(phase_output)]
            phase_output.mkdir()
            save(phase_output / "invocation.json", dict(case=name, command=sys.argv.copy(),
                 phase_recipe_sha256=hashes))
            try:
                runpy.run_path(str(runtime), run_name="__main__", init_globals={"LIVE_MISSION": mission})
                completed_phase(mission, name, phase_output)
            finally:
                restore_process_state(argv, path, cwd)
            save(output / "mission-phases.json", dict(schema_version=1, passed=False,
                 requested_phases=requested, phases=mission.phases, meal_events=mission.meal_events,
                 phase_recipe_sha256=hashes))
        names = {"meal_on_physical_tray", "meal_placed_on_left_table", "meal_placed_on_right_table"}
        order = [(row["case"], row["event"]) for row in mission.meal_events if row["event"] in names]
        expected_order = [
            ("left", "meal_on_physical_tray"), ("left", "meal_placed_on_left_table"),
            ("right", "meal_on_physical_tray"), ("right", "meal_placed_on_right_table"),
        ]
        if order != expected_order:
            raise RuntimeError("Missing or out-of-order live mission meal receipts")
        times = [row["simulation_time_seconds"] for row in mission.meal_events]
        if not all(type(t) in (int, float) and math.isfinite(t) for t in times) or times != sorted(times):
            raise RuntimeError("Mission meal receipts must have ordered native timestamps")
        phases = {row["case"]: row for row in mission.phases}
        for meal in mission.meal_events:
            phase = phases[meal["case"]]
            when = meal["simulation_time_seconds"]
            if not phase["start_simulation_seconds"] <= when <= phase["end_simulation_seconds"]:
                raise RuntimeError("A meal transfer lies outside its recorded phase")
    except (Exception, SystemExit) as error:
        failure = dict(phase=mission.case, type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        cleanup_errors = []
        video = mission.shared.pop("service_video", None)
        if video is not None:
            try:
                video.close()
            except Exception as error:
                cleanup_errors.append(dict(resource="service_video", type=type(error).__name__, message=str(error)))
                traceback.print_exc()
        restore_process_state(argv, path, cwd)
    accepted = failure is None and not cleanup_errors and len(mission.phases) == len(requested)
    status = dict(passed=accepted, completed=accepted,
                  reason="mission_complete" if accepted else "mission_phase_failed" if failure else "mission_cleanup_failed",
                  requested_phases=requested, phase_recipe_sha256=hashes, phases=mission.phases, meal_events=mission.meal_events,
                  failure=failure, cleanup_errors=cleanup_errors,
                  scope="One robot, Isaac stage, timeline and native recording. Phase receipts retain original live sensor clocks.")
    try:
        save(output / "mission-phases.json", dict(schema_version=1, passed=accepted,
             requested_phases=requested, phases=mission.phases,
             meal_events=mission.meal_events, failure=failure, phase_recipe_sha256=hashes))
        save(output / "report.json", status)
    finally:
        # Kit's fast shutdown can exit the process inside close().
        app = mission.shared.pop("app", None)
        if app is not None:
            app.close(exit_code=0 if accepted else 1)
    return 0 if accepted else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = ROOT / "experiment/configs/mission.json"
    scene_root = args.scene_root.resolve()
    output = args.output.resolve()
    os.chdir(ROOT)
    commands, hashes = read_recipe(config, scene_root)
    verify_launcher_receipt(output, config, hashes)
    return run_mission(output, commands, hashes)


if __name__ == "__main__":
    raise SystemExit(main())

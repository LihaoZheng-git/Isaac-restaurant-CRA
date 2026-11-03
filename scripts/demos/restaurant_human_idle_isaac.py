"""Actor-only long-wait/release audit; robot removed from this diagnostic session."""

import argparse
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--scene', type=Path, required=True)
parser.add_argument('--layout-survey', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--background-service-walk', action='store_true')
parser.add_argument('--background-upper-departure', action='store_true')
args = parser.parse_args()
if args.background_upper_departure and not args.background_service_walk:
    parser.error('upper departure requires background service walk')
if args.output.exists():
    parser.error('preserve existing outputs')
args.output.mkdir(parents=True)
# Resolve the existing environment pair before Kit injects its legacy archive.
# Episode imports share risk dataclasses even though this audit runs no model.
importlib.import_module('robot.perception.visual_mass_runtime')

from isaacsim import SimulationApp

app = SimulationApp({'headless': True, 'enable_motion_bvh': True, 'extra_args': [
    '--enable', 'omni.anim.behavior.core', '--/rtx/scenedb/maxHistoryTransformCount=128']})
import carb.settings
import numpy as np
import omni.timeline
import omni.usd
from isaacsim.core.simulation_manager import SimulationManager
from pxr import Usd
from robot.config import PHYSICS_DT, PHYSICS_DEVICE, ROBOT_PRIM_PATH
from robot.evaluation.restaurant_service_validation import load_authored_layout_monitor
from robot.scene.controlled_semantic_obstacle import read_world_bounds
from robot.scene.restaurant_demo_build import demo_prop_records
from robot.scene.restaurant_human_episode import RestaurantHumanEpisode
from robot.scene.restaurant_service_build import SERVICE_ROOT, service_props


def save(name, value):
    (args.output/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


human = None
try:
    if not omni.usd.get_context().open_stage(str(args.scene.resolve())):
        raise RuntimeError('scene unavailable')
    for _ in range(100):
        app.update()
    stage = omni.usd.get_context().get_stage()
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        stage.GetPrimAtPath(ROBOT_PRIM_PATH).SetActive(False)
    additions = demo_prop_records(stage)
    for prop in service_props():
        b = read_world_bounds(stage, SERVICE_ROOT+'/'+prop.name)
        additions.append(dict(bounds_minimum=b.minimum, bounds_maximum=b.maximum))
    layout = load_authored_layout_monitor(args.layout_survey, additions)
    human = RestaurantHumanEpisode(stage, mode='parallel', output=args.output,
        trigger_robot_y=9., speed_mps=.5, long_service_parallel=True,
        background_service_walk=args.background_service_walk,
        background_upper_departure=args.background_upper_departure)
    for _ in range(20):
        app.update()
    human.bake()
    SimulationManager.setup_simulation(dt=PHYSICS_DT, device=PHYSICS_DEVICE)
    carb.settings.get_settings().set('/app/player/useFixedTimeStepping', True)
    timeline = omni.timeline.get_timeline_interface()
    timeline.set_play_every_frame(True)
    timeline.set_time_codes_per_second(60)
    timeline.play()
    for _ in range(10):
        app.update()
    human.initialize()
    for _ in range(60):
        app.update()
    human.hold_initial_idle()
    start = float(timeline.get_current_time())
    origin = human.position()
    for step in range(180*60):
        app.update()
        human.update(float(timeline.get_current_time()), (7.8, 4.05), (0., 0.), layout)
        if step % 1800 == 0:
            print(f'Native held idle: {step/60:.1f} seconds', flush=True)
    held = list(human.records)
    release = float(timeline.get_current_time())
    human.start_video()
    for step in range((210 if args.background_upper_departure else 180 if args.background_service_walk else 20)*60):
        # The supplied position triggers only the authored actor, no robot exists.
        elapsed = float(timeline.get_current_time())-release
        human.update(float(timeline.get_current_time()),
                     (7.8, 4.05) if args.background_service_walk and elapsed < 120 else (28.8, 9.),
                     (0., 0.), layout,
                     navigation=True)
        app.update()
        if step % 1800 == 0:
            print(f'Native released walk: {elapsed:.1f} seconds; phase {human.phase}', flush=True)
        if human.task_succeeded:
            break
    walked = human.records[len(held):]
    drift = max(float(np.linalg.norm(np.array(r['actor_xyz'])-origin)) for r in held)
    travel = float(np.linalg.norm(human.position()[:2]-origin[:2]))
    checks = dict(hold_at_least_180_seconds=release-start >= 180-1e-5,
        held_agent_disabled=all(r['initial_idle_held'] and not r['native_agent_enabled'] for r in held),
        idle_drift_below_1mm=drift <= .001,
        released_native_walking=bool(walked) and all(r['native_agent_enabled'] and not r['initial_idle_held'] for r in walked),
        native_task_completed=human.task_succeeded, native_travel_over_3m=travel >= 3.,
        unchanged_geometry_guard=bool(human.records))
    if args.background_service_walk:
        from robot.scene.restaurant_background_walk import audit_background_walk
        circuit = audit_background_walk(walked, human.background_walk.events,
                                        minimum_background_seconds=120.,
                                        upper_departure=args.background_upper_departure)
        checks = {k: checks[k] for k in ('hold_at_least_180_seconds',
                  'held_agent_disabled', 'idle_drift_below_1mm')}
        checks.update(circuit['checks'])
        save('background-walk-validation.json', circuit)
        save('background-walk-events.json', human.background_walk.events)
    save('validation.json', dict(passed=all(checks.values()), checks=checks,
        hold_seconds=release-start, idle_drift_m=drift, native_travel_m=travel,
        minimum_layout_clearance_m=min(r['actor_layout_clearance_m'] for r in human.records),
        scope='Actor-only held native idle then released walk. Robot inactive in unsaved session; no robot perception/navigation acceptance. No teleport or relaxed clearance.'))
    print(json.dumps(dict(checks=checks, drift_m=drift, travel_m=travel)), flush=True)
finally:
    if human is not None:
        save('human-truth-evaluation-only.json', human.records)
        human.close()
    app.close()

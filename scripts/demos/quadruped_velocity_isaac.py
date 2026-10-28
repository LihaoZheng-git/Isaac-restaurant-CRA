"""Separate official-Spot velocity interface bench; never commands Carter.

Standing is mandatory before tracking. No CRA, obstacle avoidance, restaurant
navigation, or ideal reduced-order tracking is claimed by this bench.
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=('standing', 'tracking'), default='standing')
    parser.add_argument('--asset-manifest', type=Path, required=True)
    parser.add_argument('--standing-report', type=Path)
    return parser.parse_args()


def source_hashes():
    paths = ['scripts/demos/quadruped_velocity_isaac.py',
        'robot/control/planar_velocity_command.py', 'robot/evaluation/quadruped_feasibility.py']
    return {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}


def main():
    args = parse_args()
    if args.output.exists():
        raise ValueError('fresh output directory required; preserve earlier failures')
    manifest = json.loads(args.asset_manifest.read_text())
    assets = {Path(x['path']).name: x for x in manifest['files']}
    for entry in assets.values():
        if hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('asset bytes differ from audited official download')
    if args.stage == 'tracking':
        if args.standing_report is None:
            raise ValueError('independent standing report required before tracking')
        prior = json.loads(args.standing_report.read_text())
        if (not prior['passed'] or prior['stage'] != 'standing'
            or prior['asset_sha256'] != {k: v['sha256'] for k, v in assets.items()}
            or prior['source_files_sha256'] != source_hashes()):
            raise ValueError('matching passed standing gate required; no stale contract acceptance')
    args.output.mkdir(parents=True)

    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'width': 960, 'height': 640})
    try:
        import cv2
        import numpy as np
        import omni.client
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from isaacsim.core.deprecation_manager import import_module
        from isaacsim.core.rendering_manager import RenderingManager
        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents
        from isaacsim.robot.policy.examples.robots.spot import SpotFlatTerrainPolicy
        from isaacsim.storage.native import get_assets_root_path
        from pxr import Gf, UsdGeom, UsdLux, UsdPhysics, UsdShade
        from robot.control.planar_velocity_command import bounded_body_velocity
        from robot.evaluation.quadruped_feasibility import root_observation, summarize_standing, summarize_tracking

        torch = import_module('torch')
        root_url = get_assets_root_path()+'/Isaac/Robots/BostonDynamics/spot/spot.usd'
        result, _, content = omni.client.read_file(root_url)
        if result != omni.client.Result.OK or hashlib.sha256(bytes(content)).hexdigest() != assets['spot.usd']['sha256']:
            raise ValueError('runtime Spot root USD differs from audited asset')
        stage = omni.usd.get_context().get_stage()
        UsdPhysics.Scene.Define(stage, '/World/PhysicsScene')
        floor = UsdGeom.Cube.Define(stage, '/World/Floor')
        floor.CreateSizeAttr(1.)
        floor.AddTranslateOp().Set(Gf.Vec3d(0, 0, -.05))
        floor.AddScaleOp().Set(Gf.Vec3f(20, 20, .1))
        floor.CreateDisplayColorAttr([Gf.Vec3f(.35, .4, .45)])
        UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
        material = UsdShade.Material.Define(stage, '/World/GroundMaterial')
        physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics_material.CreateStaticFrictionAttr(1.)
        physics_material.CreateDynamicFrictionAttr(1.)
        physics_material.CreateRestitutionAttr(0.)
        UsdShade.MaterialBindingAPI.Apply(floor.GetPrim()).Bind(material, materialPurpose='physics')
        light = UsdLux.DomeLight.Define(stage, '/World/Light')
        light.CreateIntensityAttr(1500)
        camera = UsdGeom.Camera.Define(stage, '/World/BenchCamera')
        matrix = Gf.Matrix4d().SetLookAt(Gf.Vec3d(3, -3, 2), Gf.Vec3d(.3, 0, .4), Gf.Vec3d(0, 0, 1)).GetInverse()
        camera.AddTransformOp().Set(matrix)
        camera.CreateClippingRangeAttr(Gf.Vec2f(.01, 100.))
        camera.CreateFocalLengthAttr(20.)
        render = rep.create.render_product(str(camera.GetPath()), (960, 640))
        rgb = rep.AnnotatorRegistry.get_annotator('rgb')
        rgb.attach([render])
        SimulationManager.set_backend('numpy')
        SimulationManager.set_physics_sim_device('cpu')
        SimulationManager.set_physics_dt(.002)
        RenderingManager.set_dt(.02)
        # Use the stock official USD path. This installed wrapper's explicit
        # usd_path branch has an uninitialized is_newton local; do not patch it.
        spot = SpotFlatTerrainPolicy(prim_path='/World/Spot', position=[0., 0., .8],
            policy_path=str(Path(assets['spot_policy.pt']['path']).resolve()),
            env_config_path=str(Path(assets['spot_env.yaml']['path']).resolve()))
        if spot._dt != .002 or spot._decimation != 10:
            raise ValueError('policy timing differs from audited 500Hz/50Hz training contract')
        timeline = omni.timeline.get_timeline_interface()
        phases = [('settle', 2., [0., 0., 0.]), ('standing', 3.2, [0., 0., 0.])]
        if args.stage == 'tracking':
            phases += [('move_forward', 4., [.2, 0., 0.]), ('stop_forward', 2., [0., 0., 0.]),
                ('move_lateral', 4., [0., .15, 0.]), ('stop_lateral', 2., [0., 0., 0.]),
                ('move_yaw', 3., [0., 0., .3]), ('stop_yaw', 2., [0., 0., 0.]),
                ('move_combined', 4., [.15, .1, 0.]), ('stop_combined', 3., [0., 0., 0.])]
        boundaries = np.cumsum([p[1] for p in phases])
        records, failures = [], []
        initialized, finished, ticks = False, False, 0
        own_standing = None

        def quaternion_state(quat):
            w, x, y, z = quat/np.linalg.norm(quat)
            rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
            angles = [math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
                math.asin(float(np.clip(2*(w*y-z*x), -1, 1))),
                math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))]
            return rotation, angles

        def on_step(dt, _context):
            nonlocal initialized, finished, ticks, own_standing
            try:
                if not initialized:
                    spot.initialize()
                    initialized = True
                    return
                now = float(timeline.get_current_time())
                index = int(np.searchsorted(boundaries, now, side='right'))
                if index >= len(phases):
                    finished = True
                    return
                phase, _, command = phases[index]
                if index >= 2 and own_standing is None:
                    own_standing = summarize_standing([r for r in records if r['phase'] == 'standing'])
                    if not own_standing['passed']:
                        finished = True
                        return
                cmd = bounded_body_velocity(command)
                spot.forward(dt, torch.tensor(cmd, dtype=torch.float32))
                if ticks % 10 == 0:
                    position, orientation = spot.robot.get_world_poses()
                    linear, angular = spot.robot.get_velocities()
                    position, quat = position.numpy()[0], orientation.numpy()[0]
                    rotation, angles = quaternion_state(quat)
                    velocity = rotation.T@linear.numpy()[0]
                    angular_body = rotation.T@angular.numpy()[0]
                    records.append(root_observation(time=now, phase=phase,
                        phase_elapsed=now-(boundaries[index-1] if index else 0.),
                        command_body=cmd.tolist(), position_world=position.tolist(),
                        roll_pitch_yaw=angles, velocity_body=[*velocity[:2], angular_body[2]]))
                    if (not np.isfinite(position).all() or position[2] < .25
                        or max(abs(a) for a in angles[:2]) > math.radians(25)):
                        failures.append('unsafe_or_nonfinite_root_state')
                        finished = True
                ticks += 1
            except Exception as error:
                failures.append(repr(error))
                finished = True

        callback = SimulationManager.register_callback(on_step, IsaacEvents.POST_PHYSICS_STEP)
        video = cv2.VideoWriter(str(args.output/'native.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 10., (960, 640))
        last_frame_time = -1.
        timeline.play()
        while app.is_running() and not finished:
            app.update()
            now = float(timeline.get_current_time())
            if now-last_frame_time >= .1-1e-9:
                pixels = np.asarray(rgb.get_data())
                if pixels.shape == (640, 960, 4):
                    image = cv2.cvtColor(pixels, cv2.COLOR_RGBA2BGR)
                    video.write(image)
                    cv2.imwrite(str(args.output/'native-current.png'), image)
                last_frame_time = now
        timeline.stop()
        SimulationManager.deregister_callback(callback)
        video.release()
        if own_standing is None and len([r for r in records if r['phase'] == 'standing']) >= 30:
            own_standing = summarize_standing([r for r in records if r['phase'] == 'standing'])
        tracking = summarize_tracking(records) if args.stage == 'tracking' and own_standing and own_standing['passed'] and not failures else None
        report = dict(stage=args.stage, passed=bool(own_standing and own_standing['passed'] and not failures
                and (args.stage == 'standing' or tracking and tracking['passed'])),
            standing=own_standing, tracking=tracking, failures=failures,
            asset_sha256={k: v['sha256'] for k, v in assets.items()},
            source_files_sha256=source_hashes(), policy_class='stock SpotFlatTerrainPolicy',
            physics_engine=SimulationManager.get_active_physics_engine(), physics_dt=.002, policy_decimation=10,
            asset_root_url=root_url, used_usd_layers=[layer.identifier for layer in stage.GetUsedLayers()],
            scope='Separate freely standing/walking open-space interface bench; no robot replacement, CRA, perception or navigation acceptance')
        (args.output/'records.json').write_text(json.dumps(records))
        (args.output/'report.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
    except Exception as error:
        import traceback
        failure = dict(stage=args.stage, passed=False, error=repr(error), traceback=traceback.format_exc(),
            source_files_sha256=source_hashes(), scope='Native bench failed; no standing or movement acceptance')
        (args.output/'report.json').write_text(json.dumps(failure, indent=2))
        print(json.dumps(failure), flush=True)
    finally:
        app.close()


if __name__ == '__main__':
    main()

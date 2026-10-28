"""Separate velocity-interface calibration bench; no navigation controller.

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
    parser.add_argument('--settle-seconds', type=float, default=2.)
    parser.add_argument('--robot', choices=('spot', 'go2', 'carter'), default='spot')
    return parser.parse_args()


def source_hashes():
    paths = ['scripts/demos/quadruped_velocity_isaac.py',
        'robot/control/planar_velocity_command.py', 'robot/evaluation/quadruped_feasibility.py',
        'robot/control/differential_drive.py', 'robot/config.py']
    return {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}


def main():
    args = parse_args()
    if not math.isfinite(args.settle_seconds) or not 2. <= args.settle_seconds <= 15.:
        raise ValueError('explicit initial settling period must be within 2..15s')
    if args.output.exists():
        raise ValueError('fresh output directory required; preserve earlier failures')
    manifest = json.loads(args.asset_manifest.read_text())
    if manifest.get('robot', 'spot') != args.robot:
        raise ValueError('explicit robot differs from audited asset manifest')
    assets = {Path(x['path']).name: x for x in manifest['files']}
    specification = {
        'spot': ('/Isaac/Robots/BostonDynamics/spot/spot.usd', 'spot.usd', 'spot_policy.pt', 'spot_env.yaml', .002, 10),
        'go2': ('/Isaac/Samples/Mujoco_Menagerie/unitree_go2/go2/go2.usda', 'go2.usda', 'physx_policy.pt', 'physx_env.yaml', .005, 4),
        'carter': ('/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd', 'nova_carter.usd', None, None, 1/60, 1),
    }[args.robot]
    usd_relative, usd_name, policy_name, env_name, physics_dt, decimation = specification
    for entry in assets.values():
        if hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('asset bytes differ from audited official download')
    if args.stage == 'tracking':
        if args.standing_report is None:
            raise ValueError('independent standing report required before tracking')
        prior = json.loads(args.standing_report.read_text())
        if (not prior['passed'] or prior['stage'] != 'standing'
            or prior.get('robot') != args.robot
            or prior.get('settle_seconds') != args.settle_seconds
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
        from isaacsim.robot.policy.examples.robots.go2 import Go2FlatTerrainPolicy
        from isaacsim.storage.native import get_assets_root_path
        from pxr import Gf, PhysxSchema, UsdGeom, UsdLux, UsdPhysics, UsdShade
        from robot.control.planar_velocity_command import bounded_body_velocity
        from robot.evaluation.quadruped_feasibility import root_observation, summarize_standing, summarize_tracking
        from robot.control.differential_drive import NovaCarterController

        torch = import_module('torch')
        root_url = get_assets_root_path()+usd_relative
        result, _, content = omni.client.read_file(root_url)
        if result != omni.client.Result.OK or hashlib.sha256(bytes(content)).hexdigest() != assets[usd_name]['sha256']:
            raise ValueError('runtime quadruped root USD differs from audited asset')
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
        physx_material = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
        physx_material.CreateFrictionCombineModeAttr('multiply')
        physx_material.CreateRestitutionCombineModeAttr('multiply')
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
        SimulationManager.set_physics_dt(physics_dt)
        RenderingManager.set_dt(.02)
        # Use the stock official USD path. This installed wrapper's explicit
        # usd_path branch has an uninitialized is_newton local; do not patch it.
        spot, drive = None, None
        if args.robot == 'carter':
            initial_height = .07382
            prim = UsdGeom.Xform.Define(stage, '/World/Robot')
            prim.GetPrim().GetReferences().AddReference(root_url)
            transform = UsdGeom.Xformable(prim)
            transform.ClearXformOpOrder()
            transform.AddTransformOp(opSuffix='benchStart').Set(Gf.Matrix4d().SetTranslate(Gf.Vec3d(0, 0, initial_height)))
            drive = NovaCarterController(robot_prim_path='/World/Robot')
            robot = drive.robot
            height_bounds = dict(minimum_base_height_m=.03, maximum_base_height_m=.2)
            axes = ('forward', 'yaw')
        else:
            policy_class = SpotFlatTerrainPolicy if args.robot == 'spot' else Go2FlatTerrainPolicy
            initial_height = .5 if args.robot == 'spot' else .4
            spot = policy_class(prim_path='/World/Quadruped', position=[0., 0., initial_height],
                policy_path=str(Path(assets[policy_name]['path']).resolve()),
                env_config_path=str(Path(assets[env_name]['path']).resolve()))
            if spot._dt != physics_dt or spot._decimation != decimation:
                raise ValueError('policy timing differs from audited physics/policy-rate contract')
            if spot.policy_env_params['scene']['robot']['init_state']['pos'][2] != initial_height:
                raise ValueError('spawn height differs from audited training initial state')
            robot = spot.robot
            height_bounds = {}
            axes = ('forward', 'lateral', 'yaw', 'combined')
        timeline = omni.timeline.get_timeline_interface()
        phases = [('settle', args.settle_seconds, [0., 0., 0.]), ('standing', 3.2, [0., 0., 0.])]
        if args.stage == 'tracking':
            phases += [('move_forward', 4., [.2, 0., 0.]), ('stop_forward', 2., [0., 0., 0.]),
                ('move_lateral', 4., [0., .15, 0.]), ('stop_lateral', 2., [0., 0., 0.]),
                ('move_yaw', 3., [0., 0., .3]), ('stop_yaw', 2., [0., 0., 0.]),
                ('move_combined', 4., [.15, .1, 0.]), ('stop_combined', 3., [0., 0., 0.])]
            phases = [phase for phase in phases if not phase[0].startswith(('move_', 'stop_')) or phase[0].split('_', 1)[1] in axes]
        boundaries = np.cumsum([p[1] for p in phases])
        records, failures = [], []
        initialized, finished, ticks = False, False, 0
        own_standing = None
        initialization = {}

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
                    old_position, old_orientation = robot.get_world_poses()
                    old_linear, old_angular = robot.get_velocities()
                    if spot is not None:
                        spot.initialize()
                    else:
                        drive.verify_articulation()
                    # Physics warm-up has already run on the authored posture.
                    # Restore the configured initial state once, before any
                    # assessment, and install targets before the next tick.
                    robot.set_world_poses(positions=[[0., 0., initial_height]],
                        orientations=[[1., 0., 0., 0.]])
                    robot.set_velocities(linear_velocities=[[0., 0., 0.]],
                        angular_velocities=[[0., 0., 0.]])
                    if spot is not None:
                        robot.set_dof_position_targets(spot.default_pos.detach().cpu().numpy())
                        spot.forward(dt, torch.zeros(3, dtype=torch.float32))
                    else:
                        drive.stop()
                    initialization.update(timestamp=float(SimulationManager.get_simulation_time()),
                        old_root_position=old_position.numpy()[0].tolist(),
                        old_root_orientation=old_orientation.numpy()[0].tolist(),
                        old_linear_velocity=old_linear.numpy()[0].tolist(),
                        old_angular_velocity=old_angular.numpy()[0].tolist(),
                        restored_root_position=[0., 0., initial_height],
                        restored_root_orientation=[1., 0., 0., 0.],
                        restored_root_velocity=[0., 0., 0., 0., 0., 0.],
                        dof_names=list(robot.dof_names),
                        default_joint_positions=robot.get_dof_positions().numpy()[0].tolist(),
                        scope='one startup restore after native warm-up; no repeated state reset, mount or measurement-clock renewal')
                    (args.output/'initialization.json').write_text(json.dumps(initialization, indent=2))
                    initialized = True
                    return
                now = float(SimulationManager.get_simulation_time())
                index = int(np.searchsorted(boundaries, now, side='right'))
                if index >= len(phases):
                    finished = True
                    return
                phase, _, command = phases[index]
                if index >= 2 and own_standing is None:
                    own_standing = summarize_standing([r for r in records if r['phase'] == 'standing'], **height_bounds)
                    if not own_standing['passed']:
                        finished = True
                        return
                cmd = bounded_body_velocity(command)
                if spot is not None:
                    spot.forward(dt, torch.tensor(cmd, dtype=torch.float32))
                else:
                    if cmd[1] != 0:
                        raise ValueError('Carter cannot execute lateral body velocity; no projection in this bench')
                    drive.apply_command(float(cmd[0]), float(cmd[2]))
                if ticks % decimation == 0:
                    position, orientation = robot.get_world_poses()
                    linear, angular = robot.get_velocities()
                    position, quat = position.numpy()[0], orientation.numpy()[0]
                    rotation, angles = quaternion_state(quat)
                    velocity = rotation.T@linear.numpy()[0]
                    angular_body = rotation.T@angular.numpy()[0]
                    records.append(root_observation(time=now, phase=phase,
                        phase_elapsed=now-(boundaries[index-1] if index else 0.),
                        command_body=cmd.tolist(), position_world=position.tolist(),
                        roll_pitch_yaw=angles, velocity_body=[*velocity[:2], angular_body[2]]))
                    if (not np.isfinite(position).all() or position[2] < (.02 if drive is not None else .25)
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
            now = float(SimulationManager.get_simulation_time())
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
            own_standing = summarize_standing([r for r in records if r['phase'] == 'standing'], **height_bounds)
        tracking = summarize_tracking(records, axes=axes, **height_bounds) if args.stage == 'tracking' and own_standing and own_standing['passed'] and not failures else None
        report = dict(stage=args.stage, passed=bool(own_standing and own_standing['passed'] and not failures
                and (args.stage == 'standing' or tracking and tracking['passed'])),
            settle_seconds=args.settle_seconds, robot=args.robot,
            initial_base_height_m=initial_height, ground_friction_combine='multiply',
            initialization=initialization,
            standing=own_standing, tracking=tracking, failures=failures,
            asset_sha256={k: v['sha256'] for k, v in assets.items()},
            source_files_sha256=source_hashes(), policy_class=type(spot if spot is not None else drive).__name__,
            physics_engine=SimulationManager.get_active_physics_engine(), physics_dt=physics_dt, policy_decimation=decimation,
            asset_root_url=root_url, used_usd_layers=[layer.identifier for layer in stage.GetUsedLayers()],
            root_state_clock='SimulationManager.get_simulation_time at physics callback',
            evaluated_axes=list(axes), unsupported_axes=['lateral', 'combined'] if drive is not None else [],
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

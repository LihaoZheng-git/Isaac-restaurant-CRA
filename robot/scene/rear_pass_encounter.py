"""Separate authored lower-aisle encounter; no physical evidence or safe-space input.

The longer pedestrian leg supplies an observable walking window at unchanged
speed. Robot waypoints are ordinary obstacle-unaware service-route context.
Original restaurant layers are never modified by selecting this session setup.
"""

import math

import numpy as np


OPEN_REAR_LAYOUT = "lower-aisle"
OPEN_ROBOT_START_WORLD = (27.0, 5.1, 0.07382)
OPEN_ROBOT_YAW = math.pi / 2
OPEN_ROBOT_ROUTE_WORLD = ((27.0, 6.2), (27.0, 7.1), (27.0, 7.9))
OPEN_ACTOR_START_WORLD = (25.8, 7.0, 0.0)
OPEN_ACTOR_ROUTE_WORLD = ((29.8, 7.0, 0.0), (29.8, 12.5, 0.0))


def validate_crossing_layout(layout, *, direction, extension, crossing_y):
    if layout not in {"legacy", OPEN_REAR_LAYOUT}:
        raise ValueError("unknown authored crossing layout")
    if layout == OPEN_REAR_LAYOUT and (
        direction != "left-to-right" or extension != 0.0 or crossing_y != 8.9
    ):
        raise ValueError("lower-aisle layout requires left-to-right and default legacy selectors")
    return layout == OPEN_REAR_LAYOUT


def open_robot_route_local_xy():
    return tuple(map(tuple, (np.array(OPEN_ROBOT_ROUTE_WORLD) - OPEN_ROBOT_START_WORLD[:2])
                     @ np.array([[0.0, -1.0], [1.0, 0.0]])))


def reset_open_encounter_velocity(controller):
    """Clear relocation residual velocity once, before settling or observation.

    Never call during navigation or measurement. No ongoing pose constraint,
    friction/drive changes or drift-threshold relaxation is introduced.
    """
    controller.stop()
    controller.robot.set_velocities(linear_velocities=[[0.0, 0.0, 0.0]],
                                    angular_velocities=[[0.0, 0.0, 0.0]])
    controller.robot.set_dof_velocities(np.zeros_like(controller.robot.get_dof_positions().numpy()))


def mount_stationary_perception_bench(stage, robot_prim_path, base_prim_path):
    """User-approved stationary-only mount, authored before physics initializes.

    The caller must prohibit navigation. This is a different mounted sensor
    benchmark, never a repair of freely parked drift or a braking experiment.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    with Usd.EditContext(stage, stage.GetSessionLayer()):
        robot = UsdGeom.Xformable(stage.GetPrimAtPath(robot_prim_path))
        if not np.allclose(Gf.Transform(robot.GetLocalTransformation()).GetScale(), (1, 1, 1)):
            raise ValueError("stationary bench requires the unchanged rigid metre robot scale")
        pose = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), 90.0))
        pose.SetTranslateOnly(Gf.Vec3d(*OPEN_ROBOT_START_WORLD))
        robot.MakeMatrixXform().Set(pose)
        base = stage.GetPrimAtPath(base_prim_path)
        if not base.IsValid() or not base.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError("stationary mount needs the actual rigid chassis")
        world = UsdGeom.Xformable(base).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        position, rotation = world.ExtractTranslation(), world.ExtractRotationQuat()
        joint = UsdPhysics.FixedJoint.Define(stage, "/World/CRAStationaryPerceptionBenchMount")
        joint.CreateBody1Rel().SetTargets([base_prim_path])
        joint.CreateLocalPos0Attr(Gf.Vec3f(position))
        joint.CreateLocalRot0Attr(Gf.Quatf(rotation))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1))
    return dict(enabled=True, joint_path=str(joint.GetPath()), body_path=base_prim_path,
        world_position=list(position), world_rotation_wxyz=[rotation.GetReal(), *rotation.GetImaginary()],
        status="oracle", basis="authored_fixed_mount_for_stationary_sensor_benchmark_only",
        scope="no navigation/braking acceptance; no actor/factor/geometry evidence; original robot files unchanged")

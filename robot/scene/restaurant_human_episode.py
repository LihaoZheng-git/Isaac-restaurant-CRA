"""One native human crossing; actor truth is rendering/evaluation only.

Import after SimulationApp. The caller alone sends copied sensors to CRA.
"""

import math

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdSkel

from robot.evaluation.restaurant_service_validation import authored_raster_clearance
from robot.safety.safe_sage_configuration_space import RectangularRobotFootprint
from robot.scene.restaurant_human_motion import prepare_native_restaurant_human


class RestaurantHumanEpisode:
    def __init__(self, stage, *, mode, output):
        if mode not in {"standing", "crossing"}:
            raise ValueError("unknown human episode")
        self.stage, self.mode, self.output = stage, mode, output
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            self.setup = prepare_native_restaurant_human(stage)
        self.goal = (27.8, 8.9, 0.0)
        self.speed = 0.7
        self.phase = "idle_before"
        self.task = None
        self.task_succeeded = False
        self.records = []
        self.writer = None
        self.last_frame = -math.inf
        self.frame_count = 0

    def bake(self):
        import omni.anim.navigation.core as nav

        if not nav.acquire_interface().start_navmesh_baking_and_wait():
            raise RuntimeError("native restaurant crossing NavMesh bake failed")

    def initialize(self):
        import omni.anim.behavior.core as bh
        from isaacsim.core.experimental.prims import RigidPrim

        self.agent = bh.acquire_interface().get_agent(self.setup["rig_path"])
        if self.agent is None:
            raise RuntimeError("native human behavior unavailable")
        self.agent.set_auto_avoidance_enabled(False)
        self.agent.idle()
        self.body = RigidPrim(self.setup["collider_path"])
        if not self.body.is_physics_tensor_entity_valid():
            raise RuntimeError("human collider has no physics readback")
        skeleton = next(
            p
            for p in Usd.PrimRange(self.stage.GetPrimAtPath(self.setup["rig_path"]))
            if p.IsA(UsdSkel.Skeleton)
        )
        names = UsdSkel.Skeleton(skeleton).GetJointsAttr().Get()
        self.feet = {
            side: next(
                i for i, n in enumerate(names) if n.rsplit("/", 1)[-1] == side + "_Foot"
            )
            for side in ("L", "R")
        }

    def start_video(self):
        import cv2
        from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera

        with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            path = "/World/CRAHumanCrossingCamera"
            camera = RtxCamera.create(path, tick_rate=10)
            prim = UsdGeom.Camera.Get(self.stage, path)
            prim.MakeMatrixXform().Set(
                Gf.Matrix4d()
                .SetLookAt(
                    Gf.Vec3d(25.9, 6.7, 3.2),
                    Gf.Vec3d(29.0, 8.2, 0.7),
                    Gf.Vec3d(0, 0, 1),
                )
                .GetInverse()
            )
            prim.CreateFocalLengthAttr(18)
            prim.CreateHorizontalApertureAttr(36)
            prim.CreateVerticalApertureAttr(27)
        self.camera = CameraSensor(camera, resolution=(600, 800), annotators=["rgb"])
        self.writer = cv2.VideoWriter(
            str(self.output / "human-navigation.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            10,
            (800, 600),
        )
        if not self.writer.isOpened():
            raise RuntimeError("human navigation video unavailable")

    def position(self):
        return np.array(list(self.agent.get_world_translation()), dtype=float)

    def update(self, now, robot_xy, command, layout, *, navigation=False):
        import carb
        import cv2
        import omni.anim.behavior.core as bh
        from isaacsim.core.experimental.utils.backend import use_backend

        # This controls the scripted actor only. No actor state enters CRA.
        if (
            navigation
            and self.mode == "crossing"
            and self.task is None
            and robot_xy[1] >= 7.8
        ):
            self.agent.set_speed(self.speed)
            self.task = self.agent.move_to(self.goal, auto_brake=True)
            if self.task == bh.BEHAVIOR_TASK_ID_INVALID:
                raise RuntimeError("native crossing task rejected")
            self.phase = "move"
        if self.task is not None and not self.task_succeeded:
            status = self.agent.get_task_status(self.task)
            if status == bh.BehaviorTaskStatus.SUCCEEDED:
                self.task_succeeded = True
                self.agent.idle()
                self.phase = "stopped_after"
            elif status == bh.BehaviorTaskStatus.FAILED:
                raise RuntimeError("native crossing task failed")
        xyz = self.position()
        feet = {}
        for side, i in self.feet.items():
            p, q = carb.Float3(), carb.Float4()
            if not self.agent.get_joint_world_transform(i, p, q):
                raise RuntimeError("native human foot transform unavailable")
            feet[side] = list(p)
        with use_backend("tensor", raise_on_fallback=True):
            collider = self.body.get_world_poses()[0].numpy()[0]
        actor_clearance = authored_raster_clearance(
            layout[0],
            (*xyz[:2], 0.0),
            RectangularRobotFootprint(0.4, 0.4, 0.4, 0.4, 0.0),
            layout[1],
        )
        row = {
            "time": now,
            "phase": self.phase,
            "actor_xyz": xyz.tolist(),
            "left_foot_xyz": feet["L"],
            "right_foot_xyz": feet["R"],
            "robot_xy": list(robot_xy),
            "robot_command": list(command),
            "actor_layout_clearance_m": actor_clearance,
            "collider_alignment_error_m": float(
                np.linalg.norm(collider - (xyz + [0, 0, 0.85]))
            ),
        }
        self.records.append(row)
        if (
            actor_clearance <= 0.02
            or row["collider_alignment_error_m"] > 0.03
            or abs(xyz[2]) > 0.05
        ):
            raise RuntimeError("independent human geometry guard failed")
        if self.writer is not None and now - self.last_frame >= 0.1 - 1e-6:
            rgb, _ = self.camera.get_data("rgb")
            rgb = np.array([]) if rgb is None else rgb.numpy()
            if rgb.shape[:2] == (600, 800):
                self.writer.write(np.ascontiguousarray(rgb[:, :, :3][:, :, ::-1]))
                self.frame_count += 1
                if self.frame_count % 20 == 0:
                    cv2.imwrite(
                        str(self.output / "crossing-current.png"),
                        rgb[:, :, :3][:, :, ::-1],
                    )
            self.last_frame = now
        return row

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.camera.detach_annotators("rgb")
            self.writer = None

    def abort(self):
        if hasattr(self, "agent") and self.task is not None and not self.task_succeeded:
            self.agent.idle()
            self.phase = "aborted"

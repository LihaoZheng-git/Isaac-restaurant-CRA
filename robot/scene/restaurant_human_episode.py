"""One native human crossing; actor truth is rendering/evaluation only.

Import after SimulationApp. The caller alone sends copied sensors to CRA.
"""

import math

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdSkel

from robot.evaluation.restaurant_service_validation import authored_raster_clearance
from robot.evaluation.restaurant_baselines import crossing_due
from robot.safety.safe_sage_configuration_space import RectangularRobotFootprint
from robot.scene.restaurant_human_motion import prepare_native_restaurant_human


class RestaurantHumanEpisode:
    def __init__(
        self,
        stage,
        *,
        mode,
        output,
        trigger_robot_y=7.8,
        crossing_delay_seconds=None,
        crossing_direction="right-to-left",
        speed_mps=0.7,
        carry_bowl=False,
        carry_facing_crossing=False,
        upper_continuation=False,
        continuing_away=False,
        crossing_extension_m=0.,
        crossing_y_m=8.9,
        crossing_layout="legacy",
        long_service_parallel=False,
        background_service_walk=False,
    ):
        if background_service_walk and not long_service_parallel:
            raise ValueError("background service walk requires the separate long parallel demo")
        from robot.scene.restaurant_background_walk import BackgroundServiceWalk
        self.background_walk = BackgroundServiceWalk() if background_service_walk else None
        self.background_walk_requested = False
        if long_service_parallel and (mode != "parallel" or crossing_delay_seconds is not None
            or not 8.8 <= trigger_robot_y <= 9.2):
            raise ValueError("long-service actor requires parallel mode and explicit y=8.8..9.2 trigger")
        if mode not in {"standing", "crossing", "continuous-crossing", "parallel"}:
            raise ValueError("unknown human episode")
        if crossing_direction not in {"right-to-left", "left-to-right"} or (
            crossing_direction == "left-to-right" and mode != "crossing"
        ):
            raise ValueError("mirrored crossing is bounded to standalone walking")
        self.crossing_direction = crossing_direction
        self.parallel_oncoming = mode == 'parallel'
        self.long_service_parallel = long_service_parallel
        self.initial_idle_held = False
        from robot.scene.rear_pass_encounter import validate_crossing_layout
        self.open_layout = validate_crossing_layout(crossing_layout, direction=crossing_direction,
            extension=crossing_extension_m, crossing_y=crossing_y_m)
        if self.open_layout and not continuing_away:
            raise ValueError("lower-aisle encounter requires continuing-away mode")
        if continuing_away and (mode != "crossing" or carry_bowl or speed_mps != .5):
            raise ValueError("departure requires bare standalone crossing at .5 m/s")
        from robot.scene.restaurant_departing_pedestrian import DepartingPedestrianRoute
        if (crossing_extension_m or crossing_y_m != 8.9) and not continuing_away:
            raise ValueError("authored extension requires the explicit continuing-away fixture")
        self.departure = DepartingPedestrianRoute(crossing_direction=crossing_direction,
            crossing_extension_m=crossing_extension_m,crossing_y_m=crossing_y_m,
            crossing_layout=crossing_layout) if continuing_away else None
        if carry_bowl and mode != "continuous-crossing":
            raise ValueError("carried server requires explicit continuous crossing")
        if upper_continuation and mode != "continuous-crossing":
            raise ValueError("upper continuation requires continuous crossing")
        if carry_facing_crossing and not carry_bowl:
            raise ValueError("initial carrying heading requires a carried bowl")
        self.carry_facing_crossing = carry_facing_crossing
        if mode == "continuous-crossing" and speed_mps != 0.5:
            raise ValueError("bounded continuous circuit requires .5 m/s")
        maximum_trigger_y = 8.4 if mode == "crossing" and not carry_bowl else 7.8
        minimum_trigger_y = 6.0 if self.open_layout else 6.8
        maximum_trigger_y = 6.6 if self.open_layout else maximum_trigger_y
        if long_service_parallel:
            minimum_trigger_y, maximum_trigger_y = 8.8, 9.2
        if not math.isfinite(trigger_robot_y) or not minimum_trigger_y <= trigger_robot_y <= maximum_trigger_y:
            raise ValueError(f"bounded native crossing trigger must be y=6.8..{maximum_trigger_y}")
        self.trigger_robot_y = trigger_robot_y
        if crossing_delay_seconds is not None and (
            mode == "standing" or not math.isfinite(crossing_delay_seconds)
            or not 0 <= crossing_delay_seconds <= 30
        ):
            raise ValueError("exogenous crossing delay requires crossing and 0..30 seconds")
        self.crossing_delay_seconds = crossing_delay_seconds
        self.navigation_start_time = None
        if not math.isfinite(speed_mps) or not 0.5 <= speed_mps <= 1.0:
            raise ValueError("bounded coupled native walking speed must be .5..1 m/s")
        if self.parallel_oncoming and (speed_mps != .5 or continuing_away or carry_bowl
            or upper_continuation or (crossing_delay_seconds is None and not long_service_parallel)):
            raise ValueError('parallel fixture requires explicit timing, bare adult and .5m/s')
        self.stage, self.mode, self.output = stage, mode, output
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            self.setup = prepare_native_restaurant_human(stage, crossing_direction=crossing_direction,
                crossing_extension_m=crossing_extension_m,crossing_y_m=crossing_y_m,
                crossing_layout=crossing_layout, parallel_oncoming=self.parallel_oncoming)
        self.goal = (29.8 if crossing_direction == "left-to-right" else 27.8, 8.9, 0.0)
        if self.departure is not None:
            self.goal = self.departure.goal
        if self.parallel_oncoming:
            from robot.scene.parallel_walk_encounter import PARALLEL_ACTOR_GOAL
            self.goal = PARALLEL_ACTOR_GOAL
        self.speed = speed_mps
        self.phase = "idle_before"
        self.task = None
        self.task_succeeded = False
        self.records = []
        self.writer = None
        self.last_frame = -math.inf
        self.frame_count = 0
        from robot.scene.restaurant_continuous_server import ContinuousServerRoute

        self.circuit = (
            ContinuousServerRoute(upper_continuation=upper_continuation)
            if mode == "continuous-crossing"
            else None
        )
        self.carry = None
        self.carry_setup = None
        if carry_bowl:
            from robot.scene.restaurant_human_carry import author_native_carry_bowl

            self.carry_setup = author_native_carry_bowl(stage)

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
        if self.parallel_oncoming:
            import carb
            self.agent.idle(facing=carb.Float3(0, -1, 0))
        elif self.crossing_direction == "left-to-right":
            import carb
            self.agent.idle(facing=carb.Float3(1, 0, 0))
        elif self.departure is not None:
            import carb
            self.agent.idle(facing=carb.Float3(-1, 0, 0))
        else:
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
        if self.carry_setup is not None:
            from robot.scene.restaurant_human_carry import NativeBowlCarry

            self.carry = NativeBowlCarry(self.agent, self.carry_setup, names)

    def hold_initial_idle(self):
        """Stop native idle root drift; release before walking, never teleport."""
        if not self.long_service_parallel or self.phase != 'idle_before' or self.task is not None:
            raise RuntimeError('initial idle hold requires the settled long-service actor')
        self.agent.set_enabled(False)
        if self.agent.is_enabled():
            raise RuntimeError('native agent did not enter the initial hold')
        self.initial_idle_held = True

    def begin_carry(self):
        """Call only after the native idle/robot startup has settled."""
        if (
            self.carry is None
            or self.carry.task is not None
            or self.phase != "idle_before"
        ):
            raise RuntimeError(
                "native carry start requires a settled idle and no previous pickup"
            )
        self.carry.start()
        self.phase = "pickup"

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
                    Gf.Vec3d(25.9, 6.7, 3.2) if self.parallel_oncoming else Gf.Vec3d(26.5, 7.7, 2.4)
                    if self.carry_setup is not None
                    else Gf.Vec3d(24.5, 5.5, 3.2) if self.open_layout else Gf.Vec3d(25.9, 6.7, 3.2),
                    Gf.Vec3d(29., 8.7, .8) if self.parallel_oncoming else Gf.Vec3d(29.0, 9.5, 0.8)
                    if self.carry_setup is not None
                    else Gf.Vec3d(28.0, 7.0, 0.8) if self.open_layout else Gf.Vec3d(29.0, 8.2, 0.7),
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
            (navigation or self.background_walk_requested)
            and self.mode in {"crossing", "continuous-crossing", "parallel"}
            and self.task is None
            and self.phase == "idle_before"
            and (self.background_walk is not None or crossing_due(now=now, robot_y=robot_xy[1], trigger_y=self.trigger_robot_y,
                             delay_seconds=self.crossing_delay_seconds,
                             navigation_start=self.navigation_start_time))
        ):
            if self.initial_idle_held:
                self.agent.set_enabled(True)
                if not self.agent.is_enabled():
                    raise RuntimeError('native agent did not release the initial hold')
                self.initial_idle_held = False
            self.agent.set_speed(self.speed)
            self.task = self.agent.move_to(
                self.background_walk.goal if self.background_walk else self.circuit.goal if self.circuit else self.goal,
                auto_brake=self.circuit is None and self.departure is None and self.background_walk is None,
            )
            if self.task == bh.BEHAVIOR_TASK_ID_INVALID:
                raise RuntimeError("native crossing task rejected")
            self.phase = self.background_walk.phase if self.background_walk else "move"
        if (
            self.task is not None
            and not self.task_succeeded
            and self.phase != "aborted"
        ):
            status = self.agent.get_task_status(self.task)
            if status == bh.BehaviorTaskStatus.FAILED:
                raise RuntimeError("native crossing task failed")
            if self.background_walk is not None:
                from robot.scene.restaurant_background_walk import BACKGROUND_ENCOUNTER_TRIGGER_Y
                if self.background_walk.advance(self.position(), now,
                        encounter_due=robot_xy[1] >= BACKGROUND_ENCOUNTER_TRIGGER_Y):
                    self.task = self.agent.move_to(self.background_walk.goal, auto_brake=False)
                    if self.task == bh.BEHAVIOR_TASK_ID_INVALID:
                        raise RuntimeError("native background walk next-leg task rejected")
                    self.phase = self.background_walk.phase
                elif status == bh.BehaviorTaskStatus.SUCCEEDED:
                    raise RuntimeError("native background walk success without actual progress")
            elif self.departure is not None and self.departure.advance(self.position(), now):
                self.task = self.agent.move_to(self.departure.goal, auto_brake=True)
                if self.task == bh.BEHAVIOR_TASK_ID_INVALID:
                    raise RuntimeError("native departure task rejected")
                self.phase = "walk_away"
            elif self.circuit is not None:
                if self.circuit.advance(self.position(), now):
                    self.task = self.agent.move_to(self.circuit.goal, auto_brake=False)
                    if self.task == bh.BEHAVIOR_TASK_ID_INVALID:
                        raise RuntimeError("continuous crossing next-leg task rejected")
                elif status == bh.BehaviorTaskStatus.SUCCEEDED:
                    raise RuntimeError(
                        "native task success without actual crossing progress"
                    )
            elif status == bh.BehaviorTaskStatus.SUCCEEDED:
                self.task_succeeded = True
                self.agent.idle()
                self.phase = "away_endpoint" if self.departure is not None or self.parallel_oncoming else "stopped_after"
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
            "initial_idle_held": self.initial_idle_held,
            "native_agent_enabled": bool(self.agent.is_enabled()),
            "background_service_walk": self.background_walk is not None,
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
        if self.carry is not None:
            self.carry.tick(now, self.phase, layout)
            if self.phase == "pickup" and self.carry.ready:
                self.agent.idle(
                    facing=carb.Float3(-1, 0, 0)
                    if self.carry_facing_crossing
                    else self.agent.get_facing_direction()
                )
                self.phase = "idle_before"
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
        if (
            hasattr(self, "agent")
            and self.task is not None
            and not self.task_succeeded
            and self.phase != "aborted"
        ):
            self.agent.idle()
            self.phase = "aborted"

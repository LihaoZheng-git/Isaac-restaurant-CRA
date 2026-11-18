"""Native window-table diner for the live left delivery."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from robot.scene.restaurant_human_motion import (
    HUMAN_MOTION_LIBRARY_SHA256,
    HUMAN_MOTION_LIBRARY_URL,
    attach_moving_human_collider,
    configure_human_navmesh,
)
from robot.scene.restaurant_seated_cast import _LIBRARY, _archived_diners, _author_diner, _seat, _verified_remote
from robot.scene.restaurant_server import SERVER_PRIM_PATH
from robot.scene.restaurant_straight_walk import (
    LEFT_OUTBOUND_LOUNGE_TRIGGER_Y,
    LEFT_OUTBOUND_WEST_DINER_GUIDES,
    LEFT_OUTBOUND_WEST_DINER_START,
    StraightServiceWalk,
    audit_straight_walk,
)


WINDOW_DINER_NAME = "LeftOppositeChair"
WINDOW_DINER_NAVMESH_PATH = "/World/RestaurantWindowDinerNavMesh"
# Ease around the chair through the clear space south of the original turn.
TABLE_EXIT = (
    (1.60, 18.71, 0.), (1.40, 18.52, 0.), (.95, 18.48, 0.),
    (.75, 18.68, 0.), (.72, 16.20, 0.),
)
LOUNGE_APPROACH = ((.95, 6., 0.), (.95, 4.15, 0.), (1.41, 4.03, 0.))
LOUNGE_SEAT = (1.41, 2.97, .464)
LOUNGE_FACING = (0., 1., 0.)
TABLE_RELEASE_Y = 5.75
TABLE_RELEASE_MAX_X = 4.0
_CHAIR_TURN_TOLERANCE_M = .10
_FAILED_CHAIR_TURN_TOLERANCE_M = .15
_TABLE_EXIT_FIRST_TURN_TOLERANCE_M = .27
_SHARED_NAVMESH_FAILED_CHAIR_TURN_TOLERANCES_M = (.28, .17)
WALK_SPEED = .5
_COLLIDER_RADIUS = .4
_COLLIDER_HEIGHT = 1.7


class RestaurantWindowDinerEpisode:
    """Act out the rehearsed departure and preserve the tested aisle segment."""

    carry = carry_setup = background_walk = circuit = departure = None

    def __init__(self, stage, output, *, shared_navmesh=False):
        from pxr import BehaviorSchema, NavSchema

        self.stage = stage
        self.output = Path(output)
        self._failed_chair_turn_tolerances = (
            _SHARED_NAVMESH_FAILED_CHAIR_TURN_TOLERANCES_M if shared_navmesh
            else (_TABLE_EXIT_FIRST_TURN_TOLERANCE_M, _FAILED_CHAIR_TURN_TOLERANCE_M)
        )
        self.phase = "table_seating"
        self.task_succeeded = False
        self.records = []
        self.events = []
        self.aisle_records = []
        self.straight_walk = StraightServiceWalk(left_outbound_service=True, left_outbound_lounge_walk=True,
                                                left_outbound_west_diner=True)
        self._collider_active = False
        self._phase_started = None
        self._mark_index = 0
        self._steady_frames = 0
        self._previous_joints = None
        self._table_hip_z = None
        self._held_joints = None
        (diner, entry), = _archived_diners((WINDOW_DINER_NAME,))
        if hashlib.sha256(Path(entry["asset_path"]).read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("The rehearsed west-window diner asset changed")
        if not stage.GetPrimAtPath(_LIBRARY):
            _verified_remote(HUMAN_MOTION_LIBRARY_URL, HUMAN_MOTION_LIBRARY_SHA256)
        root_layer = stage.GetRootLayer()
        prior_metadata = dict(root_layer.customLayerData)
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            base_actor = stage.GetPrimAtPath(SERVER_PRIM_PATH)
            if not base_actor or not base_actor.IsActive():
                raise RuntimeError("The original restaurant actor slot is missing")
            UsdGeom.Imageable(base_actor).MakeInvisible()
            old_collider = stage.GetPrimAtPath(SERVER_PRIM_PATH + "/StaticCollision")
            if old_collider and not old_collider.SetActive(False):
                raise RuntimeError("Could not remove the superseded standing collider")
            if not stage.GetPrimAtPath(_LIBRARY):
                stage.DefinePrim(_LIBRARY).GetReferences().AddReference(HUMAN_MOTION_LIBRARY_URL)
            actor = _author_diner(stage, diner, entry, BehaviorSchema, NavSchema)
            rig = stage.GetPrimAtPath(actor["rig_path"])
            capsule = attach_moving_human_collider(stage, rig, radius=_COLLIDER_RADIUS, height=_COLLIDER_HEIGHT)
            self._collision_enabled = UsdPhysics.CollisionAPI(capsule.GetPrim()).CreateCollisionEnabledAttr(False)
            extent = [(0., .75, -.2), (8., 20.6, 2.5)]
            volume = NavSchema.NavMeshVolume.Define(stage, WINDOW_DINER_NAVMESH_PATH)
            volume.CreateNavVolumeTypeAttr("Include")
            volume.CreateExtentAttr([Gf.Vec3f(*corner) for corner in extent])
            self._lounge_path = _seat(stage, "WalkingDinerLounge", LOUNGE_SEAT, LOUNGE_FACING, BehaviorSchema)
        try:
            self._navmesh_settings = configure_human_navmesh(stage)
        finally:
            root_layer.customLayerData = prior_metadata
        self.setup = dict(**actor, collider_path=str(capsule.GetPath()),
                          navmesh_prim_path=WINDOW_DINER_NAVMESH_PATH,
                          collision_radius_m=_COLLIDER_RADIUS, collision_height_m=_COLLIDER_HEIGHT,
                          table_exit_world_xyz=TABLE_EXIT, table_release_robot_y=TABLE_RELEASE_Y,
                          table_release_robot_max_x=TABLE_RELEASE_MAX_X,
                          aisle_start_world_xyz=LEFT_OUTBOUND_WEST_DINER_START,
                          aisle_guides_world_xyz=LEFT_OUTBOUND_WEST_DINER_GUIDES,
                          actor_trigger_robot_world_y=LEFT_OUTBOUND_LOUNGE_TRIGGER_Y,
                          actor_goal_world_xyz=LEFT_OUTBOUND_WEST_DINER_GUIDES[-1],
                          lounge_approach_world_xyz=LOUNGE_APPROACH, lounge_seat_world_xyz=LOUNGE_SEAT,
                          speed_mps=WALK_SPEED, actor_navmesh_extent_m=extent,
                          failed_chair_turn_tolerances_m=self._failed_chair_turn_tolerances,
                          navmesh_settings_metres=self._navmesh_settings,
                          scope="Native actor choreography and passive readback; no authored pose enters CRA.")

    def bake(self):
        import omni.anim.navigation.core as nav

        root = self.stage.GetRootLayer()
        original = dict(root.customLayerData)
        try:
            root.customLayerData = {**original, "navmeshSettings": self._navmesh_settings}
            if not nav.acquire_interface().start_navmesh_baking_and_wait():
                raise RuntimeError("The window diner NavMesh bake failed")
        finally:
            root.customLayerData = original

    def initialize(self):
        import omni.anim.behavior.core as behavior
        from isaacsim.core.experimental.prims import RigidPrim

        self.agent = behavior.acquire_interface().get_agent(self.setup["rig_path"])
        if self.agent is None:
            raise RuntimeError("The west-window diner's native agent is missing")
        self.agent.set_auto_avoidance_enabled(False)
        self.agent.set_auto_impact_enabled(False)
        self.body = RigidPrim(self.setup["collider_path"])
        if not self.body.is_physics_tensor_entity_valid():
            raise RuntimeError("The west-window diner collider has no physics tensor readback")
        self._set_task(self.agent.sit(self.setup["seat_path"], snap_to_seat=True))

    def start_video(self):
        """The live service recorder already captures the diner and restaurant."""

    def hold_initial_idle(self):
        """The live tick verifies and holds the seated pose before navigation."""

    @property
    def ready_for_navigation(self):
        return self.phase == "table_hold" and not self.agent.is_enabled()

    @property
    def collider_active(self):
        return self._collider_active

    def position(self):
        return np.asarray(tuple(self.agent.get_world_translation()), dtype=float)

    def _collider_position(self):
        from isaacsim.core.experimental.utils.backend import use_backend

        with use_backend("tensor", raise_on_fallback=True):
            return self.body.get_world_poses()[0].numpy()[0]

    def _set_collision(self, enabled):
        with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            if not self._collision_enabled.Set(enabled) or self._collision_enabled.Get() is not enabled:
                raise RuntimeError("Could not set the diner collider for this motion phase")
        self._collider_active = enabled

    def _set_task(self, task):
        import omni.anim.behavior.core as behavior

        if task == behavior.BEHAVIOR_TASK_ID_INVALID:
            raise RuntimeError(f"The window diner task was rejected during {self.phase}")
        self.task = task

    def _joints(self):
        import carb

        result = {}
        for name, index in self.setup["joints"].items():
            point, rotation = carb.Float3(), carb.Float4()
            if not self.agent.get_joint_world_transform(index, point, rotation):
                raise RuntimeError(f"Missing native diner joint: {name}")
            result[name] = tuple(float(point[i]) for i in range(3))
        return result

    def _enter(self, phase, now, **details):
        self.phase = phase
        self._phase_started = now
        self._mark_index = 0
        self._steady_frames = 0
        self._previous_joints = None
        self.events.append(dict(time=now, phase=phase, **details))

    def _begin_walk(self, goal, now, *, brake):
        self.agent.set_speed(WALK_SPEED)
        self._set_task(self.agent.move_to(goal, auto_brake=brake))
        self.events.append(dict(time=now, phase=self.phase, goal=goal, auto_brake=brake))

    def _advance_walk(self, now, xyz, marks, *, run_through=(), proximity=(), chair_exit=False):
        import omni.anim.behavior.core as behavior

        goal = marks[self._mark_index]
        gap = math.dist(xyz, goal)
        status = self.agent.get_task_status(self.task)
        failed = status == behavior.BehaviorTaskStatus.FAILED
        # Native short steps can end early at the first two marks beside the
        # chair. Floor marks and aisle entry still require native success.
        turn_tolerance = _FAILED_CHAIR_TURN_TOLERANCE_M if failed else _CHAIR_TURN_TOLERANCE_M
        if chair_exit and failed and self._mark_index < len(self._failed_chair_turn_tolerances):
            turn_tolerance = self._failed_chair_turn_tolerances[self._mark_index]
        near_first_chair_turn = chair_exit and failed and self._mark_index == 0 and gap <= turn_tolerance
        handoff = (near_first_chair_turn
                   or (self._mark_index in run_through and gap <= .20)
                   or (self._mark_index in proximity and gap <= turn_tolerance))
        if failed and not handoff:
            raise RuntimeError(f"Native window diner could not walk from {xyz} to {goal} (gap {gap:.4f} m)")
        if status != behavior.BehaviorTaskStatus.SUCCEEDED and not handoff:
            return False
        if gap > .27 and not near_first_chair_turn:
            raise RuntimeError(f"Native window diner stopped away from {goal}: {xyz}")
        self._mark_index += 1
        self.events.append(dict(time=now, phase=self.phase, completed_goal=goal,
                                actor_xyz=xyz.tolist(), goal_gap_m=gap, native_status=str(status),
                                proximity_handoff=handoff))
        if self._mark_index == len(marks):
            return True
        self._begin_walk(marks[self._mark_index], now, brake=self.phase != "move" or self._mark_index == 2)
        return False

    def _settled(self, joints, seat, *, hip_tolerance):
        steady = self._previous_joints is not None and all(
            math.dist(joints[name], self._previous_joints[name]) <= .006 for name in joints
        )
        self._previous_joints = joints
        self._steady_frames = self._steady_frames + 1 if steady and math.dist(joints["Hip"], seat) <= hip_tolerance else 0
        return self._steady_frames >= 12

    def _advance(self, now, robot_xy, xyz, joints, *, navigation):
        import carb
        import omni.anim.behavior.core as behavior

        if self.phase in {"table_seating", "lounge_sit"}:
            if self.agent.get_task_status(self.task) == behavior.BehaviorTaskStatus.FAILED:
                raise RuntimeError(f"Native diner seating failed during {self.phase}")
            at_table = self.phase == "table_seating"
            seat = self.setup["world_seat_xyz"] if at_table else LOUNGE_SEAT
            if self._settled(joints, seat, hip_tolerance=.08 if at_table else .07):
                self.agent.set_enabled(False)
                if self.agent.is_enabled():
                    raise RuntimeError("Could not hold the seated diner")
                if at_table:
                    self._table_hip_z = joints["Hip"][2]
                    self._enter("table_hold", now, actor_xyz=xyz.tolist(), seated_hip_xyz=joints["Hip"])
                else:
                    self._held_joints = joints
                    self.held_endpoint_xyz = xyz.copy()
                    self.task_succeeded = True
                    self._enter("away_endpoint", now, actor_xyz=xyz.tolist(), seated_hip_xyz=joints["Hip"])
        elif self.phase == "table_hold" and navigation and robot_xy[0] < TABLE_RELEASE_MAX_X and robot_xy[1] >= TABLE_RELEASE_Y:
            self.agent.set_enabled(True)
            self._set_task(self.agent.idle(facing=carb.Float3(*self.setup["facing"])))
            self._enter("rise", now, robot_xy=list(robot_xy))
        elif self.phase == "rise" and now - self._phase_started >= 1.65:
            if joints["Hip"][2] < self._table_hip_z + .19:
                raise RuntimeError("The west-window diner did not rise from the table")
            self._enter("table_exit", now, standing_hip_xyz=joints["Hip"])
            self._begin_walk(TABLE_EXIT[0], now, brake=True)
        elif self.phase == "table_exit":
            if self._advance_walk(now, xyz, TABLE_EXIT, run_through=(0,), proximity=(1, 2),
                                  chair_exit=True):
                self._set_collision(True)
                self._enter("walk_to_aisle", now)
                self.events.append(dict(time=now, phase=self.phase, collider_enabled=True, actor_xyz=xyz.tolist()))
                self._begin_walk(LEFT_OUTBOUND_WEST_DINER_START, now, brake=True)
        elif self.phase == "walk_to_aisle" and self._advance_walk(now, xyz, (LEFT_OUTBOUND_WEST_DINER_START,)):
            if math.dist(xyz, LEFT_OUTBOUND_WEST_DINER_START) > .15:
                raise RuntimeError("The window diner did not reach the tested aisle start")
            self.agent.set_enabled(False)
            if self.agent.is_enabled():
                raise RuntimeError("Could not hold the diner at the aisle start")
            self._enter("aisle_ready", now, actor_xyz=xyz.tolist(), robot_xy=list(robot_xy))
        elif self.phase == "aisle_ready" and navigation and robot_xy[1] >= LEFT_OUTBOUND_LOUNGE_TRIGGER_Y:
            self.agent.set_enabled(True)
            self._enter("move", now, actor_xyz=xyz.tolist(), robot_xy=list(robot_xy))
            self._begin_walk(LEFT_OUTBOUND_WEST_DINER_GUIDES[0], now, brake=True)
        elif self.phase == "move" and self._advance_walk(now, xyz, LEFT_OUTBOUND_WEST_DINER_GUIDES, run_through=(0, 1)):
            self._enter("walk_to_lounge", now, aisle_endpoint_xyz=xyz.tolist())
            self._begin_walk(LOUNGE_APPROACH[0], now, brake=True)
        elif self.phase == "walk_to_lounge" and self._advance_walk(now, xyz, LOUNGE_APPROACH):
            self._set_collision(False)
            self._enter("lounge_sit", now, actor_xyz=xyz.tolist(), collider_enabled=False)
            self._set_task(self.agent.sit(self._lounge_path))

    def update(self, now, robot_xy, command, layout, *, navigation=False):
        from robot.evaluation.restaurant_service_validation import authored_raster_clearance
        from robot.safety.safe_sage_configuration_space import RectangularRobotFootprint

        xyz = self.position()
        joints = self._joints()
        active_during_sample = self._collider_active
        collider = self._collider_position()
        alignment = float(np.linalg.norm(collider - (xyz + [0., 0., _COLLIDER_HEIGHT / 2.])))
        if active_during_sample and alignment > .03:
            raise RuntimeError(f"The walking diner and its active capsule are misaligned by {alignment:.3f} m")
        previous_phase = self.phase
        clearance = authored_raster_clearance(
            layout[0], (*xyz[:2], 0.), RectangularRobotFootprint(.4, .4, .4, .4, 0.), layout[1]
        )
        self._advance(now, robot_xy, xyz, joints, navigation=navigation)
        if (active_during_sample or self._collider_active) and clearance <= .02:
            raise RuntimeError(f"The active diner capsule approaches restaurant furniture by {clearance:.3f} m")
        row = dict(time=float(now), phase=self.phase, actor_xyz=xyz.tolist(), robot_xy=list(robot_xy),
                   robot_command=list(command), left_foot_xyz=joints["L_Foot"], right_foot_xyz=joints["R_Foot"],
                   collider_xyz=collider.tolist(), collider_active=active_during_sample,
                   native_agent_enabled=bool(self.agent.is_enabled()), actor_layout_clearance_m=clearance,
                   collider_alignment_error_m=alignment, straight_service_walk=True, left_outbound_service=True,
                   left_outbound_lounge_walk=True, left_outbound_west_diner=True)
        self.records.append(row)
        if self.phase == "move":
            self.aisle_records.append(row)
        elif previous_phase == "move" and self.phase == "walk_to_lounge":
            self.aisle_records.append(dict(row, phase="away_endpoint", actual_phase=self.phase))
        return row

    def validation(self):
        phases = {row["phase"] for row in self.events}
        active = [row for row in self.records if row["collider_active"]]
        aisle = (audit_straight_walk(self.aisle_records, left_outbound_service=True,
                                    left_outbound_lounge_walk=True, left_outbound_west_diner=True)
                 if len(self.aisle_records) > 1 else {"passed": False})
        hold = self.departed_pose_readback() if self.task_succeeded else {"passed": False}
        checks = dict(window_table_seated="table_hold" in phases,
                      stood_from_window_table="table_exit" in phases,
                      joined_tested_aisle="aisle_ready" in phases,
                      completed_original_native_aisle_walk=aisle["passed"],
                      walking_capsule_aligned=bool(active) and all(
                          row["collider_alignment_error_m"] <= .03 and row["actor_layout_clearance_m"] > .02
                          for row in active),
                      lounge_seated_and_held=self.phase == "away_endpoint" and hold["passed"])
        return dict(passed=all(checks.values()), checks=checks, aisle_walk=aisle, lounge_hold=hold)

    def departed_pose_readback(self):
        if self._held_joints is None:
            raise RuntimeError("The window diner has not reached the lounge seat")
        xyz = self.position()
        collider = self._collider_position()
        joints = self._joints()
        drift = max(math.dist(joints[name], self._held_joints[name]) for name in joints)
        return dict(actor_xyz=xyz.tolist(), collider_xyz=collider.tolist(), drift_m=drift,
                    collider_active=self._collider_active,
                    passed=not self.agent.is_enabled() and not self._collider_active and drift <= .02)

    def hold_departed_pose(self):
        return self.departed_pose_readback()

    def frozen_table_gesture_binding(self):
        """Expose the seated pose while the original agent is waiting to rise."""
        if not self.ready_for_navigation or self._collider_active:
            raise RuntimeError("Table gestures require the original settled, collision-disabled diner")
        return self.setup, self.agent, self._joints()

    def frozen_lounge_gesture_binding(self):
        """Expose the fixed lounge pose for animation of a separate visual."""
        if self.phase != "away_endpoint" or not self.departed_pose_readback()["passed"]:
            raise RuntimeError("Lounge gestures require the original settled, collision-disabled diner")
        return dict(self.setup, facing=LOUNGE_FACING), self.agent, self._held_joints

    def abort(self):
        if hasattr(self, "agent") and self.agent.is_enabled():
            self.agent.set_enabled(False)

    def close(self):
        path = self.output / "window-diner-motion.json"
        path.write_text(json.dumps(dict(setup=self.setup, events=self.events, records=self.records,
                                       aisle_records=self.aisle_records, validation=self.validation()),
                                   indent=2) + "\n", encoding="utf-8")

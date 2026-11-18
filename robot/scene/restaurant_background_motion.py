"""Optional seated gestures and planted counter work for the live restaurant."""

import math
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdSkel, Vt

from robot.scene.restaurant_live_bar_handoff import BarHandoff, _rotation_at, _wrist_and_elbow
from robot.scene.restaurant_live_physical_handoff import _bar_staff
from robot.scene.restaurant_newspaper_reader import NewspaperReaderMotion
from robot.scene.restaurant_seated_cast import _HOLD_TOLERANCE_M


COUNTER_WORK_PATH = "/World/RestaurantLiveV1/StaticActors/CounterWorker"
COUNTER_WORK_POSITION = (23.25, 6.82, 0.)
_COUNTER_REACH = (23.25, 6.34, 1.03)
_UP = Gf.Vec3d(0., 0., 1.)
_PLANTED_JOINTS = ("Hip", "L_Foot", "R_Foot")
_STARTUP_GESTURE_DELAYS_S = {"FamilyFather": .2, "FamilyMother": 2.1}
_POSE_INTERVAL_S = 1. / 20.


class _SeatedGesture:
    """Drive a separate skinned visual from the exact frozen native pose."""

    def __init__(self, stage, spec, agent, anchors, seat_index, *, overlay_name="GestureVisual"):
        import carb

        self.name = spec["name"]
        self.anchors = anchors
        self.drift_m = self.head_m = self.hand_m = 0.
        self.period = 5.8 + .43 * (seat_index % 5)
        self.stagger = _STARTUP_GESTURE_DELAYS_S.get(self.name, .36 * (seat_index % 11))
        self.reach_scale = .7 + .10 * (seat_index % 4)
        session = stage.GetSessionLayer()
        actor = Sdf.Path(spec["root_path"])
        self._original_path = actor.AppendPath("Units/Visual")
        self._overlay_path = actor.AppendPath("Units").AppendChild(overlay_name)
        source_rig = Sdf.Path(spec["rig_path"])
        # Prepare the copy outside the composed stage. Isaac can register
        # Behavior and PhysX prims the instant an unfiltered copy is published.
        staging = Sdf.Layer.CreateAnonymous()
        Sdf.CreatePrimInLayer(staging, self._overlay_path.GetParentPath())
        if not Sdf.CopySpec(session, self._original_path, staging, self._overlay_path):
            raise RuntimeError(f"Cannot clone the seated visual for {self.name}")
        copied_rig = staging.GetPrimAtPath(source_rig.ReplacePrefix(self._original_path, self._overlay_path))
        schemas = copied_rig.GetInfo("apiSchemas")
        if schemas.isExplicit:
            schemas.explicitItems = [item for item in schemas.explicitItems if item != "BehaviorAgentAPI"]
        else:
            for field in ("addedItems", "appendedItems", "prependedItems", "orderedItems"):
                setattr(schemas, field, [item for item in getattr(schemas, field) if item != "BehaviorAgentAPI"])
            schemas.deletedItems = list(set(schemas.deletedItems) | {"BehaviorAgentAPI"})
        copied_rig.SetInfo("apiSchemas", schemas)
        if spec.get("collider_path"):
            collider_path = Sdf.Path(spec["collider_path"]).ReplacePrefix(self._original_path, self._overlay_path)
            removal = Sdf.BatchNamespaceEdit()
            removal.Add(collider_path, Sdf.Path.emptyPath)
            if not staging.Apply(removal):
                raise RuntimeError(f"Could not exclude the copied diner collider for {self.name}")
        if not Sdf.CopySpec(staging, self._overlay_path, session, self._overlay_path):
            raise RuntimeError(f"Cannot publish the seated visual for {self.name}")
        visual = stage.GetPrimAtPath(self._overlay_path)
        self.rig, = (UsdSkel.Root(prim) for prim in Usd.PrimRange(visual) if prim.IsA(UsdSkel.Root))
        self.skeleton, = (UsdSkel.Skeleton(prim) for prim in Usd.PrimRange(visual)
                          if prim.IsA(UsdSkel.Skeleton))
        if "BehaviorAgentAPI" in self.rig.GetPrim().GetAppliedSchemas():
            raise RuntimeError(f"The presentation visual for {self.name} has a behavior agent")
        joints = self.skeleton.GetJointsAttr().Get()
        self.topology = UsdSkel.Topology(joints)
        self.parents = self.topology.GetParentIndices()
        self.indices = {joint.rsplit("/", 1)[-1]: i for i, joint in enumerate(joints)}
        self.world_from_skel = UsdGeom.Xformable(self.skeleton).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        to_skel = self.world_from_skel.GetInverse()
        self.up = to_skel.TransformDir(_UP).GetNormalized()
        self.forward = to_skel.TransformDir(Gf.Vec3d(*spec["facing"])).GetNormalized()
        self.right = Gf.Cross(self.forward, self.up).GetNormalized()
        self.units_per_metre = to_skel.TransformDir(_UP).GetLength()
        native_world = []
        for index in range(len(joints)):
            xyz, quat = carb.Float3(), carb.Float4()
            if not agent.get_joint_world_transform(index, xyz, quat):
                raise RuntimeError(f"Missing captured seated joint {joints[index]} for {self.name}")
            orientation = Gf.Quatd(float(quat[3]), Gf.Vec3d(*(float(quat[i]) for i in range(3))))
            matrix = Gf.Matrix4d().SetRotate(orientation.GetNormalized())
            matrix.SetTranslateOnly(Gf.Vec3d(*(float(xyz[i]) for i in range(3))))
            native_world.append(matrix * to_skel)
        # A source rig can be in centimetres, while the captured positions use
        # metres. Normalize only the rotation bases to keep skinning scale 1.
        self.base_world = Vt.Matrix4dArray([m.GetOrthonormalized() for m in native_world])
        self.base_local = Vt.Matrix4dArray([
            matrix if parent < 0 else matrix * self.base_world[parent].GetInverse()
            for matrix, parent in zip(self.base_world, self.parents, strict=True)
        ])
        self.motion = UsdSkel.Animation.Define(stage, self._overlay_path.AppendChild("SeatedGesture"))
        self.motion.CreateJointsAttr(joints)
        translates, rotations, scales = UsdSkel.DecomposeTransforms(self.base_local)
        self.motion.CreateTranslationsAttr(translates)
        self.motion.CreateRotationsAttr(rotations)
        self.motion.CreateScalesAttr(scales)
        UsdSkel.BindingAPI.Apply(self.skeleton.GetPrim()).CreateAnimationSourceRel().SetTargets(
            [self.motion.GetPath()]
        )
        self.diagnostics = spec["joints"]
        base_readback = self.positions()
        if max(math.dist(base_readback[key], anchors[key]) for key in anchors) > .004:
            raise RuntimeError(f"Frozen visual does not match the seated pose for {self.name}")
        native_visual = UsdGeom.Imageable(stage.GetPrimAtPath(self._original_path))
        self._original_visibility = native_visual.GetVisibilityAttr().Get()
        native_visual.MakeInvisible()

    def positions(self):
        """Read the actual bound stage animation rather than the disabled agent."""
        cache = UsdSkel.Cache()
        cache.Populate(self.rig, Usd.PrimDefaultPredicate)
        transforms = cache.GetSkelQuery(self.skeleton).ComputeJointSkelTransforms(Usd.TimeCode.Default())
        return {
            key: tuple(self.world_from_skel.Transform(transforms[index].ExtractTranslation()))
            for key, index in self.diagnostics.items()
        }

    def tick(self, elapsed):
        beat = (elapsed - self.stagger) % self.period
        gesture = max(0., math.sin(math.pi * min(1., beat / 2.5))) ** 2
        local = Vt.Matrix4dArray(self.base_local)
        world = self.base_world
        head = self.indices["Head"]
        neck = self.parents[head]
        look = _rotation_at(world[neck], self.right, (17. * gesture))
        look = _rotation_at(look, self.up, 12. * math.sin(math.tau * beat / self.period))
        local[neck] = look * world[self.parents[neck]].GetInverse()

        upper, fore, hand = (self.indices[f"R_{name}"] for name in ("Upperarm", "Forearm", "Hand"))
        shoulder, elbow, wrist = (world[joint].ExtractTranslation() for joint in (upper, fore, hand))
        if gesture > 0.:
            offset = self.forward * .15 + self.right * .025 + self.up * .25
            # The model-space distances differ between metre and centimetre rigs.
            reach_weight = gesture * self.reach_scale
            target = wrist + offset * self.units_per_metre * reach_weight
            original_bend = (elbow - shoulder).GetNormalized()
            reaching_bend = self.right * .5 - self.forward * .2 - self.up
            elbow_hint = original_bend * (1. - reach_weight) + reaching_bend * reach_weight
            posed_wrist, posed_elbow = _wrist_and_elbow(
                shoulder, target, (elbow - shoulder).GetLength(), (wrist - elbow).GetLength(),
                elbow_hint,
            )
            turn = Gf.Matrix4d().SetRotate(Gf.Rotation(elbow - shoulder, posed_elbow - shoulder))
            posed_upper = world[upper] * turn
            posed_upper.SetTranslateOnly(shoulder)
            local[upper] = posed_upper * world[self.parents[upper]].GetInverse()
            current_fore = local[fore] * posed_upper
            current_hand = local[hand] * current_fore
            turn = Gf.Matrix4d().SetRotate(Gf.Rotation(current_hand.ExtractTranslation() - posed_elbow,
                                                       posed_wrist - posed_elbow))
            posed_fore = current_fore * turn
            posed_fore.SetTranslateOnly(posed_elbow)
            local[fore] = posed_fore * posed_upper.GetInverse()
        _, rotations, _ = UsdSkel.DecomposeTransforms(local)
        self.motion.GetRotationsAttr().Set(rotations)

        observed = self.positions()
        excursions = {key: math.dist(observed[key], self.anchors[key]) for key in observed}
        self.drift_m = max(self.drift_m, *(excursions[key] for key in _PLANTED_JOINTS))
        self.head_m = max(self.head_m, excursions["Head"])
        self.hand_m = max(self.hand_m, excursions["L_Hand"], excursions["R_Hand"])
        if self.drift_m > _HOLD_TOLERANCE_M:
            raise RuntimeError(f"Background diner {self.name} moved a lower joint by {self.drift_m:.3f} m")

    def restore_native_visual(self, stage):
        """Release the overlay before the native agent starts walking."""
        native_visual = UsdGeom.Imageable(stage.GetPrimAtPath(self._original_path))
        native_visual.GetVisibilityAttr().Set(self._original_visibility)
        UsdGeom.Imageable(stage.GetPrimAtPath(self._overlay_path)).MakeInvisible()

    def audit(self):
        return dict(name=self.name, maximum_planted_drift_m=self.drift_m,
                    maximum_head_motion_m=self.head_m, maximum_hand_motion_m=self.hand_m,
                    method="session-layer upper-body posing from frozen native seat")


class RestaurantBackgroundMotion:
    """Bring back the overview's gaze and reaches while the original seats stay fixed."""

    def __init__(self, stage, seated_cast, *, cast_manifest, names=None, standing_reader=None):
        self.stage = stage
        self.seated_cast = seated_cast
        available = tuple(row["name"] for row in seated_cast.actors)
        self.names = available if names is None else tuple(names)
        if not self.names or len(self.names) != len(set(self.names)) or not set(self.names).issubset(available):
            raise ValueError("Background motion requires selected diners on the original seated cast")
        self.reader = None
        if standing_reader is not None:
            self.attach_reader(standing_reader)
        if stage.GetPrimAtPath(COUNTER_WORK_PATH):
            raise ValueError("The background counter worker is already staged")
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            worker = _bar_staff(stage, Path(cast_manifest), performer_path=COUNTER_WORK_PATH)
            root = UsdGeom.Xformable(stage.GetPrimAtPath(COUNTER_WORK_PATH))
            self._worker_base = next(op for op in root.GetOrderedXformOps()
                                     if op.GetOpType() == UsdGeom.XformOp.TypeTranslate)
            worker_yaw = next(op for op in root.GetOrderedXformOps()
                              if op.GetOpType() == UsdGeom.XformOp.TypeRotateZ)
            self._worker_base.Set(Gf.Vec3d(*COUNTER_WORK_POSITION))
            worker_yaw.Set(0.)
            for prim in Usd.PrimRange(root.GetPrim()):
                if prim.IsA(UsdGeom.Mesh):
                    prim.AddAppliedSchema("NavMeshExcludeAPI")
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
            worker.update(position=COUNTER_WORK_POSITION, facing=(0., -1., 0.), yaw_degrees=0.)
            self._counter = BarHandoff(stage, worker, _COUNTER_REACH, _COUNTER_REACH,
                                       turn_with_plate=False, max_lean_degrees=12.)
        self._start_time = None
        self._last_time = None
        self._last_pose_time = -math.inf
        self._gestures = []
        self._table_gesture = None
        self._table_gesture_audit = None
        self._departed_diner = None
        self._samples = 0
        self._bar_wrist_start = None
        self._bar_wrist_excursion_m = 0.
        self._bar_base_drift_m = 0.

    def start(self, sim_time_s):
        """Start after the original native seating has settled and been frozen."""
        start_time = float(sim_time_s)
        if self._start_time is not None or not math.isfinite(start_time):
            raise ValueError("Background motion requires a finite, unused start time")
        with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            self._gestures = [
                _SeatedGesture(self.stage, spec, agent, anchor, index)
                for index, (spec, agent, anchor) in enumerate(self.seated_cast.frozen_seated_agents(self.names))
            ]
        self._start_time = start_time
        self._last_time = start_time

    def attach_reader(self, standing_reader):
        """Leave the original reader absent until the right phase stages him."""
        if self.reader is not None:
            raise RuntimeError("The newspaper reader already has a motion overlay")
        self.reader = NewspaperReaderMotion(self.stage, standing_reader)

    def attach_departed_diner(self, diner):
        """Animate the lounge visual using the held pose from the native walk."""
        if self._departed_diner is diner:
            return
        if self._start_time is None or self._departed_diner is not None:
            raise RuntimeError("Start seated background motion before attaching the lounge diner once")
        spec, agent, anchor = diner.frozen_lounge_gesture_binding()
        with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            self._gestures.append(_SeatedGesture(
                self.stage, spec, agent, anchor, len(self._gestures), overlay_name="LoungeGestureVisual",
            ))
        self._departed_diner = diner

    def sync_window_diner(self, diner):
        """Switch presentation visuals at settled seats and restore native walking."""
        if self._start_time is None:
            raise RuntimeError("Start seated background motion before syncing the window diner")
        if diner.phase == "table_hold" and self._table_gesture is None:
            spec, agent, anchor = diner.frozen_table_gesture_binding()
            with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
                self._table_gesture = _SeatedGesture(self.stage, spec, agent, anchor, len(self._gestures))
            self._gestures.append(self._table_gesture)
        elif diner.phase != "table_hold" and self._table_gesture is not None:
            with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
                self._table_gesture.restore_native_visual(self.stage)
            self._table_gesture_audit = self._table_gesture.audit()
            self._gestures.remove(self._table_gesture)
            self._table_gesture = None
        if diner.phase == "away_endpoint":
            self.attach_departed_diner(diner)

    def tick(self, sim_time_s):
        """Supply gestures and sample native joint readbacks on the simulation clock."""
        now = float(sim_time_s)
        if self._start_time is None or not math.isfinite(now) or now < self._last_time:
            raise ValueError("Start background motion before supplying monotonic simulation times")
        self._last_time = now
        if now - self._last_pose_time < _POSE_INTERVAL_S - 1.e-9:
            return
        self._last_pose_time = now
        elapsed = now - self._start_time
        self._samples += 1
        with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            for actor in self._gestures:
                actor.tick(elapsed)
        if self.reader is not None:
            self.reader.tick(elapsed)

        # Reach toward the counter, then settle back to the relaxed pose.
        # The same planted server rig and hand solver are used for meal pickup.
        progress = .23 * (1. - math.cos(math.tau * elapsed / 5.1))
        wrists = self._counter.update(progress)["wrists"]
        wrist_xyz = {side: row["wrist_xyz"] for side, row in wrists.items()}
        if self._bar_wrist_start is None:
            self._bar_wrist_start = wrist_xyz
        excursion = max(math.dist(wrist_xyz[side], start) for side, start in self._bar_wrist_start.items())
        self._bar_wrist_excursion_m = max(self._bar_wrist_excursion_m, excursion)
        worker_base = tuple(self._worker_base.Get())
        self._bar_base_drift_m = max(self._bar_base_drift_m, math.dist(worker_base, COUNTER_WORK_POSITION))

    def audit(self):
        """Report planted joints and observed upper-body motion for a recorded preview."""
        diners = [actor.audit() for actor in self._gestures]
        return dict(samples=self._samples, pose_update_interval_s=_POSE_INTERVAL_S,
                    seated_diners=diners, planted_joints=_PLANTED_JOINTS,
                    planted_joint_tolerance_m=_HOLD_TOLERANCE_M,
                    initial_window_seat=self._table_gesture_audit,
                    newspaper_reader=None if self.reader is None else self.reader.audit(),
                    counter_worker=dict(prim_path=COUNTER_WORK_PATH, world_xyz=COUNTER_WORK_POSITION,
                                        maximum_base_drift_m=self._bar_base_drift_m,
                                        maximum_wrist_excursion_m=self._bar_wrist_excursion_m))

"""Session-only bar-to-tray presentation at the common pickup mark."""

import hashlib
import json
import math
from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdSkel

from robot.evaluation.restaurant_right_route import PICKUP
from robot.evaluation.service_left_outbound_probe import LEFT_PICKUP
from robot.scene.restaurant_demo_build import PAYLOAD_PATH
from robot.scene.restaurant_live_bar_handoff import BarHandoff, _ease, _wrist_and_elbow
from robot.scene.restaurant_live_physical_handoff import (
    BAR_PERFORMER_PATH,
    BAR_SERVER,
    MAXIMUM_DROP_M,
    MAXIMUM_PARKED_TRAY_DRIFT_M,
    MAXIMUM_WRIST_ERROR_M,
    MEAL_PATH,
    _GRAVITY_M_S2,
    _PICKUP,
    _CarterMealPresentation,
    _SurfaceSpot,
    _bar_staff,
    _visibility,
)


RIGHT_TABLE = _SurfaceSpot(
    "/World/Environment/Restaurant/assets_restaurant/Assets_Banquettes_Right_01/Geo_TableSet_14/rh_table",
    (31.62, 20.45, .7466),
)
HANDOFF_DURATION_S = 7.5
_LIFT_END_S = 2.05
_WALK_START_S = 2.65
_WALK_END_S = 5.15
_LOWER_END_S = 6.80
_SERVER_TRAY_MARK = (26.00, 4.86, 0.)
_CARRY_HEIGHT_M = 1.03
_CARRY_REACH_M = .34


class LiveRightCarterHandoff(_CarterMealPresentation):
    """Lift at the counter, carry the plate to Carter and load the parked tray."""

    def __init__(self, stage, cast_manifest, *, meal_path=MEAL_PATH,
                 performer_path=BAR_PERFORMER_PATH, deliver_left=False):
        super().__init__(stage, meal_path=meal_path)
        self.deliver_left = deliver_left
        if not self.deliver_left:
            self.delivery = RIGHT_TABLE
        self.tray_top, self.meal_yaw = self._tray_pose()
        pickup = LEFT_PICKUP if self.deliver_left else PICKUP
        x, y, yaw = pickup
        expected = (x - .04 * math.cos(yaw), y - .04 * math.sin(yaw))
        heading_error = abs(math.remainder(self.meal_yaw - yaw, math.tau))
        if math.dist(self.tray_top[:2], expected) > .06 or heading_error > math.radians(5.):
            raise ValueError("Carter must settle at the common pickup facing the delivery aisle")
        self._loaded = self._performer_cleared = False
        self.elapsed = -math.inf
        manifest = Path(cast_manifest).resolve()
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            self.server = _bar_staff(stage, manifest, performer_path=performer_path)
            performer = UsdGeom.Xformable(stage.GetPrimAtPath(self.server["root_path"]))
            self.server_translate = next(op for op in performer.GetOrderedXformOps()
                                         if op.GetOpType() == UsdGeom.XformOp.TypeTranslate)
            self.handoff = BarHandoff(
                stage, self.server, _PICKUP.xyz, (25.80, 6.29, .93),
                transfer_via=(25.31, 5.65), lift_height=.18, max_lean_degrees=55.,
                plate_radius=.09, allow_crouch=True,
            )
            self.release_z, reach = self._preflight()
            self.drop_m = self.release_z - self.tray_top[2]
            self.drop_duration_s = math.sqrt(2. * self.drop_m / _GRAVITY_M_S2)
            self._author_meal_visual()
            self._support(_PICKUP)
            self._support(self.delivery)
            self._set_meal_pose(_PICKUP.xyz)
            for prim in self.payload:
                _visibility(prim, False)
            hidden = []
            for name in ("CounterPlateA", "CounterPlateB"):
                prim = stage.GetPrimAtPath("/World/RestaurantLiveV1/Dressing/" + name)
                if prim and prim.IsActive():
                    _visibility(prim, False)
                    hidden.append(str(prim.GetPath()))
            self._pose(0., self.release_z)
        self.metadata = dict(
            actor=self.server, cast_manifest=str(manifest),
            cast_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
            configured_pickup_world_xy_yaw=list(pickup),
            counter_pickup_world_xyz=list(_PICKUP.xyz), pickup_surface=_PICKUP.path,
            server_tray_mark_world_xyz=list(_SERVER_TRAY_MARK),
            measured_tray_top_world_xyz=list(self.tray_top), mounted_meal_path=PAYLOAD_PATH,
            table_meal_path=self.meal_path,
            delivery_surface=self.delivery.path, delivery_world_xyz=list(self.delivery.xyz),
            handoff_duration_s=HANDOFF_DURATION_S, release_drop_m=self.drop_m,
            release_drop_duration_s=self.drop_duration_s, selected_release_z_m=self.release_z,
            selected_crouch_m=self.handoff.crouch_depth_m, reach_preflight=reach,
            maximum_preflight_wrist_error_m=reach["worst_error_m"], hidden_dressing=hidden,
            transfer_mode="counter_lift_carried_to_tray_then_scripted_table_placement",
            scope="Authored stepping and two-hand plate placement before wheels; no force or free payload dynamics.",
        )

    @staticmethod
    def _carry_mark(t):
        start, end = Gf.Vec3d(*BAR_SERVER), Gf.Vec3d(*_SERVER_TRAY_MARK)
        corner = Gf.Vec3d(end[0], start[1], 0.)
        xyz = start * (1.-t)**2 + corner * (2.*t*(1.-t)) + end*t*t
        facing = ((corner-start)*(1.-t) + (end-corner)*t).GetNormalized()
        return xyz, facing

    def _set_server(self, xyz, facing):
        self.server_translate.Set(xyz)
        yaw = math.degrees(math.atan2(facing[1], facing[0])) + 90.
        self.handoff.initial_yaw = yaw
        self.handoff.yaw_op.Set(yaw)

    def _pose_with_dish(self, xyz, dish, phase, progress):
        facing = Gf.Vec3d(dish[0]-xyz[0], dish[1]-xyz[1], 0.).GetNormalized()
        self._set_server(xyz, facing)
        grip = 1. - _ease(progress, .88, .98)
        return self.handoff._pose_frame(dish, phase, progress, grip, facing, 0.)

    def _pose(self, elapsed, release_z):
        root = Gf.Vec3d(*BAR_SERVER)
        if elapsed <= _LIFT_END_S:
            self._set_server(root, Gf.Vec3d(-1., 0., 0.))
            return self.handoff.update(.51 * elapsed / _LIFT_END_S)

        if elapsed < _WALK_START_S:
            amount = _ease(elapsed, _LIFT_END_S, _WALK_START_S)
            angle = math.pi * (1. - amount)
            reach = .51 * (1. - amount) + _CARRY_REACH_M * amount
            dish = root + Gf.Vec3d(math.cos(angle)*reach, math.sin(angle)*reach,
                                   1.11 * (1. - amount) + _CARRY_HEIGHT_M * amount)
            return self._pose_with_dish(root, dish, "turn_with_plate", .60)

        if elapsed < _WALK_END_S:
            t = (elapsed - _WALK_START_S) / (_WALK_END_S - _WALK_START_S)
            eased = t*t*(3. - 2.*t)
            xyz, facing = self._carry_mark(eased)
            dish = xyz + facing * _CARRY_REACH_M + Gf.Vec3d(0., 0., _CARRY_HEIGHT_M)
            result = self._pose_with_dish(xyz, dish, "carry_to_pickup_mark", .62)
            self._step_legs(eased, facing)
            return result

        xyz, facing = self._carry_mark(1.)
        carried = xyz + facing * _CARRY_REACH_M + Gf.Vec3d(0., 0., _CARRY_HEIGHT_M)
        release = Gf.Vec3d(*self.tray_top[:2], release_z)
        lower = _ease(elapsed, _WALK_END_S, _LOWER_END_S)
        dish = carried * (1. - lower) + release * lower
        if elapsed < _LOWER_END_S:
            return self._pose_with_dish(xyz, dish, "lower_to_tray", .73 + .149 * lower)
        withdraw = .88 + .12 * _ease(elapsed, _LOWER_END_S, HANDOFF_DURATION_S)
        return self._pose_with_dish(xyz, release, "release", withdraw)

    def _step_legs(self, t, facing):
        rig = self.handoff
        pose = rig.pose
        local = UsdSkel.MakeTransforms(pose.GetTranslationsAttr().Get(), pose.GetRotationsAttr().Get(),
                                       pose.GetScalesAttr().Get())
        world = UsdSkel.ConcatJointTransforms(rig.topology, local)
        inverse = UsdGeom.Xformable(rig.skeleton).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse()
        forward = inverse.TransformDir(facing).GetNormalized()
        up = inverse.TransformDir(Gf.Vec3d(0., 0., 1.)).GetNormalized()
        weight = _ease(t, 0., .16) * (1. - _ease(t, .84, 1.))
        for _, sign, thigh, calf, foot, thigh_length, calf_length in rig.legs:
            cycle = sign * math.sin(t * math.tau * 2.)
            target = world[foot].ExtractTranslation() + weight * (forward * (.16 * cycle) + up * (.08 * max(0., cycle)))
            ankle, knee = _wrist_and_elbow(world[thigh].ExtractTranslation(), target,
                                          thigh_length, calf_length, forward)
            world = rig._aim(local, world, thigh, calf, knee)
            world = rig._aim(local, world, calf, foot, ankle)
            flat = Gf.Matrix4d(rig.base_world[foot])
            flat.SetTranslateOnly(world[foot].ExtractTranslation())
            world = rig._set_joint(local, world, foot, flat)
        translations, rotations, _ = UsdSkel.DecomposeTransforms(local)
        pose.GetTranslationsAttr().Set(translations)
        pose.GetRotationsAttr().Set(rotations)

    def _preflight(self):
        floor = float(self.tray_top[2])
        attempts = []
        for depth in (0., .12, .18, .24, .30):
            self.handoff.crouch_depth_m = depth
            for drop_cm in range(0, round(MAXIMUM_DROP_M * 100) + 1, 5):
                z = floor + drop_cm / 100.
                errors = {}
                try:
                    for i in range(161):
                        elapsed = 1.35 + (_LOWER_END_S - 1.35) * i / 161.
                        receipt = self._pose(elapsed, z)
                        phase = receipt["phase"]
                        error = max(hand["error_m"] for hand in receipt["wrists"].values())
                        errors[phase] = max(errors.get(phase, 0.), error)
                except ValueError as error:
                    attempts.append(dict(release_z_m=z, crouch_depth_m=depth, pose_error=str(error)))
                    break
                worst = max(errors.values())
                attempts.append(dict(tray_z_m=floor, release_z_m=z, crouch_depth_m=depth,
                                     maximum_wrist_error_by_phase_m=errors, worst_error_m=worst))
                if worst <= MAXIMUM_WRIST_ERROR_M:
                    return z, attempts[-1]
        raise ValueError("No verified right tray release within 25 cm: " + json.dumps(attempts))

    def tick(self, elapsed_sim_s):
        elapsed = float(elapsed_sim_s)
        if not math.isfinite(elapsed) or elapsed < max(0., self.elapsed):
            raise ValueError("Right handoff clock must be finite, nonnegative and nondecreasing")
        if self.ready:
            return self.records[-1]
        self.elapsed = elapsed
        tray, _ = self._tray_pose()
        drift = (tray-self.tray_top).GetLength()
        if drift > MAXIMUM_PARKED_TRAY_DRIFT_M:
            raise RuntimeError("Carter moved before the right pickup completed")
        with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            receipt = self._pose(elapsed, self.release_z)
            phase = receipt["phase"]
            if receipt["hand_grip_weight"] >= .999 and elapsed < _LOWER_END_S and any(
                hand["error_m"] > MAXIMUM_WRIST_ERROR_M for hand in receipt["wrists"].values()
            ):
                raise RuntimeError("Server lost the verified plate grip before the right tray release")
            if elapsed < _LOWER_END_S:
                self._set_meal_pose(receipt["dish_xyz"])
            elif elapsed < _LOWER_END_S + self.drop_duration_s:
                z = self.release_z - .5 * _GRAVITY_M_S2 * (elapsed - _LOWER_END_S)**2
                self._set_meal_pose((*tray[:2], z))
                phase = "short_visible_release"
            elif not self._loaded:
                _visibility(self.meal.GetPrim(), False)
                for prim in self.payload:
                    _visibility(prim, True)
                self._loaded = True
                self.events.append(dict(event="meal_on_physical_tray", elapsed_sim_s=elapsed,
                                        tray_world_xyz=list(tray), release_drop_m=self.drop_m))
        self.ready = self._loaded and elapsed >= HANDOFF_DURATION_S
        row = dict(elapsed_sim_s=elapsed, phase=phase, ready=self.ready, tray_drift_m=drift,
                   tray_world_xyz=list(tray), server_world_xyz=list(self.server_translate.Get()),
                   wrist_receipt=receipt)
        self.records.append(row)
        return row

    def clear_performer(self):
        if not self.ready or self._performer_cleared:
            raise RuntimeError("Right performer cut requires one completed pickup")
        with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            prim = self.stage.GetPrimAtPath(self.server["root_path"])
            if not prim or not prim.IsActive():
                raise RuntimeError("Right pickup performer is missing before the presentation cut")
            prim.SetActive(False)
        self._performer_cleared = True
        event = dict(event="bar_performer_cut_before_clean_reference", elapsed_sim_s=self.elapsed,
                     performer_path=self.server["root_path"], meal_remains_on_physical_tray=True)
        self.events.append(event)
        return event

    def finish_delivery(self):
        """Show the transported plate on the verified destination tabletop."""
        event = super().finish_delivery()
        if not self.deliver_left:
            event["event"] = "meal_placed_on_right_table"
        return event

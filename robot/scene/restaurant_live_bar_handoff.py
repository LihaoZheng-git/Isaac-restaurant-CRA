"""Two-handed bar pickup and tray placement in the live presentation session."""

from __future__ import annotations

import math

from pxr import Gf, Usd, UsdGeom, UsdSkel, Vt


_UP = Gf.Vec3d(0, 0, 1)


def _ease(value, start, end):
    weight = min(1., max(0., (value - start) / (end - start)))
    return weight**3 * (weight * (6. * weight - 15.) + 10.)


def _planar(vector):
    return Gf.Vec3d(vector[0], vector[1], 0.)


def _rotation_at(matrix, axis, degrees):
    result = matrix * Gf.Matrix4d().SetRotate(Gf.Rotation(axis, degrees))
    result.SetTranslateOnly(matrix.ExtractTranslation())
    return result


def _basis(fingers, palm):
    fingers = fingers.GetNormalized()
    palm = (palm - fingers * Gf.Dot(fingers, palm)).GetNormalized()
    cross = Gf.Cross(fingers, palm)
    return Gf.Matrix4d(
        *fingers, 0., *palm, 0., *cross, 0., 0., 0., 0., 1.,
    )


def _wrist_and_elbow(shoulder, wrist, upper_length, lower_length, elbow_hint):
    direction = wrist - shoulder
    distance = direction.GetLength()
    forward = direction / max(distance, 1e-7)
    span = min(upper_length + lower_length - .001,
               max(abs(upper_length - lower_length) + .001, distance))
    along = (upper_length**2 - lower_length**2 + span**2) / (2. * span)
    pole = (elbow_hint - forward * Gf.Dot(elbow_hint, forward)).GetNormalized()
    elbow = shoulder + forward * along + pole * math.sqrt(max(0., upper_length**2 - along**2))
    return shoulder + forward * span, elbow


class BarHandoff:
    """Drive the verified bar rig and return the choreographed plate position.

    The dish xyz is the centre of its base. The caller places the actual meal
    prop at that point and can attach it to the tray after the place phase.
    All pose changes go to the stage's session layer, through CRAStaticPose.
    """

    def __init__(self, stage, bar_actor_record, pickup_xyz, tray_xyz, *,
                 plate_radius=.10, wrist_height=.018, lift_height=.11,
                 max_lean_degrees=44., turn_with_plate=True, transfer_via=None,
                 allow_crouch=False):
        self.stage = stage
        self.pickup = Gf.Vec3d(*pickup_xyz)
        self.tray = Gf.Vec3d(*tray_xyz)
        self.via = Gf.Vec3d(*transfer_via[:2], 0.) if transfer_via is not None else None
        self.plate_radius = float(plate_radius)
        self.wrist_height = float(wrist_height)
        self.lift_height = float(lift_height)
        self.max_lean = float(max_lean_degrees)
        self.turn_with_plate = bool(turn_with_plate)
        if not all(math.isfinite(v) for v in (*self.pickup, *self.tray, self.plate_radius,
                                              self.wrist_height, self.lift_height, self.max_lean)):
            raise ValueError("Bar handoff coordinates and rig settings must be finite")
        if self.plate_radius <= 0. or self.lift_height <= 0. or not 0. <= self.max_lean <= 55.:
            raise ValueError("Invalid bar handoff plate, lift or torso lean")
        root_path = bar_actor_record["root_path"]
        root = UsdGeom.Xformable(stage.GetPrimAtPath(root_path))
        self.position = root.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
        self.initial_facing = _planar(Gf.Vec3d(*bar_actor_record["facing"])).GetNormalized()
        self.yaw_op = next(op for op in root.GetOrderedXformOps()
                           if op.GetOpType() == UsdGeom.XformOp.TypeRotateZ)
        self.initial_yaw = float(self.yaw_op.Get())
        visual = stage.GetPrimAtPath(root_path + "/Grounding/Visual")
        self.skeleton = next(UsdSkel.Skeleton(prim) for prim in Usd.PrimRange(visual)
                             if prim.IsA(UsdSkel.Skeleton))
        self.rig = next(UsdSkel.Root(prim) for prim in Usd.PrimRange(visual)
                        if prim.IsA(UsdSkel.Root))
        self.pose = UsdSkel.Animation.Get(stage, visual.GetPath().AppendChild("CRAStaticPose"))
        joints = self.skeleton.GetJointsAttr().Get()
        if not self.pose or self.pose.GetJointsAttr().Get() != joints:
            raise ValueError("Bar server requires the existing relaxed static pose")
        self.topology = UsdSkel.Topology(joints)
        self.parents = self.topology.GetParentIndices()
        self.indices = {joint.rsplit("/", 1)[-1]: index for index, joint in enumerate(joints)}
        self.base_local = UsdSkel.MakeTransforms(self.pose.GetTranslationsAttr().Get(),
                                                 self.pose.GetRotationsAttr().Get(),
                                                 self.pose.GetScalesAttr().Get())
        self.base_world = UsdSkel.ConcatJointTransforms(self.topology, self.base_local)
        self.waist = self.indices["Waist"]
        self.crouch_depth_m = 0.
        self.legs = self._stance_legs() if allow_crouch else ()
        self.arms = {}
        for side, sign in (("L", 1.), ("R", -1.)):
            upper, fore, hand, index, pinky = (self.indices[f"{side}_{part}"] for part in (
                "Upperarm", "Forearm", "Hand", "Index1", "Pinky1",
            ))
            if self.parents[fore] != upper or self.parents[hand] != fore:
                raise ValueError("Unsupported bar server arm hierarchy")
            shoulder, elbow, wrist = (self.base_world[joint].ExtractTranslation()
                                      for joint in (upper, fore, hand))
            world_to_hand = self.base_world[hand].GetInverse()
            knuckle_i = world_to_hand.Transform(self.base_world[index].ExtractTranslation())
            knuckle_p = world_to_hand.Transform(self.base_world[pinky].ExtractTranslation())
            fingers = (knuckle_i + knuckle_p) * .5
            palm = Gf.Cross(knuckle_p - knuckle_i, fingers) * sign
            self.arms[side] = (upper, fore, hand, (elbow - shoulder).GetLength(),
                               (wrist - elbow).GetLength(), _basis(fingers, palm), sign)
        if min((_planar(self._transfer(i / 20.) - self.position).GetLength()
                for i in range(21))) < self.plate_radius + .13:
            raise ValueError("Bar handoff plate path crosses the server; supply a transfer_via point")
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            # The idle SkelRoot extent is measured tightly around lowered arms.
            extent = UsdGeom.Boundable(self.rig).GetExtentAttr()
            low, high = extent.Get()
            padding = Gf.Vec3f(.72, .72, .12)
            extent.Set([low - padding, high + padding])

    def _stance_legs(self):
        hip = self.indices["Hip"]
        legs = []
        for side, sign in (("L", 1.), ("R", -1.)):
            foot = self.indices[side + "_Foot"]
            calf = self.parents[foot]
            thigh = self.parents[calf] if calf >= 0 else -1
            ancestor = self.parents[thigh] if thigh >= 0 else -1
            while ancestor >= 0 and ancestor != hip:
                ancestor = self.parents[ancestor]
            if ancestor != hip:
                raise ValueError(f"Unsupported bar-server leg hierarchy for {side}")
            thigh_xyz, calf_xyz, foot_xyz = (self.base_world[index].ExtractTranslation()
                                               for index in (thigh, calf, foot))
            thigh_length, calf_length = (calf_xyz - thigh_xyz).GetLength(), (foot_xyz - calf_xyz).GetLength()
            if not all(.2 <= length <= .7 for length in (thigh_length, calf_length)):
                raise ValueError(f"Unexpected bar-server leg lengths for {side}")
            legs.append((side, sign, thigh, calf, foot, thigh_length, calf_length))
        return legs

    def _transfer(self, weight):
        if self.via is None:
            return self.pickup * (1. - weight) + self.tray * weight
        control = 2. * self.via - .5 * (_planar(self.pickup) + _planar(self.tray))
        return (self.pickup * (1. - weight)**2 + control * (2. * weight * (1. - weight))
                + self.tray * weight**2)

    def _dish(self, progress):
        lifted = max(self.pickup[2], self.tray[2]) + self.lift_height
        if progress < .38:
            return self.pickup, "reach" if progress > .10 else "idle"
        if progress < .51:
            result = Gf.Vec3d(self.pickup)
            result[2] += (lifted - result[2]) * _ease(progress, .38, .51)
            return result, "lift"
        if progress < .73:
            result = self._transfer(_ease(progress, .51, .73))
            result[2] = lifted
            return result, "transfer"
        if progress < .88:
            result = Gf.Vec3d(self.tray)
            result[2] += (lifted - result[2]) * (1. - _ease(progress, .73, .88))
            return result, "lower"
        return self.tray, "release" if progress < .98 else "complete"

    def _set_joint(self, local, world, joint, desired):
        parent = self.parents[joint]
        local[joint] = desired if parent < 0 else desired * world[parent].GetInverse()
        return UsdSkel.ConcatJointTransforms(self.topology, local)

    def _aim(self, local, world, joint, child, desired):
        origin = world[joint].ExtractTranslation()
        current = world[child].ExtractTranslation() - origin
        delta = Gf.Matrix4d().SetRotate(Gf.Rotation(current, desired - origin))
        matrix = world[joint] * delta
        matrix.SetTranslateOnly(origin)
        return self._set_joint(local, world, joint, matrix)

    def _lean_degrees(self, grip_targets, axis, base_world):
        """Find the smallest waist lean that keeps both arms short of full stretch."""
        pivot = base_world[self.waist].ExtractTranslation()
        low, high = 0., self.max_lean
        for _ in range(13):
            angle = (low + high) / 2.
            turn = Gf.Matrix4d().SetRotate(Gf.Rotation(axis, angle))
            reachable = True
            for side, (upper, _, _, upper_length, lower_length, _, _) in self.arms.items():
                shoulder = base_world[upper].ExtractTranslation()
                moved = pivot + turn.TransformDir(shoulder - pivot)
                if (grip_targets[side] - moved).GetLength() > upper_length + lower_length - .018:
                    reachable = False
                    break
            if reachable:
                high = angle
            else:
                low = angle
        return high

    def _apply_crouch(self, local, world, progress, up, forward, left):
        depth = self.crouch_depth_m * _ease(progress, .735, .865) * (1. - _ease(progress, .90, 1.))
        if depth < .001:
            return world, 0., {}
        if not self.legs or not math.isfinite(depth) or depth > .30:
            raise ValueError("The parked bar rig requires a bounded crouch")
        hip = self.indices["Hip"]
        lowered = Gf.Matrix4d(world[hip])
        lowered.SetTranslateOnly(lowered.ExtractTranslation() - up * depth)
        world = self._set_joint(local, world, hip, lowered)
        errors = {}
        for side, sign, thigh, calf, foot, thigh_length, calf_length in self.legs:
            foot_pose = self.base_world[foot]
            foot_xyz = foot_pose.ExtractTranslation()
            ankle, knee = _wrist_and_elbow(world[thigh].ExtractTranslation(), foot_xyz,
                                          thigh_length, calf_length, forward + left * (.1 * sign))
            errors[side] = (ankle - foot_xyz).GetLength()
            if errors[side] > .003:
                raise ValueError(f"Bar-server crouch cannot keep its {side} foot planted")
            world = self._aim(local, world, thigh, calf, knee)
            world = self._aim(local, world, calf, foot, ankle)
            world = self._set_joint(local, world, foot, foot_pose)
        return world, depth, errors

    def update(self, progress):
        """Set the pose at normalized progress and return the dish and wrist readbacks."""
        progress = float(progress)
        if not math.isfinite(progress):
            raise ValueError("Bar handoff progress must be finite")
        progress = min(1., max(0., progress))
        dish, phase = self._dish(progress)
        grip = _ease(progress, .10, .33) * (1. - _ease(progress, .88, .98))
        facing = _planar(dish - self.position).GetNormalized()
        facing_delta = math.degrees(math.atan2(
            Gf.Dot(_UP, Gf.Cross(self.initial_facing, facing)), Gf.Dot(self.initial_facing, facing),
        ))
        with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            return self._pose_frame(dish, phase, progress, grip, facing, facing_delta)

    def _pose_frame(self, dish, phase, progress, grip, facing, facing_delta):
        if self.turn_with_plate:
            self.yaw_op.Set(self.initial_yaw + facing_delta * _ease(progress, .06, .21))
        else:
            facing = self.initial_facing
        skel_to_world = UsdGeom.Xformable(self.skeleton).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default(),
        )
        to_skel = skel_to_world.GetInverse()
        up = to_skel.TransformDir(_UP).GetNormalized()
        forward = to_skel.TransformDir(facing).GetNormalized()
        left = Gf.Cross(up, forward).GetNormalized()
        plate = to_skel.Transform(dish)
        target = {}
        for side, (*_, sign) in self.arms.items():
            # Wrists approach the dish sides from slightly behind. The actual
            # fingers extend under the rim instead of through the food.
            target[side] = (plate + left * (sign * (self.plate_radius + .053))
                            - forward * .033 + up * self.wrist_height)
        local = Vt.Matrix4dArray(self.base_local)
        world, crouch_depth, foot_errors = self._apply_crouch(local, self.base_world, progress, up, forward, left)
        lean = self._lean_degrees(target, left, world) * grip
        waist = _rotation_at(world[self.waist], left, lean)
        world = self._set_joint(local, world, self.waist, waist)
        hands = {}
        for side, (upper, fore, hand, upper_length, lower_length, source_basis, sign) in self.arms.items():
            wanted = world[hand].ExtractTranslation() * (1. - grip) + target[side] * grip
            elbow_hint = left * (.46 * sign) - up * .9 + forward * .10
            wrist, elbow = _wrist_and_elbow(world[upper].ExtractTranslation(), wanted,
                                          upper_length, lower_length, elbow_hint)
            if grip > 0.:
                world = self._aim(local, world, upper, fore, elbow)
                world = self._aim(local, world, fore, hand, wrist)
                fingers = (forward * .58 - left * (.82 * sign)).GetNormalized()
                oriented = source_basis.GetInverse() * _basis(fingers, up)
                rest = world[hand].ExtractRotationQuat()
                rotated = Gf.Slerp(grip, rest, oriented.ExtractRotationQuat())
                posed = Gf.Matrix4d().SetRotate(rotated)
                posed.SetTranslateOnly(world[hand].ExtractTranslation())
                world = self._set_joint(local, world, hand, posed)
            measured = world[hand].ExtractTranslation()
            hands[side] = dict(
                wrist_xyz=list(skel_to_world.Transform(measured)),
                target_xyz=list(skel_to_world.Transform(wanted)),
                elbow_xyz=list(skel_to_world.Transform(world[fore].ExtractTranslation())),
                error_m=(measured - wanted).GetLength(),
            )
        translations, rotations, _ = UsdSkel.DecomposeTransforms(local)
        if self.legs:
            self.pose.GetTranslationsAttr().Set(translations)
        self.pose.GetRotationsAttr().Set(rotations)
        return dict(dish_xyz=list(dish), phase=phase, progress=progress,
                    hand_grip_weight=grip, waist_lean_degrees=lean,
                    facing=list(facing), wrists=hands,
                    crouch_depth_m=crouch_depth, planted_foot_ik_error_m=foot_errors)

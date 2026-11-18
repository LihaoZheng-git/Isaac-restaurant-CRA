"""Newspaper pose and reading motion for the stationary right-side actor."""

import math

from pxr import Gf, Usd, UsdGeom, UsdSkel, Vt

from robot.scene.isaac_restaurant_server import update_static_pose_bounds
from robot.scene.restaurant_live_bar_handoff import _basis, _rotation_at, _wrist_and_elbow


def _pose_reader(stage, visual):
    rig, = (UsdSkel.Root(prim) for prim in Usd.PrimRange(visual) if prim.IsA(UsdSkel.Root))
    skeleton, = (UsdSkel.Skeleton(prim) for prim in Usd.PrimRange(visual) if prim.IsA(UsdSkel.Skeleton))
    joints = skeleton.GetJointsAttr().Get()
    indices = {joint.rsplit("/", 1)[-1]: i for i, joint in enumerate(joints)}
    topology = UsdSkel.Topology(joints)
    parents = topology.GetParentIndices()
    local = Vt.Matrix4dArray(skeleton.GetRestTransformsAttr().Get())
    world = UsdSkel.ConcatJointTransforms(topology, local)

    def set_rotation(joint, matrix):
        matrix.SetTranslateOnly(world[joint].ExtractTranslation())
        local[joint] = matrix * world[parents[joint]].GetInverse()
        return UsdSkel.ConcatJointTransforms(topology, local)

    def aim(joint, child, target):
        origin = world[joint].ExtractTranslation()
        current = world[child].ExtractTranslation() - origin
        rotation = Gf.Matrix4d().SetRotate(Gf.Rotation(current, target - origin))
        return set_rotation(joint, world[joint] * rotation)

    for side, sign in (("L", 1.), ("R", -1.)):
        upper, fore, hand, index, pinky = (
            indices[f"{side}_{part}"] for part in ("Upperarm", "Forearm", "Hand", "Index1", "Pinky1")
        )
        if parents[fore] != upper or parents[hand] != fore:
            raise ValueError("Unsupported newspaper reader arm hierarchy")
        shoulder, elbow, wrist = (world[joint].ExtractTranslation() for joint in (upper, fore, hand))
        to_hand = world[hand].GetInverse()
        index_knuckle = to_hand.Transform(world[index].ExtractTranslation())
        pinky_knuckle = to_hand.Transform(world[pinky].ExtractTranslation())
        fingers = (index_knuckle + pinky_knuckle) * .5
        palm = Gf.Cross(pinky_knuckle - index_knuckle, fingers) * sign
        source_basis = _basis(fingers, palm)

        # This source skeleton is in centimetres and faces local -Y.
        target = Gf.Vec3d(sign * 15., -20., 131.)
        wrist, elbow = _wrist_and_elbow(shoulder, target, (elbow - shoulder).GetLength(),
                                      (wrist - elbow).GetLength(), Gf.Vec3d(sign * .15, -.13, -.9))
        world = aim(upper, fore, elbow)
        world = aim(fore, hand, wrist)
        fingers = Gf.Vec3d(-sign * .86, -.38, .18)
        palm = Gf.Vec3d(0., .53, .85)
        world = set_rotation(hand, source_basis.GetInverse() * _basis(fingers, palm))

    head = indices["Head"]
    world = set_rotation(head, _rotation_at(world[head], Gf.Vec3d(1., 0., 0.), 27.))
    translations, rotations, scales = UsdSkel.DecomposeTransforms(local)
    pose = UsdSkel.Animation.Define(stage, visual.GetPath().AppendChild("ReadingPose"))
    pose.CreateJointsAttr(joints)
    pose.CreateTranslationsAttr(translations)
    pose.CreateRotationsAttr(rotations)
    pose.CreateScalesAttr(scales)
    UsdSkel.BindingAPI.Apply(skeleton.GetPrim()).CreateAnimationSourceRel().SetTargets([pose.GetPath()])
    return pose.GetPath(), rig, skeleton


def _author_newspaper(stage, path):
    """Folded broadsheet and print, drawn on both faces of the paper."""
    newspaper = UsdGeom.Xform.Define(stage, path).GetPrim()
    width, height = .25, .138

    def point(x, row, offset):
        return Gf.Vec3f(x, -.207 - .072 * row - .012 * (1. - abs(x) / (width / 2)) + offset,
                        1.180 + height * row)

    def quads(name, color, rectangles):
        points = []
        for left, right, bottom, top, offset in rectangles:
            points.extend(point(x, row, offset) for x, row in (
                (left, bottom), (right, bottom), (right, top), (left, top),
            ))
        mesh = UsdGeom.Mesh.Define(stage, path.AppendChild(name))
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateDoubleSidedAttr(True)
        mesh.CreatePointsAttr(points)
        mesh.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(points))
        mesh.CreateFaceVertexCountsAttr([4] * len(rectangles))
        mesh.CreateFaceVertexIndicesAttr(list(range(len(points))))
        mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set([Gf.Vec3f(*color)] * len(points))

    half = width / 2
    quads("Paper", (.87, .84, .76), [(-half, 0., 0., 1., 0.), (0., half, 0., 1., 0.)])
    print_rectangles = []
    for offset in (-.001, .001):
        for left in (-.114, .011):
            print_rectangles.append((left, left + .101, .80, .89, offset))
            for column in (left, left + .053):
                for line in range(9):
                    row = .12 + .07 * line
                    length = .036 if line in (0, 3) else .046
                    print_rectangles.append((column, column + length, row, row + .017, offset))
    quads("Print", (.08, .09, .10), print_rectangles)
    return newspaper


def author_reader_visual(stage, root_path, source):
    grounding = UsdGeom.Xform.Define(stage, root_path.AppendChild("Grounding"))
    visual = UsdGeom.Xform.Define(stage, grounding.GetPath().AppendChild("Visual"))
    # A slight stature adjustment keeps the reader within the prior body's footprint.
    visual.AddScaleOp().Set(Gf.Vec3d(.009))
    visual.GetPrim().GetReferences().AddReference(source.as_posix())
    pose, rig, skeleton = _pose_reader(stage, visual.GetPrim())
    low, _ = update_static_pose_bounds(visual.GetPrim(), rig, skeleton)
    # Ground in scene metres. The mesh extent was authored in the rig's own units.
    grounding.AddTranslateOp().Set(Gf.Vec3d(0., 0., -float(low[2])))
    paper = _author_newspaper(stage, root_path.AppendChild("Newspaper"))
    for prim in (rig.GetPrim(), paper):
        prim.AddAppliedSchema("NavMeshExcludeAPI")
    return pose, paper.GetPath()


class NewspaperReaderMotion:
    """Look across the page and briefly glance up with feet and hands planted."""

    def __init__(self, stage, reader):
        self.stage = stage
        self.path = reader["prim_path"]
        self.pose = UsdSkel.Animation.Get(stage, reader["pose_path"])
        joints = self.pose.GetJointsAttr().Get()
        self.topology = UsdSkel.Topology(joints)
        self.parents = self.topology.GetParentIndices()
        by_name = {joint.rsplit("/", 1)[-1]: index for index, joint in enumerate(joints)}
        self.head = by_name["Head"]
        self.neck = self.parents[self.head]
        self.base_local = UsdSkel.MakeTransforms(
            self.pose.GetTranslationsAttr().Get(), self.pose.GetRotationsAttr().Get(),
            self.pose.GetScalesAttr().Get(),
        )
        self.base_world = UsdSkel.ConcatJointTransforms(self.topology, self.base_local)
        self.samples = 0
        self.maximum_glance_degrees = 0.

    def tick(self, elapsed_seconds):
        reading_cycle = (float(elapsed_seconds) % 8.) / 8.
        # Ease briefly toward the room and look back to the paper.
        glance = math.sin(math.pi * min(1., max(0., (reading_cycle - .45) / .34))) ** 2
        tilt = -23. * glance + 3. * math.sin(math.tau * 2. * reading_cycle)
        scan = (7. + 6. * glance) * math.sin(math.tau * reading_cycle)
        local = Vt.Matrix4dArray(self.base_local)
        world = self.base_world
        neck = _rotation_at(world[self.neck], Gf.Vec3d(1., 0., 0.), tilt)
        neck = _rotation_at(neck, Gf.Vec3d(0., 0., 1.), scan)
        local[self.neck] = neck * world[self.parents[self.neck]].GetInverse()
        _, rotations, _ = UsdSkel.DecomposeTransforms(local)
        with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            self.pose.GetRotationsAttr().Set(rotations)
        self.samples += 1
        self.maximum_glance_degrees = max(self.maximum_glance_degrees, math.hypot(tilt, scan))

    def audit(self):
        return dict(prim_path=self.path, samples=self.samples,
                    maximum_glance_degrees=self.maximum_glance_degrees,
                    method="neck-only newspaper reading with fixed arms and feet")

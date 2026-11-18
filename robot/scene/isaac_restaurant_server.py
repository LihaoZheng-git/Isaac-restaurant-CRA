"""Compose the official Isaac human as a stationary restaurant role asset.

NVIDIA geometry/textures remain external references. Pose, collider and bounds
are authored scene data, never perception or CRA inputs.
"""

import hashlib
import os
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdSkel

from robot.scene.controlled_semantic_obstacle import read_world_bounds
from robot.scene.restaurant_server import (
    SERVER_PRIM_PATH,
    SERVER_WORLD_XYZ,
    SERVER_YAW_DEGREES,
)

ISAAC_HUMAN_URL = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/People/Characters/F_Business_02/F_Business_02.usd"
ISAAC_HUMAN_SHA256 = "a1dceaae15e59e1cda4b4697a1e233c0cc570279178a418e544289b202c78201"


def validate_isaac_human_bytes(data):
    if hashlib.sha256(data).hexdigest() != ISAAC_HUMAN_SHA256:
        raise ValueError(
            "official Isaac human fingerprint changed; review before using"
        )


def author_relaxed_static_pose(stage, visual_prim):
    """Lower the two upper arms in the supplied rig's own rest coordinates.

    This is an authored static presentation pose, not retargeted idle/walking
    animation. Local pose arrays are generated only in the caller's scene layer.
    """
    roots = [p for p in Usd.PrimRange(visual_prim) if p.IsA(UsdSkel.Root)]
    skeletons = [p for p in Usd.PrimRange(visual_prim) if p.IsA(UsdSkel.Skeleton)]
    if len(roots) != 1 or len(skeletons) != 1:
        raise ValueError("stationary human requires one unambiguous skeleton/root")
    skeleton = UsdSkel.Skeleton(skeletons[0])
    joints = list(skeleton.GetJointsAttr().Get())
    rest = skeleton.GetRestTransformsAttr().Get()
    topology = UsdSkel.Topology(joints)
    world = UsdSkel.ConcatJointTransforms(topology, rest)
    translations, rotations, scales = UsdSkel.DecomposeTransforms(rest)
    parents = topology.GetParentIndices()
    for arm, forearm, side in (
        ("L_Upperarm", "L_Forearm", 1),
        ("R_Upperarm", "R_Forearm", -1),
    ):
        matches = [i for i, name in enumerate(joints) if name.rsplit("/", 1)[-1] == arm]
        children = [
            i for i, name in enumerate(joints) if name.rsplit("/", 1)[-1] == forearm
        ]
        if (
            len(matches) != 1
            or len(children) != 1
            or parents[children[0]] != matches[0]
        ):
            raise ValueError("unsupported upper-arm/forearm topology")
        index, child = matches[0], children[0]
        direction = (
            world[child].ExtractTranslation() - world[index].ExtractTranslation()
        )
        delta = Gf.Matrix4d().SetRotate(
            Gf.Rotation(direction, Gf.Vec3d(0.15 * side, 0, -0.98))
        )
        posed = world[index] * delta
        posed.SetTranslateOnly(world[index].ExtractTranslation())
        local = posed * world[parents[index]].GetInverse()
        rotations[index] = Gf.Quatf(Gf.Transform(local).GetRotation().GetQuat())
    pose = UsdSkel.Animation.Define(
        stage, str(visual_prim.GetPath()) + "/CRAStaticPose"
    )
    pose.CreateJointsAttr(joints)
    pose.CreateTranslationsAttr(translations)
    pose.CreateRotationsAttr(rotations)
    pose.CreateScalesAttr(scales)
    UsdSkel.BindingAPI.Apply(skeleton.GetPrim()).CreateAnimationSourceRel().SetTargets(
        [pose.GetPath()]
    )
    return update_static_pose_bounds(visual_prim, UsdSkel.Root(roots[0]), skeleton)


def update_static_pose_bounds(visual_prim, rig, skeleton):
    """Update the rig's extent from the composed, skinned default pose."""
    cache = UsdSkel.Cache()
    cache.Populate(rig, Usd.PrimDefaultPredicate)
    query = cache.GetSkelQuery(skeleton)
    skin = query.ComputeSkinningTransforms(Usd.TimeCode.Default())
    skel_world = UsdGeom.Xformable(skeleton).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    points_world = []
    for prim in Usd.PrimRange(visual_prim):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        skin_query = cache.GetSkinningQuery(prim)
        if skin_query:
            if not skin_query.ComputeSkinnedPoints(
                skin, points, Usd.TimeCode.Default()
            ):
                raise ValueError("static human skinning failed")
            transform = skel_world
        else:
            transform = mesh.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        points_world.extend(tuple(transform.Transform(Gf.Vec3d(*p))) for p in points)
    points = np.asarray(points_world)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("human has no finite posed mesh geometry")
    low, high = points.min(0), points.max(0)
    if not 1.2 <= high[2] - low[2] <= 2.2 or max(high[:2] - low[:2]) > 1.1:
        raise ValueError("posed human extent violates the bounded stationary role")
    # A SkelRoot's declared extent must cover its actual posed surface, rather
    # than leaving the bind-pose T-pose extent as the independent monitor box.
    root_inverse = (
        UsdGeom.Xformable(rig)
        .ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        .GetInverse()
    )
    root_points = np.asarray(
        [tuple(root_inverse.Transform(Gf.Vec3d(*p))) for p in points]
    )
    UsdGeom.Boundable(rig).CreateExtentAttr(
        [Gf.Vec3f(*root_points.min(0)), Gf.Vec3f(*root_points.max(0))]
    )
    return low, high


def build_isaac_server_scene(output, base_scene, *, asset_url=ISAAC_HUMAN_URL):
    output, base_scene = Path(output).resolve(), Path(base_scene).resolve()
    additions = output.with_name(output.stem + "_server.usda")
    if output.exists() or additions.exists() or output == base_scene:
        raise ValueError("Isaac server requires fresh output layers")
    base = Usd.Stage.Open(str(base_scene))
    source = Usd.Stage.Open(str(asset_url))
    if any(
        UsdGeom.GetStageMetersPerUnit(s) != 1 or UsdGeom.GetStageUpAxis(s) != "Z"
        for s in (base, source)
    ):
        raise ValueError("Isaac server source and restaurant require metres/Z-up")
    if (
        not source.GetDefaultPrim().IsValid()
        or base.GetPrimAtPath(SERVER_PRIM_PATH).IsValid()
        or base.GetEndTimeCode() <= base.GetStartTimeCode()
    ):
        raise ValueError("invalid source, duplicate server or missing simulation clock")
    before = hashlib.sha256(base_scene.read_bytes()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, "Z")
    stage.SetStartTimeCode(base.GetStartTimeCode())
    stage.SetEndTimeCode(base.GetEndTimeCode())
    stage.SetTimeCodesPerSecond(base.GetTimeCodesPerSecond())
    stage.SetFramesPerSecond(base.GetFramesPerSecond())
    layer = Sdf.Layer.CreateNew(str(additions))
    stage.GetRootLayer().subLayerPaths = [
        additions.name,
        os.path.relpath(base_scene, output.parent).replace("\\", "/"),
    ]
    with Usd.EditContext(stage, layer):
        server = UsdGeom.Xform.Define(stage, SERVER_PRIM_PATH)
        grounding = UsdGeom.Xform.Define(stage, SERVER_PRIM_PATH + "/Grounding")
        visual = UsdGeom.Xform.Define(stage, SERVER_PRIM_PATH + "/Grounding/Visual")
        visual.GetPrim().GetReferences().AddReference(str(asset_url))
        low, high = author_relaxed_static_pose(stage, visual.GetPrim())
        grounding.AddTranslateOp().Set(Gf.Vec3d(0, 0, -float(low[2])))
        radius = float(np.linalg.norm(high[:2] - low[:2]) / 2)
        height = float(high[2] - low[2])
        if 2 * radius >= height:
            raise ValueError("invalid static human collider envelope")
        collider = UsdGeom.Capsule.Define(stage, SERVER_PRIM_PATH + "/StaticCollision")
        collider.CreateAxisAttr("Z")
        collider.CreateRadiusAttr(radius)
        collider.CreateHeightAttr(height - 2 * radius)
        collider.AddTranslateOp().Set(
            Gf.Vec3d(
                float((low[0] + high[0]) / 2), float((low[1] + high[1]) / 2), height / 2
            )
        )
        collider.CreatePurposeAttr("guide")
        UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
        server.AddTranslateOp().Set(Gf.Vec3d(*SERVER_WORLD_XYZ))
        server.AddRotateZOp().Set(SERVER_YAW_DEGREES)
    layer.Save()
    stage.GetRootLayer().Save()
    if hashlib.sha256(base_scene.read_bytes()).hexdigest() != before:
        raise RuntimeError("source restaurant changed")
    bounds = read_world_bounds(stage, SERVER_PRIM_PATH)
    return {
        "scene": str(output),
        "base_unchanged": True,
        "base_sha256": before,
        "asset_url": str(asset_url),
        "provider": "NVIDIA Isaac Sim People F_Business_02",
        "pose": "project-authored relaxed upper arms; static default transforms, no walking/idle retargeting",
        "authored_posed_height_m": height,
        "static_collision_radius_m": radius,
        "bounds_minimum": bounds.minimum,
        "bounds_maximum": bounds.maximum,
        "humans_added": 1,
        "actor_motion_enabled": False,
        "factor_inputs_supplied": False,
        "nvidia_asset_bytes_vendored": False,
        "scope": "stationary scene asset replacement; no perception/navigation acceptance or calibrated human mass",
    }

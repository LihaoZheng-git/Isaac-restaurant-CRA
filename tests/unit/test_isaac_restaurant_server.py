"""External human composition contracts, using a tiny original test rig."""

import hashlib

import numpy as np
import pytest
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdSkel

from robot.scene.isaac_restaurant_server import (
    author_relaxed_static_pose,
    build_isaac_server_scene,
    validate_isaac_human_bytes,
)
from robot.scene.restaurant_server import SERVER_PRIM_PATH


def human_fixture(path, *, axis="Z"):
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, axis)
    root = UsdGeom.Xform.Define(stage, "/Human")
    stage.SetDefaultPrim(root.GetPrim())
    UsdSkel.Root.Define(stage, "/Human/Rig")
    skel = UsdSkel.Skeleton.Define(stage, "/Human/Rig/Skeleton")
    joints = [
        "Root",
        "Root/L_Upperarm",
        "Root/L_Upperarm/L_Forearm",
        "Root/R_Upperarm",
        "Root/R_Upperarm/R_Forearm",
        "Root/Head",
        "Root/Foot",
    ]
    positions = [
        (0, 0, 1),
        (0.1, 0, 0.3),
        (0.4, 0, 0),
        (-0.1, 0, 0.3),
        (-0.4, 0, 0),
        (0, 0, 0.7),
        (0, 0, -1),
    ]
    rest = [Gf.Matrix4d().SetTranslate(Gf.Vec3d(*p)) for p in positions]
    skel.CreateJointsAttr(joints)
    skel.CreateRestTransformsAttr(rest)
    skel.CreateBindTransformsAttr(
        UsdSkel.ConcatJointTransforms(UsdSkel.Topology(joints), rest)
    )
    mesh = UsdGeom.Mesh.Define(stage, "/Human/Rig/Body")
    mesh.CreatePointsAttr([(0.5, 0, 1.3), (-0.5, 0, 1.3), (0, 0, 1.7), (0, 0, 0)])
    mesh.CreateFaceVertexCountsAttr([3, 3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 3, 1])
    bind = UsdSkel.BindingAPI.Apply(mesh.GetPrim())
    bind.CreateSkeletonRel().SetTargets([skel.GetPath()])
    bind.CreateGeomBindTransformAttr(Gf.Matrix4d(1))
    bind.CreateJointIndicesPrimvar(False, 1).Set([2, 4, 5, 6])
    bind.CreateJointWeightsPrimvar(False, 1).Set([1, 1, 1, 1])
    stage.GetRootLayer().Save()
    return stage


def test_relaxed_pose_moves_arms_down_without_changing_height_or_retiming(tmp_path):
    stage = human_fixture(tmp_path / "human.usda")
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        low, high = author_relaxed_static_pose(stage, stage.GetDefaultPrim())
    assert high[0] - low[0] < 0.5
    assert high[2] - low[2] == pytest.approx(1.7)
    pose = UsdSkel.Animation(stage.GetPrimAtPath("/Human/CRAStaticPose"))
    assert pose.GetRotationsAttr().GetTimeSamples() == []
    assert pose.GetRotationsAttr().Get(0) == pose.GetRotationsAttr().Get(1_000_000)
    cache = UsdSkel.Cache()
    cache.Populate(
        UsdSkel.Root(stage.GetPrimAtPath("/Human/Rig")), Usd.PrimDefaultPredicate
    )
    query = cache.GetSkelQuery(
        UsdSkel.Skeleton(stage.GetPrimAtPath("/Human/Rig/Skeleton"))
    )
    world = query.ComputeJointSkelTransforms()
    assert world[2].ExtractTranslation()[2] < 1.0
    assert world[4].ExtractTranslation()[2] < 1.0


def test_isaac_layer_preserves_sources_and_stays_static_without_factor_labels(tmp_path):
    source = tmp_path / "human.usda"
    human_fixture(source)
    base = tmp_path / "restaurant.usda"
    stage = Usd.Stage.CreateNew(str(base))
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, "Z")
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(1000)
    stage.SetTimeCodesPerSecond(60)
    UsdGeom.Xform.Define(stage, "/World/ExistingRestaurant")
    stage.GetRootLayer().Save()
    hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in (base, source)]
    output = tmp_path / "new.usda"
    report = build_isaac_server_scene(output, base, asset_url=str(source))
    assert hashes == [
        hashlib.sha256(p.read_bytes()).hexdigest() for p in (base, source)
    ]
    assert (
        report["base_unchanged"]
        and not report["actor_motion_enabled"]
        and not report["factor_inputs_supplied"]
    )
    result = Usd.Stage.Open(str(output))
    assert result.GetPrimAtPath("/World/ExistingRestaurant").IsValid()
    assert result.GetTimeCodesPerSecond() == 60 and result.GetEndTimeCode() == 1000
    collision = result.GetPrimAtPath(SERVER_PRIM_PATH + "/StaticCollision")
    assert collision.HasAPI(UsdPhysics.CollisionAPI)
    assert UsdGeom.Imageable(collision).GetPurposeAttr().Get() == "guide"
    assert not collision.HasAPI(UsdPhysics.RigidBodyAPI)
    assert all(not p.HasAPI(UsdPhysics.MassAPI) for p in result.Traverse())
    assert report["bounds_minimum"][2] == pytest.approx(0, abs=1e-6)
    assert np.isfinite(report["bounds_maximum"]).all()
    with pytest.raises(ValueError, match="fresh"):
        build_isaac_server_scene(output, base, asset_url=str(source))


def test_changed_official_asset_is_not_silently_accepted():
    with pytest.raises(ValueError, match="fingerprint"):
        validate_isaac_human_bytes(b"different or unavailable character")


def test_unsupported_human_topology_remains_a_build_failure(tmp_path):
    stage = human_fixture(tmp_path / "human.usda")
    stage.RemovePrim("/Human/Rig/Skeleton")
    with pytest.raises(ValueError, match="unambiguous"):
        author_relaxed_static_pose(stage, stage.GetDefaultPrim())

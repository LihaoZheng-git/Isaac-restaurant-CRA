"""Real prop geometry remains rendering/evaluation-only and original assets stay intact."""

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics
import pytest

from robot.scene.restaurant_container_bench import CONTAINER_PATH
from robot.scene.restaurant_demo_build import DEMO_ROOT, author_demo_props
from robot.scene.restaurant_real_asset_bench import KINDS, VASE_VARIANTS, author_real_asset
from test_restaurant_demo_build import _stage


@pytest.mark.parametrize("kind", KINDS)
def test_original_mesh_is_grounded_and_not_a_factor_input(kind):
    stage = _stage()
    author_demo_props(stage)
    UsdGeom.Xform.Define(stage, "/World/CRAServiceActors/Server")
    before = stage.GetRootLayer().ExportToString()
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        row = author_real_asset(stage, kind)
    assert stage.GetRootLayer().ExportToString() == before
    assert not stage.GetPrimAtPath(DEMO_ROOT + "/RoundedBin").IsActive()
    assert not stage.GetPrimAtPath("/World/CRAServiceActors/Server").IsActive()
    assert row["available_as_physical_factor_input"] is False
    lo, hi = np.array(row["bounds_minimum"]), np.array(row["bounds_maximum"])
    assert abs(lo[2] - (.45 if kind == "cleaver" else 0.)) < 1e-6
    np.testing.assert_allclose(((lo + hi) / 2)[:2], [28.15, 6.1], atol=1e-6)
    if kind in ("cleaver", "broken_plate"):
        np.testing.assert_allclose(max((hi - lo)[:2]), .32 if kind == "cleaver" else .28, atol=1e-6)
        assert 0 < hi[2] - lo[2] < .03  # Do not thicken the edge for LiDAR.
    else:
        assert .2 < hi[2] < .7
    if kind == "simple_trash_can":
        np.testing.assert_allclose(hi[2], .50, atol=1e-6)
        np.testing.assert_allclose((hi - lo)[:2], [.29600188, .29600188], atol=1e-5)
    meshes = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(CONTAINER_PATH)):
        assert not prim.HasAPI(UsdPhysics.MassAPI)
        assert not prim.HasAttribute("phi")
        assert not prim.HasAttribute("semantic:class")
        if prim.IsA(UsdGeom.Mesh):
            meshes.append(prim)
            assert prim.HasAPI(UsdPhysics.CollisionAPI)
            if kind == "simple_trash_can":
                mesh = UsdGeom.Mesh(prim)
                assert len(mesh.GetFaceVertexCountsAttr().Get()) == 668
                for attr in (mesh.GetPointsAttr(), mesh.GetFaceVertexIndicesAttr()):
                    assert all(spec.layer not in (stage.GetRootLayer(), stage.GetSessionLayer())
                               for spec in attr.GetPropertyStack())
    assert len(meshes) == (3 if kind in ("cleaver", "broken_plate") else
                           2 if kind in VASE_VARIANTS else 1)
    support = stage.GetPrimAtPath(DEMO_ROOT + "/AssetBenchSupport")
    assert bool(support) == (kind == "cleaver")
    if support:
        # Support is independent occupancy, not part of the target's bounds.
        support_bounds = UsdGeom.BBoxCache(0, ["default", "render"]).ComputeWorldBound(support).ComputeAlignedRange()
        assert abs(support_bounds.GetMax()[2] - lo[2]) < 1e-6
    with pytest.raises(ValueError, match="original bin"):
        author_real_asset(stage, kind)


def test_vase_pair_preserves_common_scale_and_original_inner_outer_meshes():
    extents, scales = {}, []
    for kind, selected in VASE_VARIANTS.items():
        stage = _stage()
        author_demo_props(stage)
        row = author_real_asset(stage, kind)
        orient = stage.GetPrimAtPath(CONTAINER_PATH + "/Orientation")
        scales.append(orient.GetAttribute("xformOp:scale").Get())
        parent = stage.GetPrimAtPath(str(orient.GetPath()) + "/Asset/root/GLTF_SceneRootNode")
        assert [p.GetName() for p in parent.GetChildren()] == [selected]
        meshes = [UsdGeom.Mesh(p) for p in Usd.PrimRange(parent) if p.IsA(UsdGeom.Mesh)]
        assert len(meshes) == 2
        # No local point/topology edits: both original ceramic surfaces survive.
        for mesh in meshes:
            for attr in (mesh.GetPointsAttr(), mesh.GetFaceVertexIndicesAttr()):
                assert all(spec.layer != stage.GetRootLayer() for spec in attr.GetPropertyStack())
        extents[kind] = np.array(row["bounds_maximum"]) - row["bounds_minimum"]
    np.testing.assert_allclose(scales[0], scales[1], rtol=0, atol=0)
    np.testing.assert_allclose(extents["intact_vase"][2], .40, atol=1e-6)
    np.testing.assert_allclose(extents["intact_vase"][:2], extents["broken_vase"][:2], atol=1e-6)
    assert .34 < extents["broken_vase"][2] < .36  # Not independently enlarged.

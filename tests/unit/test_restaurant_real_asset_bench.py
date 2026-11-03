"""Real prop geometry remains rendering/evaluation-only and original assets stay intact."""

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics
import pytest

from robot.scene.restaurant_container_bench import CONTAINER_PATH
from robot.scene.restaurant_demo_build import DEMO_ROOT, author_demo_props
from robot.scene.restaurant_real_asset_bench import KINDS, author_real_asset
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
    assert abs(lo[2]) < 1e-6
    np.testing.assert_allclose(((lo + hi) / 2)[:2], [28.15, 6.1], atol=1e-6)
    assert .2 < hi[2] < .7
    meshes = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(CONTAINER_PATH)):
        assert not prim.HasAPI(UsdPhysics.MassAPI)
        assert not prim.HasAttribute("phi")
        assert not prim.HasAttribute("semantic:class")
        if prim.IsA(UsdGeom.Mesh):
            meshes.append(prim)
            assert prim.HasAPI(UsdPhysics.CollisionAPI)
    assert len(meshes) == 1
    with pytest.raises(ValueError, match="original bin"):
        author_real_asset(stage, kind)

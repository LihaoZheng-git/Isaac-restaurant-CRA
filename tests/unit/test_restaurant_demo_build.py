"""Scene boundary checks: collision/placement never become physical evidence."""

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics
import pytest

from robot.scene.restaurant_demo_build import (
    CHAIR_PATH, DEMO_ROOT, DEMO_OBSERVATIONS, author_demo_props, demo_prop_records,
    author_demo_payload, payload_geometry_check, PAYLOAD_PATH,
)


def _stage():
    stage = Usd.Stage.CreateInMemory()
    chair = UsdGeom.Xform.Define(stage, CHAIR_PATH)
    chair.AddTranslateOp().Set(Gf.Vec3d(31, 9.5, .4))
    cube = UsdGeom.Cube.Define(stage, CHAIR_PATH+"/Seat")
    cube.CreateSizeAttr(.4)
    return stage


def test_demo_layer_keeps_original_scene_and_all_collision_without_score_labels():
    stage = _stage()
    before = stage.GetRootLayer().ExportToString()
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        records = author_demo_props(stage)
    assert stage.GetRootLayer().ExportToString() == before
    assert len(records) == 5 and records == demo_prop_records(stage)
    assert all(not r["available_as_physical_factor_input"] for r in records)
    chair = next(r for r in records if r["name"] == "PulledChair")
    assert np.mean([chair["bounds_minimum"][0], chair["bounds_maximum"][0]]) == pytest.approx(30.85)
    for prim in Usd.PrimRange(stage.GetPrimAtPath(DEMO_ROOT)):
        if prim.IsA(UsdGeom.Gprim):
            assert prim.HasAPI(UsdPhysics.CollisionAPI)
        assert not prim.HasAPI(UsdPhysics.MassAPI)
        assert not prim.HasAttribute("phi")
        assert not prim.HasAttribute("semantic:class")
    with pytest.raises(ValueError, match="already exists"):
        author_demo_props(stage)


def test_demo_observations_are_evaluation_views_and_legacy_scene_is_empty():
    assert demo_prop_records(_stage()) == []
    assert len(DEMO_OBSERVATIONS) == 3
    stage = _stage()
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        records = author_demo_props(stage)
    caddies = [r for r in records if r["name"] in {"ClosedCaddy", "UtensilCaddy"}]
    assert all(r["bounds_minimum"][2] == pytest.approx(.8, abs=1e-5) for r in caddies)
    assert all(r["bounds_maximum"][2] < 1.1 for r in caddies)


def test_mounted_tray_stays_inside_original_footprint_without_new_rigid_body():
    stage = _stage()
    parent = UsdGeom.Xform.Define(stage, "/World/Robot/chassis_link")
    parent.AddTranslateOp().Set(Gf.Vec3d(4, 5, .1))
    original = stage.GetRootLayer().ExportToString()
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        check = author_demo_payload(stage)
    assert stage.GetRootLayer().ExportToString() == original
    assert check["within_original_planar_footprint"] and check["present"]
    assert payload_geometry_check(stage) == check
    for prim in Usd.PrimRange(stage.GetPrimAtPath(PAYLOAD_PATH)):
        assert not prim.HasAPI(UsdPhysics.RigidBodyAPI)
        assert not prim.HasAPI(UsdPhysics.MassAPI)
    with pytest.raises(ValueError, match="already exists"):
        author_demo_payload(stage)

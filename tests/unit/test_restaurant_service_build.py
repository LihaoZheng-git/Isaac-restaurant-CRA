import hashlib
import math

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics
import pytest

from robot.scene.restaurant_service_build import (
    ASSET_ROOT,
    LONG_ROUTE_WORLD_XY,
    SERVICE_ROOT,
    ServiceProp,
    audit_source_assets,
    author_service_props,
    build_service_scene,
    service_props,
)


def test_sources_match_cc0_intake_and_meshes_keep_uvs_and_cavities():
    manifest = audit_source_assets()
    assert len(manifest["selected_models"]) == 13
    for model in {prop.model for prop in service_props()}:
        stage = Usd.Stage.Open(str(ASSET_ROOT / f"usd/{model}.usda"))
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/Asset/Geometry"))
        points = np.asarray(mesh.GetPointsAttr().Get())
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        uv = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st").Get()
        assert len(uv) == len(indices)
        assert min(indices) >= 0 and max(indices) < len(points)
        assert points[:, 2].min() == pytest.approx(0, abs=1e-6)
        assert UsdGeom.GetStageUpAxis(stage) == "Z"
        assert UsdGeom.GetStageMetersPerUnit(stage) == 1
        assert UsdPhysics.MeshCollisionAPI(mesh).GetApproximationAttr().Get() == "none"
        assert not mesh.GetPrim().HasAPI(UsdPhysics.MassAPI)


def test_session_authoring_preserves_source_and_places_actual_bases():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Cube.Define(stage, "/World/ExistingTable")
    before = stage.GetRootLayer().ExportToString()
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        records = author_service_props(stage)
    assert stage.GetRootLayer().ExportToString() == before
    assert stage.GetPrimAtPath("/World/ExistingTable").IsValid()
    assert len(records) == 11
    for record, spec in zip(records, service_props(), strict=True):
        low, high = np.array(record["bounds_minimum"]), np.array(record["bounds_maximum"])
        assert low[2] == pytest.approx(spec.base_world_xyz_m[2], abs=1e-6)
        assert ((low + high) / 2)[:2] == pytest.approx(spec.base_world_xyz_m[:2], abs=1e-5)
        assert not record["available_as_physical_factor_input"]
        assert record["provenance"] == "authored_placement_and_rendering_only"
    text = stage.GetSessionLayer().ExportToString()
    assert "physics:mass" not in text
    assert "phi" not in text
    assert "semantic" not in text
    with pytest.raises(ValueError, match="already exists"):
        author_service_props(stage)


def test_saved_additions_reopen_and_base_cannot_be_overwritten(tmp_path):
    base = tmp_path / "base.usda"
    stage = Usd.Stage.CreateNew(str(base))
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(1000000)
    stage.SetTimeCodesPerSecond(60)
    UsdGeom.Cube.Define(stage, "/World/Existing")
    stage.GetRootLayer().Save()
    original = hashlib.sha256(base.read_bytes()).hexdigest()
    result = build_service_scene(tmp_path / "service.usda", base_scene=base)
    assert result["base_unchanged"] and not result["humans_added"]
    assert hashlib.sha256(base.read_bytes()).hexdigest() == original
    reopened = Usd.Stage.Open(result["scene"])
    assert reopened.GetPrimAtPath(f"{SERVICE_ROOT}/FoodBowl/Geometry").IsValid()
    assert reopened.GetPrimAtPath("/World/Existing").IsValid()
    assert UsdGeom.GetStageMetersPerUnit(reopened) == 1
    assert reopened.GetTimeCodesPerSecond() == 60
    assert reopened.GetEndTimeCode() == 1000000
    with pytest.raises(ValueError, match="new scene path"):
        build_service_scene(base, base_scene=base)
    with pytest.raises(ValueError, match="new scene path"):
        build_service_scene(tmp_path / "service.usda", base_scene=base)


def test_zero_length_source_timeline_is_rejected_before_creating_output(tmp_path):
    base=tmp_path/"bad-time.usda"
    source=Usd.Stage.CreateNew(str(base))
    source.GetRootLayer().Save()
    output=tmp_path/"service.usda"
    with pytest.raises(ValueError,match="nonempty simulation time range"):
        build_service_scene(output,base_scene=base)
    assert not output.exists()


@pytest.mark.parametrize("change", [dict(width_m=0), dict(width_m=math.nan), dict(model="../outside"), dict(name="not a path"), dict(base_world_xyz_m=(1, 2, math.inf))])
def test_bad_placement_rejected(change):
    fields = dict(name="Test", model="bowl", base_world_xyz_m=(0, 0, 0), width_m=.3)
    fields.update(change)
    with pytest.raises(ValueError):
        ServiceProp(**fields)


def test_left_start_route_remains_a_proposal_and_exceeds_thirty_metres():
    route = np.array(LONG_ROUTE_WORLD_XY)
    assert route[0, 0] < 8
    assert np.linalg.norm(np.diff(route, axis=0), axis=1).sum() > 30

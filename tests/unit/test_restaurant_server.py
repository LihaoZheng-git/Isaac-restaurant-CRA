import hashlib
import json

import numpy as np
import pytest
from pxr import Usd, UsdGeom, UsdPhysics

from robot.scene.restaurant_server import (
    SERVER_ASSET_ROOT,
    SERVER_PRIM_PATH,
    audit_server_sources,
    bake_server_asset,
    build_server_scene,
)


def test_audited_original_server_files_and_static_uv_geometry():
    manifest = audit_server_sources()
    assert len(manifest["files"]) == 4 and manifest["license"] == "CC0-1.0"
    stage = Usd.Stage.Open(str(SERVER_ASSET_ROOT / "usd/server.usda"))
    conversion = json.loads((SERVER_ASSET_ROOT / "usd/server.json").read_text())
    assert (
        hashlib.sha256((SERVER_ASSET_ROOT / "usd/server.usda").read_bytes()).hexdigest()
        == conversion["asset_sha256"]
    )
    assert (
        UsdGeom.GetStageMetersPerUnit(stage) == 1
        and UsdGeom.GetStageUpAxis(stage) == "Z"
    )
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/Server/Body"))
    points = np.array(mesh.GetPointsAttr().Get())
    assert np.isfinite(points).all() and len(points) == 804
    assert points[:, 2].min() == pytest.approx(0)
    assert points[:, 2].max() == pytest.approx(1.75)
    assert (
        np.ptp(points[:, 0]) < 0.7
    )  # Frozen idle arms down, not bind-pose outstretched arms.
    assert mesh.GetPrim().HasAPI(UsdPhysics.CollisionAPI)
    uv = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st")
    assert len(uv.GetIndices()) == sum(mesh.GetFaceVertexCountsAttr().Get())
    assert np.isfinite(np.array(uv.Get())).all()
    assert stage.GetPrimAtPath("/Server/ApronBib").IsValid()
    assert all(not prim.HasAPI(UsdPhysics.MassAPI) for prim in stage.Traverse())
    assert all(
        prim.GetTypeName() not in {"Skeleton", "SkelAnimation"}
        for prim in stage.Traverse()
    )
    assert all(
        not any(
            term in str(attr.GetName()).lower()
            for term in ("sharpness", "temperature", "cra_score")
        )
        for prim in stage.Traverse()
        for attr in prim.GetAttributes()
        if attr.HasAuthoredValueOpinion()
    )


def _base(path, *, units=1, axis="Z"):
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, units)
    UsdGeom.SetStageUpAxis(stage, axis)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(1000)
    stage.SetTimeCodesPerSecond(60)
    UsdGeom.Cube.Define(stage, "/World/ExistingTable")
    stage.GetRootLayer().Save()


def test_server_layer_preserves_existing_scene_and_rejects_overwrite(tmp_path):
    base = tmp_path / "base.usda"
    _base(base)
    before = hashlib.sha256(base.read_bytes()).hexdigest()
    output = tmp_path / "server-scene.usda"
    report = build_server_scene(output, base)
    assert report["base_unchanged"] and report["actor_motion_enabled"] is False
    assert report["factor_inputs_supplied"] is False
    assert hashlib.sha256(base.read_bytes()).hexdigest() == before
    stage = Usd.Stage.Open(str(output))
    assert stage.GetPrimAtPath("/World/ExistingTable").IsValid()
    assert stage.GetPrimAtPath(SERVER_PRIM_PATH).IsValid()
    assert stage.GetTimeCodesPerSecond() == 60 and stage.GetEndTimeCode() == 1000
    assert report["bounds_minimum"][2] == pytest.approx(0)
    with pytest.raises(ValueError):
        build_server_scene(output, base)
    with pytest.raises(ValueError):
        build_server_scene(tmp_path / "duplicate.usda", output)
    assert not (tmp_path / "duplicate.usda").exists()


@pytest.mark.parametrize("units,axis", [(0.01, "Z"), (1, "Y")])
def test_server_build_rejects_incompatible_scene_contract(tmp_path, units, axis):
    base = tmp_path / "bad-source.usda"
    _base(base, units=units, axis=axis)
    with pytest.raises(ValueError, match="metre/Z-up"):
        build_server_scene(tmp_path / "new.usda", base)
    assert not (tmp_path / "new.usda").exists()


def test_server_source_integrity_and_existing_asset_protection(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "license": "CC0-1.0",
                "files": [{"local_path": "../outside", "sha256": "invalid"}],
            }
        )
    )
    with pytest.raises(ValueError):
        audit_server_sources(tmp_path)
    with pytest.raises(ValueError):
        bake_server_asset("unused", "unused", SERVER_ASSET_ROOT / "usd/server.usda")

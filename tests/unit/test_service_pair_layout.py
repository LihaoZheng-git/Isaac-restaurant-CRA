"""Regressions for additive V3 scene placement and recipe preservation."""
import importlib.util
import json
from pathlib import Path

import pytest

BUILDER_PATH = Path(__file__).resolve().parents[2] / "scripts/scene_creation/build_service_pair_layout.py"
spec = importlib.util.spec_from_file_location("service_pair_builder", BUILDER_PATH)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

from pxr import Gf, Sdf, Usd, UsdGeom  # noqa: E402
from robot.scene.restaurant_demo_build import CHAIR_PATH, DEMO_NAMES, DEMO_ROOT  # noqa: E402


@pytest.fixture
def source_root(tmp_path):
    generated = tmp_path / "scenes/generated"
    generated.mkdir(parents=True)
    (tmp_path / "configs").mkdir()
    baseline = builder.ROOT / "configs/restaurant_service_demo_v3_balanced_contact.json"
    (tmp_path / "configs" / baseline.name).write_bytes(baseline.read_bytes())
    asset = generated / "licensed-asset.txt"
    asset.write_text("unchanged licensed source")
    for layout, bin_y, crate_y in (("balanced", 3.4, 3.4), ("open_aisle", 3.2, 3.0)):
        for suffix in ("contact", "bench"):
            path = generated / f"restaurant_service_demo_v3_{layout}_{suffix}.usda"
            additions = path.with_name(path.stem + "_props.usda")
            patch = Sdf.Layer.CreateNew(str(additions))
            stage = Usd.Stage.CreateNew(str(path))
            stage.GetRootLayer().subLayerPaths = [additions.name]
            stage.SetStartTimeCode(0)
            stage.SetEndTimeCode(1000000)
            stage.SetTimeCodesPerSecond(60)
            UsdGeom.SetStageMetersPerUnit(stage, 1)
            UsdGeom.SetStageUpAxis(stage, "Z")
            with Usd.EditContext(stage, patch):
                for name in DEMO_NAMES:
                    stage.OverridePrim(DEMO_ROOT + "/" + name).SetActive(False)
                for name, x, y in (("ServiceContainer", 21.2, bin_y), ("BrokenCrate", 17.5, crate_y)):
                    prop = UsdGeom.Xform.Define(stage, DEMO_ROOT + "/" + name)
                    prop.AddTranslateOp().Set(Gf.Vec3d(x, y, .4))
                    UsdGeom.Cube.Define(stage, prop.GetPath().AppendChild("Mesh")).CreateSizeAttr(.3)
                UsdGeom.Cube.Define(stage, CHAIR_PATH).AddTranslateOp().Set(Gf.Vec3d(31, 9, 0))
                server = UsdGeom.Xform.Define(stage, "/World/CRAServiceActors/Server")
                server.AddTranslateOp().Set(Gf.Vec3d(28, 23.5, 0))
                server.GetPrim().SetActive(suffix == "contact")
            patch.Save()
            stage.GetRootLayer().Save()
            builder.write_json(path.with_suffix(".json"), dict(
                scene=str(path), additions=str(additions),
                source_sha256={str(asset): builder.digest(asset)},
                props=builder.demo_prop_records(stage),
                stationary_props_only=suffix == "bench",
                contact_prop_centers_world_xy=[[21.2, bin_y], [17.5, crate_y]],
            ))
    return tmp_path


@pytest.mark.parametrize("layout,parked_y", [("original-v3", 4.05), ("relief20", 3.85)])
def test_overlays_preserve_sources_timing_actor_and_runtime_settings(source_root, layout, parked_y):
    before = {p: builder.digest(p) for p in source_root.rglob("*") if p.is_file()}
    output = source_root / "scenes/generated/repeat"
    recipe = source_root / "configs/repeat.json"
    result = builder.build(layout, output, recipe, root=source_root)
    assert result["expected_bounds_verified"] and result["parent_sources_unchanged"]
    assert all(builder.digest(p) == sha for p, sha in before.items())
    for name in result["scenes"]:
        stage = Usd.Stage.Open(name)
        manifest = json.loads(Path(name).with_suffix(".json").read_text())
        assert stage.GetStartTimeCode() == 0
        assert stage.GetEndTimeCode() == 1000000
        assert stage.GetTimeCodesPerSecond() == 60
        assert UsdGeom.GetStageMetersPerUnit(stage) == 1
        assert UsdGeom.GetStageUpAxis(stage) == "Z"
        assert stage.GetPrimAtPath("/World/CRAServiceActors/Server").IsActive() == ("_contact" in name)
        assert all(builder.digest(p) == sha for p, sha in manifest["source_sha256"].items())
        parent = Path(manifest["layout_parent"])
        assert str(parent) in manifest["source_sha256"]
        assert str(parent.with_suffix(".json")) in manifest["source_sha256"]
    previous = json.loads((source_root / "configs/restaurant_service_demo_v3_balanced_contact.json").read_text())["commands"]
    current = json.loads(recipe.read_text())["commands"]
    for case, cmd in current.items():
        old = previous[case]
        scene = cmd.index("--scene" if "--scene" in cmd else "--restaurant-scene") + 1
        assert Path(cmd[scene]).name == Path(old[scene]).name
        cmd[scene] = old[scene]
        if case == "crate-static":
            y = cmd.index("--stationary-prop-world-pose") + 2
            assert float(cmd[y]) == pytest.approx(parked_y)
            cmd[y] = old[y]
        assert cmd == old
    saved = {p: builder.digest(p) for p in output.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="fresh"):
        builder.build(layout, output, recipe, root=source_root)
    assert all(builder.digest(p) == sha for p, sha in saved.items())


@pytest.mark.parametrize("damage", ["source", "bounds", "centers"])
def test_corrupt_parent_is_rejected_before_creating_outputs(source_root, damage):
    parent = source_root / "scenes/generated/restaurant_service_demo_v3_balanced_contact.json"
    if damage == "source":
        (parent.parent / "licensed-asset.txt").write_text("modified source")
    else:
        data = json.loads(parent.read_text())
        if damage == "bounds":
            data["props"][0]["bounds_minimum"][0] += .1
        else:
            data["contact_prop_centers_world_xy"][0][1] += .1
        parent.write_text(json.dumps(data))
    output = source_root / "scenes/generated/rejected"
    recipe = source_root / "configs/rejected.json"
    with pytest.raises(ValueError):
        builder.build("relief20", output, recipe, root=source_root)
    assert not output.exists() and not recipe.exists()

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


@pytest.fixture
def closer_source_root(source_root):
    parent = builder.ROOT / "configs/restaurant_service_risk_calibration_witnesses.json"
    (source_root / "configs" / parent.name).write_bytes(parent.read_bytes())
    builder.build("original-v3", "scenes/generated/service_pair_original_v3",
                  "configs/original-v3.json", root=source_root)
    return source_root


@pytest.mark.parametrize("layout,centers,deltas", [
    ("v6-closer", [[21.2, 3.65], [17.5, 3.45]], {"ServiceContainer": .45, "BrokenCrate": .45}),
    ("v6-clearance", [[21.2, 3.4], [17.5, 3.1]], {"ServiceContainer": .2, "BrokenCrate": .1}),
])
def test_closer_pair_only_translates_two_props_and_preserves_calibrated_commands(closer_source_root, layout, centers, deltas):
    root = closer_source_root
    before = {p: builder.digest(p) for p in root.rglob("*") if p.is_file()}
    output = root / "scenes/generated" / ("service_pair_" + layout.replace("-", "_"))
    recipe_path = root / "configs/closer.json"
    result = builder.build(layout, output, recipe_path, root=root)
    assert result["parent_sources_unchanged"] and result["expected_bounds_verified"]
    assert all(builder.digest(p) == sha for p, sha in before.items())
    for name in result["scenes"]:
        stage = Usd.Stage.Open(name)
        manifest = json.loads(Path(name).with_suffix(".json").read_text())
        assert manifest["contact_prop_centers_world_xy"] == centers
        assert manifest["bin_translation_delta_world_y_m"] == pytest.approx(deltas["ServiceContainer"])
        assert manifest["crate_translation_delta_world_y_m"] == pytest.approx(deltas["BrokenCrate"])
        parent = Path(manifest["layout_parent"])
        assert parent.parent.name == "service_pair_original_v3"
        source = Usd.Stage.Open(str(parent))
        assert stage.GetStartTimeCode() == source.GetStartTimeCode()
        assert stage.GetEndTimeCode() == source.GetEndTimeCode()
        assert stage.GetTimeCodesPerSecond() == source.GetTimeCodesPerSecond()
        assert UsdGeom.GetStageMetersPerUnit(stage) == UsdGeom.GetStageMetersPerUnit(source)
        assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.GetStageUpAxis(source)
        assert stage.GetPrimAtPath("/World/CRAServiceActors/Server").IsActive() == ("_contact" in name)
        original_props = {p["name"]: p for p in builder.demo_prop_records(source)}
        for prop in builder.demo_prop_records(stage):
            original = original_props[prop["name"]]
            delta = deltas.get(prop["name"], 0.)
            for key in ("bounds_minimum", "bounds_maximum"):
                assert prop[key] == pytest.approx([original[key][0], original[key][1] + delta,
                                                  original[key][2]])
            assert {k: v for k, v in prop.items() if k not in ("bounds_minimum", "bounds_maximum")} == {
                k: v for k, v in original.items() if k not in ("bounds_minimum", "bounds_maximum")}
        layer = Sdf.Layer.FindOrOpen(manifest["additions"])
        assert list(layer.GetPrimAtPath(DEMO_ROOT + "/ServiceContainer").properties.keys()) == ["xformOp:translate"]
        assert list(layer.GetPrimAtPath(DEMO_ROOT + "/BrokenCrate").properties.keys()) == ["xformOp:translate"]
    parent_path = root / "configs/restaurant_service_risk_calibration_witnesses.json"
    previous = json.loads(parent_path.read_text())
    current = json.loads(recipe_path.read_text())
    committed = json.loads((builder.ROOT / "configs" / ("restaurant_service_" + layout.replace("-", "_") + "_calibrated.json")).read_text())
    assert current == committed
    assert current["layout_derivation"]["parent_recipe_sha256"] == builder.digest(parent_path)
    for case, cmd in current["commands"].items():
        old = previous["commands"][case]
        index = cmd.index("--scene" if "--scene" in cmd else "--restaurant-scene") + 1
        assert Path(cmd[index]).parent == output.relative_to(root)
        assert Path(cmd[index]).name == Path(old[index]).name
        cmd[index] = old[index]
        assert cmd == old
    frozen = {p: builder.digest(p) for p in output.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="fresh"):
        builder.build(layout, output, recipe_path, root=root)
    assert all(builder.digest(p) == sha for p, sha in frozen.items())


@pytest.mark.parametrize("damage", ["source", "bounds", "centers"])
@pytest.mark.parametrize("layout", ["v6-closer", "v6-clearance"])
def test_closer_pair_corrupt_accepted_parent_rejected_before_outputs(closer_source_root, damage, layout):
    root = closer_source_root
    parent = root / ("scenes/generated/service_pair_original_v3/"
                     "restaurant_service_demo_v3_balanced_contact.json")
    if damage == "source":
        (root / "scenes/generated/licensed-asset.txt").write_text("modified source")
    else:
        data = json.loads(parent.read_text())
        if damage == "bounds":
            data["props"][0]["bounds_minimum"][0] += .1
        else:
            data["contact_prop_centers_world_xy"][0][1] += .1
        parent.write_text(json.dumps(data))
    output = root / "scenes/generated/rejected-closer"
    recipe = root / "configs/rejected-closer.json"
    with pytest.raises(ValueError):
        builder.build(layout, output, recipe, root=root)
    assert not output.exists() and not recipe.exists()


@pytest.mark.parametrize("layout,centers,scope", [
    ("v6-closer", [[21.2, 3.65], [17.5, 3.45]], builder.V6_SCOPE),
    ("v6-clearance", [[21.2, 3.4], [17.5, 3.1]], builder.V6_CLEARANCE_SCOPE),
])
def test_committed_closer_recipe_preserves_all_calibrated_parent_commands(layout, centers, scope):
    parent = builder.ROOT / "configs/restaurant_service_risk_calibration_witnesses.json"
    previous = json.loads(parent.read_text())
    current = json.loads((builder.ROOT / "configs" / ("restaurant_service_" + layout.replace("-", "_") + "_calibrated.json")).read_text())
    assert current["layout_derivation"]["parent_recipe_sha256"] == builder.digest(parent)
    assert current["layout_derivation"]["native_acceptance_pending"] is True
    assert current["layout_derivation"]["contact_prop_centers_world_xy"] == centers
    assert current["original_invocations"] == previous["original_invocations"]
    assert current["proof_variant"] == previous["proof_variant"]
    assert current["scope"] == scope
    assert current["source_revision_role"] == builder.V6_SOURCE_REVISION_ROLE
    assert current["layout_derivation"]["parent_source_revision"] == previous["source_revision"]
    assert current["layout_derivation"]["parent_scope"] == previous["scope"]
    assert current["commands"].keys() == previous["commands"].keys()
    for case, command in current["commands"].items():
        index = command.index("--scene" if "--scene" in command else "--restaurant-scene") + 1
        assert Path(command[index]).parent.as_posix() == "scenes/generated/service_pair_" + layout.replace("-", "_")
        assert Path(command[index]).name == Path(previous["commands"][case][index]).name
        command[index] = previous["commands"][case][index]
        assert command == previous["commands"][case]


@pytest.mark.parametrize("case", ["long-parallel", "long-parallel-near", "long-parallel-component",
                                  "long-idle-component", "bin-static", "crate-static"])
@pytest.mark.parametrize("layout", ["v6-closer", "v6-clearance"])
def test_closer_recipe_cli_parses_without_starting_isaac(monkeypatch, case, layout):
    recipe = json.loads((builder.ROOT / "configs" / ("restaurant_service_" + layout.replace("-", "_") + "_calibrated.json")).read_text())
    command = recipe["commands"][case] + ["--output", "outputs/unused-v6-cli-check"]
    script = builder.ROOT / command[0]
    monkeypatch.setattr("sys.argv", command)
    exec(compile(script.read_text().split("args.output.mkdir", 1)[0], str(script), "exec"),
         {"__file__": str(script)})

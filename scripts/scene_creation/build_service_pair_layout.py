"""Reproduce a V3 service placement with fresh USD overlays and a launch recipe.

Use --layout original-v3 for the accepted open-aisle coordinates, or relief20
for the balanced placement with the crate shifted 20 cm farther from the route.
The v6-closer placement uses the calibrated witness recipe and moves only
the two props in additive layers over the accepted original-v3 scene.
The original generated layers and licensed assets must already be available.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
dll = Path(sys.executable).parent / "Lib/site-packages/usd_exchange.libs"
if os.name == "nt" and dll.is_dir():
    _dll_cookie = os.add_dll_directory(str(dll))

import numpy as np  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom  # noqa: E402

from robot.scene.restaurant_demo_build import DEMO_ROOT, demo_prop_records  # noqa: E402

V6_SCOPE = (
    "Separate V6 closer-prop placement using the calibrated witness method. "
    "Move only the bin to (21.20, 3.65) and crate to (17.50, 3.45), each +0.45 m world Y "
    "from the accepted original-V3 layout. Object meshes, scales, human timing, witness "
    "bounds, ownership/evidence policies, factor scoring, palette, fixed lambda1.25, "
    "obstacle-unaware nominal, full N6/guidance100 controller and physical limits remain "
    "unchanged. Fresh six prerequisites, own60 and full native arrival/stopping/source "
    "audits are required. Historical source_revision/original_invocations identify "
    "parent provenance only; the launcher records the new frozen native revision. "
    "Preserve previous scenes, demos and failures. Not a risk-only causal comparison."
)
V6_SOURCE_REVISION_ROLE = "historical parent-runtime snapshot; actual native revision is invocation.json.source_revision"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")


def checked_source(path):
    """Validate the original layer and dependencies before authoring an overlay."""
    manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    sources = dict(manifest["source_sha256"])
    for source in (path, path.with_suffix(".json"), Path(manifest["additions"])):
        sources[str(source.resolve())] = digest(source)
    if any(digest(p) != expected for p, expected in sources.items()):
        raise ValueError(f"source hash mismatch for {path}")
    stage = Usd.Stage.Open(str(path))
    props = json.loads(json.dumps(demo_prop_records(stage)))
    if props != manifest["props"]:
        raise ValueError(f"composed source bounds differ from manifest for {path}")
    return stage, manifest, sources, props


def build(layout, output_directory, recipe_path, *, root=ROOT):
    root = Path(root).resolve()
    output_directory = (root / output_directory).resolve()
    recipe_path = (root / recipe_path).resolve()
    if not output_directory.is_relative_to(root) or not recipe_path.is_relative_to(root):
        raise ValueError("scene directory and recipe must be inside the repository")
    if output_directory.exists() or recipe_path.exists():
        raise ValueError("fresh scene directory and recipe paths required")
    if recipe_path.suffix != ".json":
        raise ValueError("recipe must have a .json suffix")
    if layout not in ("original-v3", "relief20", "v6-closer"):
        raise ValueError("choose original-v3, relief20 or v6-closer")
    original_layout = layout == "original-v3"
    closer_layout = layout == "v6-closer"
    parent_name = "open_aisle" if original_layout else "balanced"
    centers = [[21.2, 3.2], [17.5, 3.0]] if original_layout else [[21.2, 3.4], [17.5, 3.2]]
    crate_delta = 0.0 if original_layout else -0.2
    bin_delta = 0.0
    base_recipe_path = root / "configs/restaurant_service_demo_v3_balanced_contact.json"
    if closer_layout:
        centers = [[21.2, 3.65], [17.5, 3.45]]
        bin_delta = centers[0][1] - 3.2
        crate_delta = centers[1][1] - 3.0
        base_recipe_path = root / "configs/restaurant_service_risk_calibration_witnesses.json"
    deltas = {"ServiceContainer": bin_delta, "BrokenCrate": crate_delta}
    base_recipe_hash = digest(base_recipe_path)
    recipe = json.loads(base_recipe_path.read_text(encoding="utf-8"))
    commands = copy.deepcopy(recipe["commands"])
    sources = {}
    for suffix in ("contact", "bench"):
        parent = root / f"scenes/generated/restaurant_service_demo_v3_{parent_name}_{suffix}.usda"
        if closer_layout:
            parent = root / ("scenes/generated/service_pair_original_v3/"
                             f"restaurant_service_demo_v3_balanced_{suffix}.usda")
        sources[suffix] = (parent, *checked_source(parent))
    expected_parent_centers = copy.deepcopy(centers)
    expected_parent_centers[0][1] -= bin_delta
    expected_parent_centers[1][1] -= crate_delta
    for parent, stage, manifest, _, props in sources.values():
        if not np.allclose(manifest["contact_prop_centers_world_xy"], expected_parent_centers, atol=1e-9, rtol=0):
            raise ValueError(f"unexpected parent placement in {parent}")
        observed_centers = []
        for name in ("ServiceContainer", "BrokenCrate"):
            item = next(p for p in props if p["name"] == name)
            observed_centers.append(((np.array(item["bounds_minimum"]) + item["bounds_maximum"]) / 2)[:2])
        if not np.allclose(observed_centers, expected_parent_centers, atol=1e-9, rtol=0):
            raise ValueError(f"unexpected composed prop centers in {parent}")
        for name, delta in deltas.items():
            if delta and not stage.GetPrimAtPath(DEMO_ROOT + "/" + name).GetAttribute("xformOp:translate"):
                raise ValueError(f"{name} needs the original translation operation")
    # Validate the new recipe completely before creating files.
    for case, command in commands.items():
        flag = "--scene" if "--scene" in command else "--restaurant-scene"
        index = command.index(flag) + 1
        name = Path(command[index]).name
        if name not in ("restaurant_service_demo_v3_balanced_contact.usda",
                        "restaurant_service_demo_v3_balanced_bench.usda"):
            raise ValueError(f"unexpected scene in {case}")
        command[index] = (output_directory / name).relative_to(root).as_posix()
        if case == "crate-static" and layout == "relief20":
            parked_y = command.index("--stationary-prop-world-pose") + 2
            command[parked_y] = format(float(command[parked_y]) + crate_delta, ".12g")
    output_directory.mkdir(parents=True)
    results = []
    for suffix, (parent, base, manifest, hashes, before) in sources.items():
        output = output_directory / f"restaurant_service_demo_v3_balanced_{suffix}.usda"
        additions = output.with_name(output.stem + "_placement.usda")
        patch = Sdf.Layer.CreateNew(str(additions))
        stage = Usd.Stage.CreateNew(str(output))
        stage.GetRootLayer().subLayerPaths = [
            additions.name, os.path.relpath(parent, output_directory).replace("\\", "/")]
        stage.SetStartTimeCode(base.GetStartTimeCode())
        stage.SetEndTimeCode(base.GetEndTimeCode())
        stage.SetTimeCodesPerSecond(base.GetTimeCodesPerSecond())
        UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(base))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(base))
        with Usd.EditContext(stage, patch):
            for name, delta in deltas.items():
                if delta:
                    translation = stage.GetPrimAtPath(DEMO_ROOT + "/" + name).GetAttribute("xformOp:translate")
                    translation.Set(translation.Get() + Gf.Vec3d(0, delta, 0))
        patch.Save()
        stage.GetRootLayer().Save()
        after = json.loads(json.dumps(demo_prop_records(Usd.Stage.Open(str(output)))))
        expected = copy.deepcopy(before)
        for prop in expected:
            if prop["name"] in deltas:
                for key in ("bounds_minimum", "bounds_maximum"):
                    prop[key][1] += deltas[prop["name"]]
        if len(after) != len(expected):
            raise ValueError("authored prop count changed")
        for actual, wanted in zip(after, expected):
            for key in ("bounds_minimum", "bounds_maximum"):
                if not np.allclose(actual[key], wanted[key], atol=1e-9, rtol=0):
                    raise ValueError(f"unexpected composed {key} for {actual['name']}")
            if {k: v for k, v in actual.items() if k not in ("bounds_minimum", "bounds_maximum")} != {
                    k: v for k, v in wanted.items() if k not in ("bounds_minimum", "bounds_maximum")}:
                raise ValueError("authored prop metadata changed")
        if any(digest(p) != expected for p, expected in hashes.items()):
            raise ValueError("original source changed during authoring")
        result = dict(manifest, scene=str(output), additions=str(additions), source_sha256=hashes,
                      props=after, layout_parent=str(parent), derived_service_layout=layout,
                      contact_prop_centers_world_xy=centers,
                      crate_translation_delta_world_y_m=crate_delta)
        if closer_layout:
            result["bin_translation_delta_world_y_m"] = bin_delta
        results.append(result)
    if digest(base_recipe_path) != base_recipe_hash:
        raise ValueError("base recipe changed during authoring")
    for _, _, _, hashes, _ in sources.values():
        if any(digest(p) != expected for p, expected in hashes.items()):
            raise ValueError("parent source changed before manifest publication")
    for result in results:
        write_json(Path(result["scene"]).with_suffix(".json"), result)
    recipe["commands"] = commands
    if closer_layout:
        recipe["layout_derivation"] = dict(
            parent_recipe=base_recipe_path.relative_to(root).as_posix(),
            parent_recipe_sha256=base_recipe_hash,
            layout=layout, contact_prop_centers_world_xy=centers,
            command_changes="scene paths only; original parked poses and full N6/guidance100 retained",
            native_acceptance_pending=True,
            parent_scope=recipe["scope"], parent_source_revision=recipe["source_revision"])
        recipe["scope"] = V6_SCOPE
        recipe["source_revision_role"] = V6_SOURCE_REVISION_ROLE
    else:
        recipe["scope"] += f" Reproduced {layout} placement. Fresh native gates and mission acceptance required."
    recipe_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(recipe_path, recipe)
    return dict(layout=layout, scenes=[r["scene"] for r in results], recipe=str(recipe_path),
                parent_sources_unchanged=True, expected_bounds_verified=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", choices=("original-v3", "relief20", "v6-closer"), required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.layout, args.output_directory, args.recipe), indent=2))

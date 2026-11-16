"""Licensed real-mesh props in separate stationary scenes; never factor inputs."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from robot.config import PROJECT_ROOT
from robot.scene.restaurant_container_bench import CONTAINER_CENTER_XY, CONTAINER_PATH
from robot.scene.restaurant_demo_build import DEMO_ROOT, _shape, demo_prop_records

ASSETS = PROJECT_ROOT / "assets/restaurant/sketchfab_contact_props"
VASE_VARIANTS = {
    "intact_vase": "Vase_ModelMain_0",
    "broken_vase": "VaseBroken2_ModelMain_2",
}
KINDS = ("recycle_bin", "intact_crate", "broken_crate", "cleaver", "broken_plate", *VASE_VARIANTS, "simple_trash_can")


def author_real_asset(stage, kind):
    if kind not in KINDS:
        raise ValueError("unknown real prop")
    old = stage.GetPrimAtPath(DEMO_ROOT + "/RoundedBin")
    if not old or not old.IsActive() or stage.GetPrimAtPath(CONTAINER_PATH):
        raise ValueError("requires original bin and unused target path")
    folder_name = ("crates" if kind in ("intact_crate", "broken_crate") else
                   "vases" if kind in VASE_VARIANTS else kind)
    folder = ASSETS / folder_name
    source = Usd.Stage.Open(str(folder / "converted.usdc"))
    if UsdGeom.GetStageUpAxis(source) != "Y" or UsdGeom.GetStageMetersPerUnit(source) != 1:
        raise ValueError("unexpected converter units/up axis")
    old.SetActive(False)
    server = stage.GetPrimAtPath("/World/CRAServiceActors/Server")
    if server:
        server.SetActive(False)
    root = UsdGeom.Xform.Define(stage, CONTAINER_PATH)
    shift = root.AddTranslateOp()
    root.AddRotateZOp().Set(-90.)
    orient = UsdGeom.Xform.Define(stage, CONTAINER_PATH + "/Orientation")
    orient.AddRotateXOp().Set(90.)  # glTF Y-up -> restaurant Z-up.
    scale_op = orient.AddScaleOp()
    reference_path = CONTAINER_PATH + "/Orientation/Asset"
    reference = stage.DefinePrim(reference_path)
    layer_path = Path(stage.GetEditTarget().GetLayer().realPath)
    asset_path = folder / "converted.usdc"
    relative = (os.path.relpath(asset_path, layer_path.parent).replace("\\", "/")
                if str(layer_path) != "." else str(asset_path))
    reference.GetReferences().AddReference(relative)
    if kind == "recycle_bin":
        # The original glTF includes an assembled low-poly mesh, its duplicate
        # component meshes and co-located high-poly baking meshes. Keep the
        # original textured assembled mesh only, without changing its shape.
        parent = stage.GetPrimAtPath(reference_path + "/root/GLTF_SceneRootNode")
        if len(parent.GetChildren()) != 7:
            raise ValueError("unexpected bin source hierarchy")
        for child in parent.GetChildren():
            child.SetActive(child.GetName() == "Plane_l_001_0")
        cache = UsdGeom.BBoxCache(0, ["default", "render"])
        bounds = cache.ComputeWorldBound(root.GetPrim()).ComputeAlignedRange()
        scale = .65 / (bounds.GetMax()[2] - bounds.GetMin()[2])
    elif kind == "simple_trash_can":
        # Preserve the complete downloaded body/lid mesh and its proportions.
        # 50 cm is an authored indoor-bin size, not a measured physical input.
        bounds = UsdGeom.BBoxCache(0, ["default", "render"]).ComputeWorldBound(root.GetPrim()).ComputeAlignedRange()
        scale = .50 / (bounds.GetMax()[2] - bounds.GetMin()[2])
    elif kind in ("intact_crate", "broken_crate"):
        parent = stage.GetPrimAtPath(reference_path + "/node_18c1021f1704bc6a4ddc61ad48bf678_fbx/RootNode")
        keep = "Crate_01_GEO" if kind == "intact_crate" else "Crate_02_GEO"
        if {p.GetName() for p in parent.GetChildren()} != {"Crate_01_GEO", "Crate_02_GEO"}:
            raise ValueError("unexpected crate source hierarchy")
        for child in parent.GetChildren():
            child.SetActive(child.GetName() == keep)
        scale = .65  # SAME authored uniform scale for the original crate pair.
    elif kind in VASE_VARIANTS:
        parent = stage.GetPrimAtPath(reference_path + "/root/GLTF_SceneRootNode")
        expected = {"Vase_ModelMain_0", "VaseBroken1_ModelMain_1",
                    "VaseBroken2_ModelMain_2", "VaseBroken3_ModelMain_3"}
        if {p.GetName() for p in parent.GetChildren()} != expected:
            raise ValueError("unexpected vase source hierarchy")
        # Derive ONE scale from the original intact asset for both versions.
        # Broken2 retains a substantial upright body and its original jagged
        # rim; selection is an authored fixture, never a contact-risk label.
        intact = parent.GetChild(VASE_VARIANTS["intact_vase"])
        bounds = UsdGeom.BBoxCache(0, ["default", "render"]).ComputeWorldBound(intact).ComputeAlignedRange()
        scale = .40 / (bounds.GetMax()[2] - bounds.GetMin()[2])
        for child in parent.GetChildren():
            child.SetActive(child.GetName() == VASE_VARIANTS[kind])
    else:
        # Original complete meshes, at ordinary utensil/dish scale. The cleaver
        # lies flat on a low preparation surface; the plate lies on the floor.
        # These transforms never supply detector boxes or physical evidence.
        bounds = UsdGeom.BBoxCache(0, ["default", "render"]).ComputeWorldBound(root.GetPrim()).ComputeAlignedRange()
        extent = np.array(bounds.GetMax()) - np.array(bounds.GetMin())
        scale = (.32 if kind == "cleaver" else .28) / max(extent[:2])
    scale_op.Set(Gf.Vec3f(scale))
    bounds = UsdGeom.BBoxCache(0, ["default", "render"]).ComputeWorldBound(root.GetPrim()).ComputeAlignedRange()
    lo, hi = np.array(bounds.GetMin()), np.array(bounds.GetMax())
    if not np.isfinite([lo, hi]).all() or np.any(hi <= lo):
        raise ValueError("invalid visible mesh bounds")
    center = (lo + hi) / 2
    base_height = .45 if kind == "cleaver" else 0.
    shift.Set(Gf.Vec3d(CONTAINER_CENTER_XY[0] - center[0],
                       CONTAINER_CENTER_XY[1] - center[1], base_height - lo[2]))
    if kind == "cleaver":
        support = DEMO_ROOT + "/AssetBenchSupport"
        UsdGeom.Xform.Define(stage, support)
        x, y = CONTAINER_CENTER_XY
        _shape(stage, support + "/Top", UsdGeom.Cube, (x, y, .425),
               (.275, .275, .025), (.40, .40, .40))
        for dx in (-.23, .23):
            for dy in (-.23, .23):
                _shape(stage, support + f"/Leg{'P' if dx > 0 else 'N'}{'P' if dy > 0 else 'N'}",
                       UsdGeom.Cube, (x + dx, y + dy, .2),
                       (.015, .015, .2), (.35, .35, .35))
    meshes = [p for p in Usd.PrimRange(root.GetPrim()) if p.IsA(UsdGeom.Mesh)]
    expected_meshes = (3 if kind in ("cleaver", "broken_plate") else
                       2 if kind in VASE_VARIANTS else 1)
    if len(meshes) != expected_meshes:
        raise ValueError("isolated prop must retain all selected original meshes")
    for prim in meshes:
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("none")
    return next(p for p in demo_prop_records(stage) if p["name"] == "ServiceContainer")


def build_real_asset_scene(output, *, base_scene, kind):
    output, base_scene = Path(output).resolve(), Path(base_scene).resolve()
    additions = output.with_name(output.stem + "_asset.usda")
    if output == base_scene or any(p.exists() for p in (output, additions, output.with_suffix(".json"))):
        raise ValueError("fresh scene/additions/manifest paths required")
    if kind not in KINDS:
        raise ValueError("unknown real prop")
    base = Usd.Stage.Open(str(base_scene))
    paths = {Path(p.realPath) for p in base.GetUsedLayers() if p.realPath and Path(p.realPath).is_file()}
    paths.update(p for p in ASSETS.rglob("*") if p.is_file())
    sources = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output))
    layer = Sdf.Layer.CreateNew(str(additions))
    stage.GetRootLayer().subLayerPaths = [additions.name, os.path.relpath(base_scene, output.parent).replace("\\", "/")]
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, "Z")
    stage.SetStartTimeCode(base.GetStartTimeCode())
    stage.SetEndTimeCode(base.GetEndTimeCode())
    stage.SetTimeCodesPerSecond(base.GetTimeCodesPerSecond())
    with Usd.EditContext(stage, layer):
        target = author_real_asset(stage, kind)
    layer.Save()
    stage.GetRootLayer().Save()
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != h for p, h in sources.items()):
        raise RuntimeError("source changed during separate scene build")
    result = dict(scene=str(output), additions=str(additions), source_sha256=sources,
                  target=target, props=demo_prop_records(stage), fixture_asset=kind,
                  source_files_unchanged=True, stationary_props_only=True,
                  scope="Original licensed mesh/texture stationary probe; no category, density, sharpness or factor labels; no navigation acceptance")
    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    return result

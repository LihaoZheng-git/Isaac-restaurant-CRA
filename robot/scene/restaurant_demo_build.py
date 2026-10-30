"""First service-demo dressing; authored geometry is never CRA evidence."""

import hashlib
import json
import os
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from robot.scene.controlled_semantic_obstacle import read_world_bounds
from robot.scene.restaurant_service_build import ASSET_ROOT, audit_source_assets


DEMO_ROOT = "/World/CRAServiceDemoV1"
PAYLOAD_PATH = "/World/Robot/chassis_link/DemoServingTray"
CHAIR_PATH = ("/World/Environment/Restaurant/assets_restaurant/Assets_Banquettes_Right/"
              "Geo_TableSet_14/Chair_Array")
DEMO_OBSERVATIONS = {
    "demo-bin": ("RoundedBin", (28.8, 5.9, .07382), 0.0),
    "demo-rounded": ("ClosedCaddy", (29.0, 5.25, .07382), 0.0),
    "demo-utensils": ("UtensilCaddy", (29.0, 6.3, .07382), -.40),
}
DEMO_NAMES = ("ClearingStation", "RoundedBin", "ClosedCaddy", "UtensilCaddy")


def _shape(stage, path, kind, position, scale, color, *, radius=None, height=None):
    shape = kind.Define(stage, path)
    if radius is not None:
        shape.CreateRadiusAttr(radius)
    if height is not None:
        shape.CreateHeightAttr(height)
        shape.CreateAxisAttr("Z")
    shape.AddTranslateOp().Set(Gf.Vec3d(*position))
    shape.AddScaleOp().Set(Gf.Vec3d(*scale))
    shape.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
    return shape


def demo_prop_records(stage):
    """Authored boxes for independent abort/visibility scoring, never perception."""
    if not stage.GetPrimAtPath(DEMO_ROOT):
        return []
    result = []
    for name, path in [(n, f"{DEMO_ROOT}/{n}") for n in DEMO_NAMES] + [("PulledChair", CHAIR_PATH)]:
        bounds = read_world_bounds(stage, path)
        result.append(dict(name=name, prim_path=path, bounds_minimum=bounds.minimum,
                           bounds_maximum=bounds.maximum,
                           available_as_physical_factor_input=False,
                           provenance="authored_rendering_and_independent_evaluation_only"))
    return result


def author_demo_props(stage):
    """Build plausible clearing station, rounded bin and visible fork caddy."""
    if stage.GetPrimAtPath(DEMO_ROOT):
        raise ValueError("demo dressing already exists")
    if not stage.GetPrimAtPath(CHAIR_PATH):
        raise ValueError("expected existing restaurant chair is missing")
    audit_source_assets()
    UsdGeom.Xform.Define(stage, DEMO_ROOT)
    station = f"{DEMO_ROOT}/ClearingStation"
    UsdGeom.Xform.Define(stage, station)
    _shape(stage, station+"/Top", UsdGeom.Cube, (30.80, 5.45, .77), (.40, .40, .03), (.30, .18, .10))
    for x in (-.32, .32):
        for y in (-.32, .32):
            _shape(stage, station+f"/Leg{'P' if x>0 else 'N'}{'P' if y>0 else 'N'}",
                   UsdGeom.Cube, (30.80+x, 5.45+y, .37), (.025, .025, .37), (.18, .18, .18))
    bin_path = f"{DEMO_ROOT}/RoundedBin"
    UsdGeom.Xform.Define(stage, bin_path)
    _shape(stage, bin_path+"/Body", UsdGeom.Cylinder, (30.65, 6.05, .29), (1, 1, 1),
           (.19, .30, .31), radius=.19, height=.58)
    _shape(stage, bin_path+"/Lid", UsdGeom.Sphere, (30.65, 6.05, .58), (1, 1, .22),
           (.28, .39, .40), radius=.195)
    for name, y in (("ClosedCaddy", 5.25), ("UtensilCaddy", 5.65)):
        path = f"{DEMO_ROOT}/{name}"
        UsdGeom.Xform.Define(stage, path)
        # A solid visual holder is sufficient here; no authored material/mass cue.
        _shape(stage, path+"/Holder", UsdGeom.Cylinder, (30.50, y, .865), (1, 1, 1),
               (.65, .70, .70), radius=.075, height=.13)
        if name == "ClosedCaddy":
            _shape(stage, path+"/Lid", UsdGeom.Sphere, (30.50, y, .93), (1, 1, .28),
                   (.70, .75, .75), radius=.078)
        else:
            source = ASSET_ROOT / "usd/utensil-fork.usda"
            layer_path = stage.GetEditTarget().GetLayer().realPath
            reference = str(source) if not layer_path else os.path.relpath(source, Path(layer_path).parent).replace("\\", "/")
            for i, offset in enumerate((-.035, 0, .035)):
                fork = UsdGeom.Xform.Define(stage, path+f"/Fork{i}")
                fork.GetPrim().GetReferences().AddReference(reference)
                fork.AddTranslateOp().Set(Gf.Vec3d(30.50, y+offset, .95))
                fork.AddRotateYOp().Set(90.0)
                fork.AddScaleOp().Set(Gf.Vec3d(.48))
    chair = UsdGeom.Xformable(stage.GetPrimAtPath(CHAIR_PATH))
    cache = UsdGeom.XformCache()
    world = cache.GetLocalToWorldTransform(chair.GetPrim())
    world.SetTranslateOnly(world.ExtractTranslation()+Gf.Vec3d(-.05, 0, 0))
    parent = cache.GetLocalToWorldTransform(chair.GetPrim().GetParent())
    chair.MakeMatrixXform().Set(world*parent.GetInverse())
    return demo_prop_records(stage)


def author_demo_payload(stage):
    """Rigid mounted presentation load inside the existing base footprint.

    No free food dynamics or calibrated payload mass is claimed. Colliders
    belong to the existing chassis, without introducing a new rigid body.
    """
    if stage.GetPrimAtPath(PAYLOAD_PATH):
        raise ValueError("serving tray already exists")
    UsdGeom.Xform.Define(stage, PAYLOAD_PATH)
    _shape(stage, PAYLOAD_PATH+"/Tray", UsdGeom.Cube, (-.04, 0, .447),
           (.12, .14, .008), (.16, .12, .08))
    for model, width, z in (("plate-dinner", .18, .455), ("bread", .08, .480)):
        source = ASSET_ROOT / f"usd/{model}.usda"
        asset = Usd.Stage.Open(str(source))
        bounds = read_world_bounds(asset, "/Asset")
        scale = width/bounds.size[0]
        prim = UsdGeom.Xform.Define(stage, PAYLOAD_PATH+"/"+model.replace("-", "_"))
        layer_path = stage.GetEditTarget().GetLayer().realPath
        reference = str(source) if not layer_path else os.path.relpath(source, Path(layer_path).parent).replace("\\", "/")
        prim.GetPrim().GetReferences().AddReference(reference)
        prim.AddTranslateOp().Set(Gf.Vec3d(-.04-scale*bounds.center[0],
                                         -scale*bounds.center[1], z-scale*bounds.minimum[2]))
        prim.AddScaleOp().Set(Gf.Vec3d(scale))
        mesh = stage.GetPrimAtPath(str(prim.GetPath())+"/Geometry")
        UsdPhysics.MeshCollisionAPI.Apply(mesh).CreateApproximationAttr("convexHull")
    return payload_geometry_check(stage)


def payload_geometry_check(stage):
    if not stage.GetPrimAtPath(PAYLOAD_PATH):
        return {"present": False}
    parent = stage.GetPrimAtPath("/World/Robot/chassis_link")
    box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeRelativeBound(
        stage.GetPrimAtPath(PAYLOAD_PATH), parent).ComputeAlignedRange()
    low, high = tuple(box.GetMin()), tuple(box.GetMax())
    within = low[0] >= -.587351 and high[0] <= .14 and low[1] >= -.2466 and high[1] <= .2466
    if not within or low[2] < .439-1e-6 or high[2] > .515:
        raise ValueError("mounted presentation load exceeds declared footprint/height scope")
    return dict(present=True, parent_frame_bounds_min=low, parent_frame_bounds_max=high,
                within_original_planar_footprint=within, rigid_attachment=True,
                scope="mounted serving presentation; no free payload/food dynamics or calibrated load study")


def build_demo_scene(output, *, base_scene):
    output, base_scene = Path(output).resolve(), Path(base_scene).resolve()
    additions = output.with_name(output.stem+"_dressing.usda")
    if output == base_scene or output.exists() or additions.exists():
        raise ValueError("new scene and dressing paths required")
    base = Usd.Stage.Open(str(base_scene))
    if base.GetEndTimeCode() <= base.GetStartTimeCode():
        raise ValueError("nonempty simulation time range required")
    sources = {p.realPath: hashlib.sha256(Path(p.realPath).read_bytes()).hexdigest()
               for p in base.GetUsedLayers() if p.realPath and Path(p.realPath).is_file()}
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
        props = author_demo_props(stage)
        payload = author_demo_payload(stage)
    layer.Save()
    stage.GetRootLayer().Save()
    if not all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in sources.items()):
        raise RuntimeError("source scene changed")
    result = dict(scene=str(output), additions=str(additions), source_sha256=sources,
                  props=props, source_files_unchanged=True, payload_added=True, payload=payload,
                  scope="scene construction only; no perception or navigation acceptance")
    output.with_suffix(".json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    return result

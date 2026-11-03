"""Restaurant dressing from audited CC0 meshes; authored data is not CRA evidence.

The original restaurant stays intact.  Props are static visual/collision meshes
in a separate layer; no mass, temperature, semantic risk or desired phi is set.
"""

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from robot.config import PROJECT_ROOT
from robot.scene.controlled_semantic_obstacle import read_world_bounds


ASSET_ROOT = PROJECT_ROOT / "assets/restaurant/kenney_food_kit"
SERVICE_ROOT = "/World/CRAServiceDressing"
LONG_ROUTE_WORLD_XY = (
    (7.8, 4.05), (12.0, 4.05), (20.0, 4.05), (25.8, 4.05),
    (27.4, 7.4), (28.8, 9.5), (28.8, 12.2), (29.8, 14.6), (30.4, 15.8),
)
WIDE_CORNER_ROUTE_WORLD_XY = (*LONG_ROUTE_WORLD_XY[:3], (26.2, 4.05), *LONG_ROUTE_WORLD_XY[4:])
# Separate bounded development route: the final long route is not yet accepted.
SHORT_START_WORLD_XYZ = (28.8, 5.6, 0.07382)
SHORT_START_YAW = math.pi / 3
SHORT_GOAL_LOCAL_XY = (2.2 * math.sin(SHORT_START_YAW), 2.2 * math.cos(SHORT_START_YAW))

# These poses select an experiment view, never an estimator target or score.
# Only the original bowl view has a motion gate at this stage.
RESTAURANT_OBSERVATIONS = {
    "bowl": ("FoodBowl", SHORT_START_WORLD_XYZ, SHORT_START_YAW),
    "pot": ("SecondPot", (30.4, 14.6, .07382), math.radians(-70)),
    "counter": ("CounterKnifeBlock", (26.6, 4.6, .07382), math.radians(145)),
    "front": ("FrontCup", (14.9, 4.1, .07382), math.radians(-135)),
}


def restaurant_route_local_xy(name="short", *, human_start_view="original", wide_corner=False):
    """Obstacle-unaware nominal waypoints in the original bowl start frame.

    The 3.95 m service segment stays in the existing bounded field; it is not
    the 30.88 m final route and does not extend evidence timestamps.
    """
    if not isinstance(wide_corner, bool) or wide_corner and name != 'long-service':
        raise ValueError('wide corner is an explicit long-service route variant')
    if human_start_view not in {"original", "approach-forward", "crossing-approach"}:
        raise ValueError("unknown human start view")
    if human_start_view != "original" and name != "human-service":
        raise ValueError("experimental start is bounded to the human-service route")
    if name == "short":
        return (SHORT_GOAL_LOCAL_XY,)
    if name == "long-service":
        # A geometric route only; hazards never alter nominal waypoints.
        route = WIDE_CORNER_ROUTE_WORLD_XY if wide_corner else LONG_ROUTE_WORLD_XY
        start = np.asarray(route[0])
        return tuple(map(tuple, np.asarray(route[1:]) - start))
    if name not in {"service", "human-service"}:
        raise ValueError("unknown restaurant route")
    start = np.asarray(SHORT_START_WORLD_XYZ[:2])
    if human_start_view == "crossing-approach":
        start = np.array([28.8, 7.4])
    yaw = math.pi / 2 if human_start_view != "original" else SHORT_START_YAW
    c, s = math.cos(yaw), math.sin(yaw)
    world = np.array([(28.8, 7.8), (28.8 if name == "human-service" else 29.0, 8.7), (28.8, 9.5)])
    return tuple(map(tuple, (world - start) @ np.array([[c, -s], [s, c]])))


@dataclass(frozen=True)
class ServiceProp:
    name: str
    model: str
    base_world_xyz_m: tuple[float, float, float]
    width_m: float
    yaw_degrees: float = 0.0

    def __post_init__(self):
        if not Sdf.Path.IsValidIdentifier(self.name):
            raise ValueError("prop name must be a USD identifier")
        if not self.model or Path(self.model).name != self.model:
            raise ValueError("model must be a simple asset name")
        if len(self.base_world_xyz_m) != 3 or not all(
            math.isfinite(v) for v in self.base_world_xyz_m
        ):
            raise ValueError("prop base must have three finite coordinates")
        if not math.isfinite(self.width_m) or not 0.01 <= self.width_m <= 1.5:
            raise ValueError("prop width must lie in [.01, 1.5] metres")
        if not math.isfinite(self.yaw_degrees):
            raise ValueError("yaw must be finite")


def service_props() -> tuple[ServiceProp, ...]:
    """Ordinary mixed table/counter arrangements, not factor-specific profiles."""
    return (
        ServiceProp("FrontPlate", "plate-dinner", (13.35, 2.55, .7466), .26),
        ServiceProp("FrontBread", "bread", (13.35, 2.55, .769), .14),
        ServiceProp("FrontCup", "cup-coffee", (13.78, 2.55, .7466), .11),
        ServiceProp("CounterKnifeBlock", "knife-block", (24.65, 6.05, .90), .18, 90),
        ServiceProp("CounterPlate", "plate-dinner", (24.1, 6.05, .90), .26),
        ServiceProp("CounterKnife", "utensil-knife", (24.1, 6.05, .935), .23, 20),
        ServiceProp("FoodBowl", "bowl-soup", (31.25, 7.40, .7466), .30),
        ServiceProp("FoodSpoon", "utensil-spoon", (31.68, 7.40, .7466), .20, 90),
        ServiceProp("SecondPot", "pot-stew", (31.20, 12.25, .7466), .34),
        ServiceProp("SecondPlate", "plate-dinner", (31.64, 11.75, .7466), .24),
        ServiceProp("DestinationPlate", "plate-dinner", (31.46, 17.75, .7466), .26),
    )


def audit_source_assets(asset_root: Path = ASSET_ROOT) -> dict:
    """Verify every selected original file against the intake manifest."""
    manifest = json.loads((asset_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["license"] != "CC0-1.0":
        raise ValueError("expected audited CC0 source")
    for item in manifest["files"]:
        path = (asset_root / item["local_path"]).resolve()
        if not path.is_relative_to(asset_root.resolve()):
            raise ValueError("asset manifest escapes its root")
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"source fingerprint mismatch: {item['local_path']}")
    return manifest


def convert_food_assets(asset_root: Path = ASSET_ROOT) -> tuple[Path, ...]:
    """Convert only audited OBJ meshes with existing trimesh; no code downloads.

    Kenney OBJ is Y-up.  Rotate (x,y,z) -> (x,-z,y), preserve face UVs, and
    normalize the mesh base to z=0.  Metric sizing is an authoring choice later.
    """
    import trimesh

    manifest = audit_source_assets(asset_root)
    converted = asset_root / "usd"
    converted.mkdir(exist_ok=True)
    result = []
    for name in manifest["selected_models"]:
        path = converted / f"{name}.usda"
        model = trimesh.load(asset_root / f"source/{name}.obj", force="mesh", process=False)
        points = np.asarray(model.vertices)[:, [0, 2, 1]].copy()
        points[:, 1] *= -1
        points[:, 2] -= points[:, 2].min()
        if not np.all(np.isfinite(points)) or len(model.faces) == 0:
            raise ValueError(f"invalid source mesh: {name}")
        stage = Usd.Stage.CreateInMemory()
        asset = UsdGeom.Xform.Define(stage, "/Asset")
        stage.SetDefaultPrim(asset.GetPrim())
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, "Z")
        mesh = UsdGeom.Mesh.Define(stage, "/Asset/Geometry")
        mesh.CreatePointsAttr([Gf.Vec3f(*v) for v in points])
        mesh.CreateFaceVertexCountsAttr([3] * len(model.faces))
        mesh.CreateFaceVertexIndicesAttr(model.faces.reshape(-1).tolist())
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDoubleSidedAttr(True)
        mesh.CreateExtentAttr([Gf.Vec3f(*points.min(0)), Gf.Vec3f(*points.max(0))])
        uv = np.asarray(model.visual.uv)
        if uv.shape != (len(points), 2) or not np.all(np.isfinite(uv)):
            raise ValueError(f"missing/invalid source UVs: {name}")
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
        ).Set([Gf.Vec2f(*v) for v in uv[model.faces.reshape(-1)]])
        material = UsdShade.Material.Define(stage, "/Asset/Material")
        surface = UsdShade.Shader.Define(stage, "/Asset/Material/Surface")
        surface.CreateIdAttr("UsdPreviewSurface")
        surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.65)
        tex = UsdShade.Shader.Define(stage, "/Asset/Material/Color")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("../source/Textures/colormap.png"))
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        st = UsdShade.Shader.Define(stage, "/Asset/Material/UV")
        st.CreateIdAttr("UsdPrimvarReader_float2")
        st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
        surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
        material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        # A static triangle mesh retains bowl cavities and thin utensils.  No
        # authored rigid mass or simplified convex hull enters the estimator.
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")
        stage.GetRootLayer().Export(str(path))
        result.append(path)
    return tuple(result)


def author_service_props(stage: Usd.Stage, *, asset_root: Path = ASSET_ROOT) -> list[dict]:
    """Add static props to the current edit target; preserve existing prims."""
    if stage.GetPrimAtPath(SERVICE_ROOT).IsValid():
        raise ValueError("restaurant service dressing already exists")
    manifest = audit_source_assets(asset_root)
    available = set(manifest["selected_models"])
    UsdGeom.Xform.Define(stage, SERVICE_ROOT)
    records = []
    for item in service_props():
        if item.model not in available:
            raise ValueError("unaudited model")
        path = asset_root / f"usd/{item.model}.usda"
        if not path.is_file():
            raise FileNotFoundError(f"convert audited mesh first: {path}")
        root = UsdGeom.Xform.Define(stage, f"{SERVICE_ROOT}/{item.name}")
        layer_path = stage.GetEditTarget().GetLayer().realPath
        reference = str(path) if not layer_path else os.path.relpath(path, Path(layer_path).parent).replace("\\", "/")
        root.GetPrim().GetReferences().AddReference(reference)
        before = read_world_bounds(stage, str(root.GetPath()))
        scale = item.width_m / before.size[0]
        # Asset conversion centers remain source-centered in XY; placement
        # explicitly centers the actual scaled bounds, not authored semantics.
        center = np.array(before.center)
        yaw = math.radians(item.yaw_degrees)
        rotated = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]) @ center[:2]
        origin = np.array(item.base_world_xyz_m) - [*(scale * rotated), scale * before.minimum[2]]
        root.AddTranslateOp().Set(Gf.Vec3d(*origin))
        root.AddRotateZOp().Set(item.yaw_degrees)
        root.AddScaleOp().Set(Gf.Vec3d(scale))
        bounds = read_world_bounds(stage, str(root.GetPath()))
        records.append({"name":item.name, "model":item.model, "prim_path":str(root.GetPath()),
                        "base_world_xyz_m":item.base_world_xyz_m,
                        "bounds_minimum":bounds.minimum, "bounds_maximum":bounds.maximum,
                        "source_license":"CC0-1.0", "provenance":"authored_placement_and_rendering_only",
                        "available_as_physical_factor_input":False})
    return records


def build_service_scene(output: Path, *, base_scene: Path, asset_root: Path = ASSET_ROOT) -> dict:
    """Save a new composed scene with a separate additions layer; never save base."""
    output, base_scene = output.resolve(), base_scene.resolve()
    if output == base_scene or output.exists():
        raise ValueError("build requires a new scene path distinct from the base")
    original_hash = hashlib.sha256(base_scene.read_bytes()).hexdigest()
    base_layer = Sdf.Layer.FindOrOpen(str(base_scene))
    if base_layer.endTimeCode <= base_layer.startTimeCode:
        raise ValueError("base scene needs a nonempty simulation time range")
    output.parent.mkdir(parents=True, exist_ok=True)
    additions = output.with_name(output.stem + "_additions.usda")
    if additions.exists():
        raise ValueError("additions layer already exists")
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, "Z")
    # Root-layer timing is not supplied by a weaker sublayer. Without it Kit
    # can loop a zero-length timeline and invalidate acquisition epochs.
    stage.SetStartTimeCode(base_layer.startTimeCode)
    stage.SetEndTimeCode(base_layer.endTimeCode)
    stage.SetTimeCodesPerSecond(base_layer.timeCodesPerSecond)
    stage.SetFramesPerSecond(base_layer.framesPerSecond)
    layer = Sdf.Layer.CreateNew(str(additions))
    stage.GetRootLayer().subLayerPaths = [additions.name, os.path.relpath(base_scene, output.parent).replace("\\", "/")]
    with Usd.EditContext(stage, layer):
        props = author_service_props(stage, asset_root=asset_root)
    layer.Save()
    stage.GetRootLayer().Save()
    unchanged = original_hash == hashlib.sha256(base_scene.read_bytes()).hexdigest()
    if not unchanged:
        raise RuntimeError("base scene changed")
    return {"scene":str(output), "additions":str(additions), "base_scene":str(base_scene),
            "base_sha256":original_hash, "base_unchanged":unchanged, "props":props,
            "time_codes_per_second":stage.GetTimeCodesPerSecond(),
            "start_time_code":stage.GetStartTimeCode(),"end_time_code":stage.GetEndTimeCode(),
            "long_route_proposal_world_xy_m":LONG_ROUTE_WORLD_XY,
            "humans_added":False,"robot_payload_added":False,"wheel_motion":False,
            "scope":"static authored restaurant dressing; no perception or CRA validation"}

"""Audited CC0 server rendering asset; authored anatomy is never CRA evidence."""

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdSkel

from robot.config import PROJECT_ROOT
from robot.scene.controlled_semantic_obstacle import read_world_bounds
from robot.scene.restaurant_service_build import SHORT_START_WORLD_XYZ, SHORT_START_YAW

SERVER_ASSET_ROOT = PROJECT_ROOT / "assets/restaurant/kenney_server"
SERVER_PRIM_PATH = "/World/CRAServiceActors/Server"
SERVER_WORLD_XYZ = (29.8, 8.9, 0.0)
# Wait facing the service-aisle approach, rather than presenting a side profile.
# This authored presentation choice is never an estimator input.
SERVER_YAW_DEGREES = -17.0
SERVER_OBSERVATION = ("Server", SHORT_START_WORLD_XYZ, SHORT_START_YAW)


def audit_server_sources(asset_root=SERVER_ASSET_ROOT):
    root = Path(asset_root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["license"] != "CC0-1.0":
        raise ValueError("server source must have the audited CC0 license")
    for item in manifest["files"]:
        path = (root / item["local_path"]).resolve()
        if (
            not path.is_relative_to(root)
            or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]
        ):
            raise ValueError("server source fingerprint/path mismatch")
    return manifest


def bake_server_asset(
    body_path, idle_path, output, *, asset_root=SERVER_ASSET_ROOT, height_m=1.75
):
    """Bake the matching source rig's idle t=0 pose into static Z-up geometry.

    Inputs are outputs of the already installed Isaac FBX converter. Only
    authored mesh/UV/material data are exported, with no mass or risk labels.
    """
    audit_server_sources(asset_root)
    output = Path(output).resolve()
    if output.exists() or not math.isfinite(height_m) or not 1.2 <= height_m <= 2.1:
        raise ValueError(
            "server conversion needs a new output and explicit bounded authored height"
        )
    body, idle = Usd.Stage.Open(str(body_path)), Usd.Stage.Open(str(idle_path))
    if any(
        UsdGeom.GetStageUpAxis(stage) != "Y"
        or UsdGeom.GetStageMetersPerUnit(stage) != 1
        for stage in (body, idle)
    ):
        raise ValueError("unexpected converter coordinate/unit contract")
    skeleton = UsdSkel.Skeleton(body.GetPrimAtPath("/World/Root/Skeleton"))
    animation = UsdSkel.Animation(idle.GetPrimAtPath("/World/Root/Root_Idle"))
    if list(skeleton.GetJointsAttr().Get()) != list(animation.GetJointsAttr().Get()):
        raise ValueError("body and idle joint order disagree")
    original = UsdGeom.Mesh(body.GetPrimAtPath("/World/Root/characterMedium"))
    for path in ("/World/Root", str(skeleton.GetPath()), str(original.GetPath())):
        if not np.allclose(
            np.asarray(
                UsdGeom.Xformable(body.GetPrimAtPath(path)).GetLocalTransformation()
            ),
            np.eye(4),
        ):
            raise ValueError("unexpected nonidentity converted rig transform")
    with Usd.EditContext(body, body.GetSessionLayer()):
        Sdf.CreatePrimInLayer(body.GetSessionLayer(), "/World/Root")
        Sdf.CopySpec(
            idle.Flatten(),
            str(animation.GetPath()),
            body.GetSessionLayer(),
            "/World/Root/Idle",
        )
        UsdSkel.BindingAPI.Apply(
            skeleton.GetPrim()
        ).CreateAnimationSourceRel().SetTargets(["/World/Root/Idle"])
        cache = UsdSkel.Cache()
        cache.Populate(
            UsdSkel.Root(body.GetPrimAtPath("/World/Root")), Usd.PrimDefaultPredicate
        )
        points = original.GetPointsAttr().Get()
        if not cache.GetSkinningQuery(original.GetPrim()).ComputeSkinnedPoints(
            cache.GetSkelQuery(skeleton).ComputeSkinningTransforms(0), points, 0
        ):
            raise RuntimeError("idle skinning failed")
    vertices = np.asarray(points, dtype=float)[:, [0, 2, 1]].copy()
    vertices[:, 1] *= -1
    vertices[:, 2] -= vertices[:, 2].min()
    if not np.isfinite(vertices).all() or vertices[:, 2].max() <= 0:
        raise ValueError("invalid posed geometry")
    vertices *= height_m / vertices[:, 2].max()
    vertices[:, :2] -= (vertices[:, :2].min(0) + vertices[:, :2].max(0)) / 2
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, "Z")
    asset = UsdGeom.Xform.Define(stage, "/Server")
    stage.SetDefaultPrim(asset.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Server/Body")
    mesh.CreatePointsAttr([Gf.Vec3f(*v) for v in vertices])
    mesh.CreateFaceVertexCountsAttr(original.GetFaceVertexCountsAttr().Get())
    mesh.CreateFaceVertexIndicesAttr(original.GetFaceVertexIndicesAttr().Get())
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateExtentAttr([Gf.Vec3f(*vertices.min(0)), Gf.Vec3f(*vertices.max(0))])
    old_uv = UsdGeom.PrimvarsAPI(original).GetPrimvar("st")
    uv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, old_uv.GetInterpolation()
    )
    uv.Set(old_uv.Get())
    uv.SetIndices(old_uv.GetIndices())
    material = UsdShade.Material.Define(stage, "/Server/Material")
    shader = UsdShade.Shader.Define(stage, "/Server/Material/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
    texture = UsdShade.Shader.Define(stage, "/Server/Material/Texture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(
            os.path.relpath(
                Path(asset_root) / "source/Skins/skaterMaleA.png", output.parent
            ).replace("\\", "/")
        )
    )
    texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    reader = UsdShade.Shader.Define(stage, "/Server/Material/UV")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.ConnectableAPI(), "result"
    )
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        texture.ConnectableAPI(), "rgb"
    )
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")
    # Code-native wardrobe geometry, not a texture edit or physical parameter.
    for name, center, size in (
        ("ApronBib", (0, -0.05, 0.88), (0.28, 0.012, 0.35)),
        ("ApronSkirt", (0, -0.05, 0.64), (0.36, 0.012, 0.28)),
        ("LeftStrap", (-0.10, -0.05, 1.055), (0.025, 0.012, 0.12)),
        ("RightStrap", (0.10, -0.05, 1.055), (0.025, 0.012, 0.12)),
    ):
        cloth = UsdGeom.Cube.Define(stage, "/Server/" + name)
        cloth.CreateSizeAttr(1.0)
        cloth.AddTranslateOp().Set(Gf.Vec3d(*center))
        cloth.AddScaleOp().Set(Gf.Vec3d(*size))
        cloth.CreateDisplayColorAttr([Gf.Vec3f(0.95)])
        UsdPhysics.CollisionAPI.Apply(cloth.GetPrim())
    output.parent.mkdir(parents=True, exist_ok=True)
    stage.GetRootLayer().Export(str(output))
    return {
        "source_license": "CC0-1.0",
        "posed_source_vertices": len(vertices),
        "pose": "original idle animation t=0; frozen static geometry",
        "authored_height_m": height_m,
        "factor_inputs_supplied": False,
        "asset_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


def build_server_scene(output, base_scene, *, asset_root=SERVER_ASSET_ROOT):
    """Layer one static staff-role avatar over the dressed source restaurant."""
    output, base_scene = Path(output).resolve(), Path(base_scene).resolve()
    additions = output.with_name(output.stem + "_server.usda")
    if output.exists() or additions.exists() or output == base_scene:
        raise ValueError("server scene requires new output layers")
    audit_server_sources(asset_root)
    source_hash = hashlib.sha256(base_scene.read_bytes()).hexdigest()
    base = Usd.Stage.Open(str(base_scene))
    if UsdGeom.GetStageMetersPerUnit(base) != 1 or UsdGeom.GetStageUpAxis(base) != "Z":
        raise ValueError("server placement requires a metre/Z-up source scene")
    if (
        base.GetPrimAtPath(SERVER_PRIM_PATH).IsValid()
        or base.GetEndTimeCode() <= base.GetStartTimeCode()
    ):
        raise ValueError("base already contains server or lacks simulation timing")
    asset = Path(asset_root) / "usd/server.usda"
    if not asset.is_file():
        raise FileNotFoundError("bake audited server asset first")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, "Z")
    stage.SetStartTimeCode(base.GetStartTimeCode())
    stage.SetEndTimeCode(base.GetEndTimeCode())
    stage.SetTimeCodesPerSecond(base.GetTimeCodesPerSecond())
    stage.SetFramesPerSecond(base.GetFramesPerSecond())
    layer = Sdf.Layer.CreateNew(str(additions))
    stage.GetRootLayer().subLayerPaths = [
        additions.name,
        os.path.relpath(base_scene, output.parent).replace("\\", "/"),
    ]
    with Usd.EditContext(stage, layer):
        server = UsdGeom.Xform.Define(stage, SERVER_PRIM_PATH)
        server.GetPrim().GetReferences().AddReference(
            os.path.relpath(asset, additions.parent).replace("\\", "/")
        )
        server.AddTranslateOp().Set(Gf.Vec3d(*SERVER_WORLD_XYZ))
        server.AddRotateZOp().Set(SERVER_YAW_DEGREES)
    layer.Save()
    stage.GetRootLayer().Save()
    if hashlib.sha256(base_scene.read_bytes()).hexdigest() != source_hash:
        raise RuntimeError("source restaurant changed")
    bounds = read_world_bounds(stage, SERVER_PRIM_PATH)
    return {
        "scene": str(output),
        "additions": str(additions),
        "base_scene": str(base_scene),
        "base_sha256": source_hash,
        "base_unchanged": True,
        "server_prim": SERVER_PRIM_PATH,
        "bounds_minimum": bounds.minimum,
        "bounds_maximum": bounds.maximum,
        "humans_added": 1,
        "actor_motion_enabled": False,
        "robot_payload_added": False,
        "factor_inputs_supplied": False,
        "scope": "static stylized staff-role rendering asset, not calibrated human anatomy/mass or perception acceptance",
    }

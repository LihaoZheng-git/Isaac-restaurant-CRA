"""Matched floor-standing bin fixtures; geometry is evaluation-only, never CRA input."""

import hashlib
import json
import os
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom

from robot.scene.restaurant_demo_build import DEMO_ROOT, _shape, demo_prop_records


CONTAINER_PATH = DEMO_ROOT + "/ServiceContainer"
CONTAINER_CENTER_XY = (28.15, 6.10)


def author_service_container(stage, shape, *, props_only=False):
    """Replace the old bin in a separate layer, with matched width/height/color.

    A box-like service bin and a rounded lidded bin are contact-outline probes.
    Neither is assigned a category, material density, physical sharpness or phi.
    """
    if shape not in {"angular", "rounded"}:
        raise ValueError("container shape must be angular or rounded")
    old = stage.GetPrimAtPath(DEMO_ROOT + "/RoundedBin")
    if not old or not old.IsActive() or stage.GetPrimAtPath(CONTAINER_PATH):
        raise ValueError("requires an active original bin and no existing service container")
    old.SetActive(False)
    if props_only:
        server = stage.GetPrimAtPath("/World/CRAServiceActors/Server")
        if server:
            server.SetActive(False)
    root = UsdGeom.Xform.Define(stage, CONTAINER_PATH)
    root.AddTranslateOp().Set(Gf.Vec3d(*CONTAINER_CENTER_XY, 0.))
    body, lid = (.19, .30, .31), (.28, .39, .40)
    if shape == "angular":
        _shape(stage, CONTAINER_PATH + "/Body", UsdGeom.Cube,
               (0., 0., .23), (.16, .16, .23), body)
        _shape(stage, CONTAINER_PATH + "/Lid", UsdGeom.Cube,
               (0., 0., .47), (.16, .16, .01), lid)
    else:
        _shape(stage, CONTAINER_PATH + "/Body", UsdGeom.Cylinder,
               (0., 0., .23), (1., 1., 1.), body, radius=.16, height=.46)
        _shape(stage, CONTAINER_PATH + "/Lid", UsdGeom.Sphere,
               (0., 0., .46), (1., 1., .125), lid, radius=.16)
    return next(p for p in demo_prop_records(stage) if p["name"] == "ServiceContainer")


def build_container_scene(output, *, base_scene, shape, props_only=False):
    """Write additive layers, preserving every byte of the accepted demo scene."""
    output, base_scene = Path(output).resolve(), Path(base_scene).resolve()
    additions = output.with_name(output.stem + "_container.usda")
    if output == base_scene or output.exists() or additions.exists() or output.with_suffix(".json").exists():
        raise ValueError("new scene, additions and manifest paths required")
    if shape not in {"angular", "rounded"}:
        raise ValueError("container shape must be angular or rounded")
    base = Usd.Stage.Open(str(base_scene))
    if base.GetEndTimeCode() <= base.GetStartTimeCode():
        raise ValueError("nonempty simulation timeline required")
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
        target = author_service_container(stage, shape, props_only=props_only)
        props = demo_prop_records(stage)
    layer.Save()
    stage.GetRootLayer().Save()
    if not all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in sources.items()):
        raise RuntimeError("source scene changed")
    result = dict(scene=str(output), additions=str(additions), source_sha256=sources,
                  source_files_unchanged=True, props=props, target=target, fixture_shape=shape,
                  stationary_props_only=props_only,
                  scope="authored restaurant service-bin comparison; no supplied risk labels, physical sharpness or navigation acceptance")
    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result

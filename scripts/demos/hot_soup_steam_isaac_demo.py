"""Author a stationary hot-soup plume proxy in Isaac Sim.

This scene uses translucent USD spheres, not fluid simulation, temperature,
or a trained steam detector. It exports a fixed-camera USD scene for future
perception-data and false-positive work. No robot or wheel command is created.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import traceback


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keep Warp's generated cache inside the writable project workspace.
os.environ.setdefault(
    "WARP_CACHE_PATH", str(ROOT / "outputs" / "hot_soup_steam_isaac" / "warp")
)

from isaacsim import SimulationApp


HEADLESS = os.environ.get("ISAAC_SIM_HEADLESS", "1") != "0"
simulation_app = SimulationApp({"headless": HEADLESS})


import isaacsim.core.experimental.utils.stage as stage_utils
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade


OUTPUT_DIRECTORY = ROOT / "outputs" / "hot_soup_steam_isaac"
CAMERA_PATH = "/World/Camera"


def material(stage, path: str, color: tuple[float, float, float], opacity=1.0):
    """Create a small USD Preview Surface material."""

    result = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*color)
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
    result.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return result


def bind(prim, visual_material) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(visual_material)


def set_transform(prim, position, scale=None) -> None:
    xform = UsdGeom.Xformable(prim)
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))
    if scale is not None:
        xform.AddScaleOp().Set(Gf.Vec3f(*scale))


def create_scene():
    stage = stage_utils.create_new_stage(template="empty")
    stage_utils.set_stage_units(meters_per_unit=1.0, kilograms_per_unit=1.0)
    stage_utils.set_stage_up_axis("Z")

    blue = material(stage, "/World/Looks/Bowl", (0.08, 0.22, 0.55))
    soup = material(stage, "/World/Looks/Soup", (0.82, 0.22, 0.04))
    plume = material(stage, "/World/Looks/Plume", (0.92, 0.94, 0.97), 0.28)
    table = material(stage, "/World/Looks/Table", (0.35, 0.16, 0.06))

    tabletop = UsdGeom.Cube.Define(stage, "/World/Table").GetPrim()
    set_transform(tabletop, (0.0, 0.0, 0.36), (1.2, 0.85, 0.08))
    bind(tabletop, table)

    bowl = UsdGeom.Cylinder.Define(stage, "/World/HotSoup/Bowl")
    bowl.CreateRadiusAttr(0.34)
    bowl.CreateHeightAttr(0.18)
    set_transform(bowl.GetPrim(), (0.0, 0.0, 0.58))
    bind(bowl.GetPrim(), blue)

    surface = UsdGeom.Cylinder.Define(stage, "/World/HotSoup/SoupSurface")
    surface.CreateRadiusAttr(0.29)
    surface.CreateHeightAttr(0.025)
    set_transform(surface.GetPrim(), (0.0, 0.0, 0.72))
    bind(surface.GetPrim(), soup)

    rim = UsdGeom.Cylinder.Define(stage, "/World/HotSoup/OpenRim")
    rim.CreateRadiusAttr(0.35)
    rim.CreateHeightAttr(0.035)
    set_transform(rim.GetPrim(), (0.0, 0.0, 0.695))
    bind(rim.GetPrim(), blue)

    plume_specs = (
        ((-0.09, 0.00, 0.86), (0.10, 0.08, 0.14)),
        ((0.07, 0.02, 1.02), (0.13, 0.09, 0.18)),
        ((-0.03, 0.00, 1.23), (0.15, 0.10, 0.22)),
        ((0.11, 0.03, 1.47), (0.19, 0.12, 0.27)),
        ((-0.04, 0.04, 1.72), (0.22, 0.15, 0.30)),
    )
    for index, (position, scale) in enumerate(plume_specs):
        sphere = UsdGeom.Sphere.Define(stage, f"/World/HotSoup/Plume_{index:02d}")
        sphere.CreateRadiusAttr(1.0)
        set_transform(sphere.GetPrim(), position, scale)
        bind(sphere.GetPrim(), plume)

    light = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
    light.CreateIntensityAttr(1000.0)
    light.CreateAngleAttr(2.0)
    UsdGeom.Xformable(light.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(45, 0, 35))

    camera = UsdGeom.Camera.Define(stage, CAMERA_PATH)
    camera.GetPrim().ApplyAPI("OmniSensorAPI")
    camera.CreateFocalLengthAttr(38.0)
    camera.CreateHorizontalApertureAttr(36.0)
    view = Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(2.3, -3.2, 1.75),
        Gf.Vec3d(0.0, 0.0, 0.95),
        Gf.Vec3d(0.0, 0.0, 1.0),
    )
    UsdGeom.Xformable(camera.GetPrim()).AddTransformOp().Set(view.GetInverse())
    return stage


def author_stationary_scene() -> Path:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    stage = create_scene()
    usd_path = OUTPUT_DIRECTORY / "hot_soup_plume_proxy.usda"
    stage.GetRootLayer().Export(str(usd_path))

    plume_prims = tuple(
        prim
        for prim in stage.Traverse()
        if str(prim.GetPath()).startswith("/World/HotSoup/Plume_")
    )
    if len(plume_prims) != 5:
        raise RuntimeError("expected five authored plume proxy prims")
    if not stage.GetPrimAtPath(CAMERA_PATH).IsA(UsdGeom.Camera):
        raise RuntimeError("stationary diagnostic camera was not authored")
    if not stage.GetPrimAtPath("/World/HotSoup/SoupSurface").IsValid():
        raise RuntimeError("soup surface was not authored")
    print("STATIONARY ISAAC HOT-SOUP PLUME FIXTURE")
    print(f"stage={usd_path}")
    print(f"plume_proxy_prims={len(plume_prims)}; camera={CAMERA_PATH}")
    print("robot_motion=none; detector=none; temperature=unavailable")
    print("PASS: Isaac authored a stationary visual proxy scene", flush=True)
    return usd_path


if __name__ == "__main__":
    try:
        author_stationary_scene()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()

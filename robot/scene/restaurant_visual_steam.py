"""Animated rendering fixture only; never a supplied thermal score or sensor mask."""

import math

from pxr import Gf, UsdGeom

from robot.scene.controlled_semantic_obstacle import read_world_bounds

STEAM_PATH = "/World/CRAServiceDemoV2Steam"


def author_visual_steam(stage, *, source_path):
    if stage.GetPrimAtPath(STEAM_PATH):
        raise ValueError("visual steam already exists")
    bounds = read_world_bounds(stage, source_path)
    UsdGeom.Xform.Define(stage, STEAM_PATH)
    # Match the already-used bounded RGB puff fixture. This is not a physical
    # aerosol simulation. No collider, mass, temperature, category or CRA label.
    for i in range(5):
        puff = UsdGeom.Sphere.Define(stage, STEAM_PATH+f"/Puff{i}")
        puff.CreateRadiusAttr(.045+.009*i)
        # Translation must precede shape scaling in USD's ordered transform
        # stack, otherwise the vertical stretch also scales the world height.
        position = puff.AddTranslateOp()
        puff.AddScaleOp().Set(Gf.Vec3f(1, 1, 1.5))
        puff.CreateDisplayColorAttr([Gf.Vec3f(.88, .89, .90)])
        for tick in range(3001):
            t = tick/10
            position.Set(Gf.Vec3d(bounds.center[0]+.018*math.sin(2*t+i),
                bounds.center[1]+.012*math.cos(1.3*t+i),
                bounds.maximum[2]+.12+.075*i+.012*math.sin(3*t+i)),
                t*stage.GetTimeCodesPerSecond())
    return dict(prim_path=STEAM_PATH, source_prim=source_path,
                animated_seconds=300, colliders=False, available_as_factor_input=False,
                scope="stylized visible puffs; RGB cue validation required; no physical steam/temperature")

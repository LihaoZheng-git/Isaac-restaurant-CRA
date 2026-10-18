import copy

import numpy as np
import pytest
from pxr import Usd, UsdGeom, UsdPhysics

from robot.scene.restaurant_human_carry import (
    BOWL_PATH,
    author_native_carry_bowl,
    summarize_carry_records,
)


def records():
    return [
        dict(
            time=float(t),
            phase="idle_before" if t < 1 else "move" if t < 4 else "stopped_after",
            held=True,
            bowl_xyz=[0, 0.5 * np.clip(t - 1, 0, 3), 1.0],
            hand_distance_m=0.15,
            relative_position_error_m=0.002,
            relative_angle_error_degrees=0.5,
            tilt_degrees=8.0,
            bottom_world_z=0.9,
            layout_clearance_m=0.2,
        )
        for t in np.arange(0, 5, 0.02)
    ]


def test_native_carry_contract_accepts_bounded_attachment_without_perception_claim():
    result = summarize_carry_records(records())
    assert result["passed"] and not result["separate_bowl_perception_validated"]


@pytest.mark.parametrize(
    "failure",
    [
        "detached",
        "sliding",
        "twisting",
        "tilted",
        "dragged",
        "collision",
        "static_prop",
        "repeated_time",
        "nonfinite",
        "nonfinite_pose",
    ],
)
def test_native_carry_rejects_visual_or_attachment_failures(failure):
    rows = copy.deepcopy(records())
    key_value = {
        "detached": ("held", False),
        "sliding": ("relative_position_error_m", 0.06),
        "twisting": ("relative_angle_error_degrees", 30),
        "tilted": ("tilt_degrees", 60),
        "dragged": ("bottom_world_z", 0.3),
        "collision": ("layout_clearance_m", 0),
        "nonfinite": ("relative_position_error_m", float("nan")),
    }
    if failure in key_value:
        k, v = key_value[failure]
        rows[70][k] = v
    elif failure == "static_prop":
        for r in rows:
            r["bowl_xyz"] = [0, 0, 1]
    elif failure == "nonfinite_pose":
        rows[70]["bowl_xyz"] = [float("inf"), 0, 1]
    else:
        rows[70]["time"] = rows[69]["time"]
    assert not summarize_carry_records(rows)["passed"]


def test_bowl_reference_collision_and_scale_are_session_only():
    stage = Usd.Stage.CreateInMemory()
    before = stage.GetRootLayer().ExportToString()
    setup = author_native_carry_bowl(stage)
    assert stage.GetRootLayer().ExportToString() == before
    assert setup["license"].startswith("CC0")
    root = stage.GetPrimAtPath(BOWL_PATH)
    assert root.HasAPI(UsdPhysics.RigidBodyAPI) and not root.HasAPI(UsdPhysics.MassAPI)
    assert UsdPhysics.RigidBodyAPI(root).GetKinematicEnabledAttr().Get()
    mesh = next(p for p in Usd.PrimRange(root) if p.IsA(UsdGeom.Mesh))
    assert UsdPhysics.CollisionAPI(mesh).GetCollisionEnabledAttr().Get()
    assert (
        UsdPhysics.MeshCollisionAPI(mesh).GetApproximationAttr().Get() == "convexHull"
    )
    assert setup["local_half_extents"][0] * 2 == pytest.approx(0.30)

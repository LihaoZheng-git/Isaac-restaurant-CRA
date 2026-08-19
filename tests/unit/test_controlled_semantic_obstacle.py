import math

import numpy as np
from pxr import Gf, Usd, UsdGeom
import pytest

from robot.scene.controlled_semantic_obstacle import (
    PlanarFootprint,
    WorldBounds3D,
    line_segment_intersects_planar_bounds,
    oriented_footprint_clearance_to_bounds,
    place_existing_prim_world_origin,
)


def _make_stage() -> tuple[Usd.Stage, str]:
    stage = Usd.Stage.CreateInMemory()
    parent = UsdGeom.Xform.Define(stage, "/World/ScaledParent")
    parent.AddTranslateOp().Set(Gf.Vec3d(10.0, 5.0, 0.0))
    parent.AddRotateZOp().Set(90.0)
    parent.AddScaleOp().Set(Gf.Vec3f(0.01, 0.01, 0.01))
    obstacle_path = "/World/ScaledParent/Obstacle"
    obstacle = UsdGeom.Xform.Define(stage, obstacle_path)
    obstacle.AddTranslateOp().Set(Gf.Vec3d(100.0, 200.0, 0.0))
    obstacle.AddRotateZOp().Set(25.0)
    cube = UsdGeom.Cube.Define(stage, f"{obstacle_path}/Geometry")
    cube.CreateSizeAttr(100.0)
    return stage, obstacle_path


def test_session_placement_reaches_target_without_changing_root_layer() -> None:
    stage, obstacle_path = _make_stage()
    root_before = stage.GetRootLayer().ExportToString()

    result = place_existing_prim_world_origin(
        stage,
        prim_path=obstacle_path,
        target_world_origin=(27.75, 5.0, 0.0),
    )

    assert result.target_world_origin == pytest.approx((27.75, 5.0, 0.0))
    assert result.placed_bounds.size == pytest.approx(
        result.original_bounds.size
    )
    assert stage.GetRootLayer().ExportToString() == root_before
    assert "Obstacle" in stage.GetSessionLayer().ExportToString()


def test_session_placement_rejects_missing_translate_op() -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/Obstacle")

    with pytest.raises(RuntimeError, match="exactly one"):
        place_existing_prim_world_origin(
            stage,
            prim_path="/World/Obstacle",
            target_world_origin=(1.0, 2.0, 0.0),
        )


def test_oriented_footprint_clearance_is_positive_when_separated() -> None:
    bounds = WorldBounds3D(
        minimum=(1.0, -0.25, 0.0),
        maximum=(1.5, 0.25, 1.0),
    )
    footprint = PlanarFootprint(
        front_extent=0.2,
        rear_extent=0.4,
        left_extent=0.25,
        right_extent=0.25,
    )

    clearance = oriented_footprint_clearance_to_bounds(
        (0.0, 0.0, 0.0),
        footprint=footprint,
        obstacle_bounds=bounds,
    )

    assert clearance == pytest.approx(0.8)


def test_oriented_footprint_clearance_detects_overlap_after_rotation() -> None:
    bounds = WorldBounds3D(
        minimum=(0.15, 0.15, 0.0),
        maximum=(0.45, 0.45, 1.0),
    )
    footprint = PlanarFootprint(
        front_extent=0.4,
        rear_extent=0.4,
        left_extent=0.1,
        right_extent=0.1,
    )

    clearance = oriented_footprint_clearance_to_bounds(
        (0.0, 0.0, math.radians(45.0)),
        footprint=footprint,
        obstacle_bounds=bounds,
    )

    assert clearance == 0.0


@pytest.mark.parametrize(
    ("end", "expected"),
    [
        ((3.0, 0.0), True),
        ((3.0, 1.0), False),
        ((0.5, 0.0), False),
    ],
)
def test_nominal_segment_intersection(end, expected) -> None:
    bounds = WorldBounds3D(
        minimum=(1.0, -0.25, 0.0),
        maximum=(1.5, 0.25, 1.0),
    )

    assert (
        line_segment_intersects_planar_bounds(
            np.array([0.0, 0.0]),
            end,
            obstacle_bounds=bounds,
        )
        is expected
    )

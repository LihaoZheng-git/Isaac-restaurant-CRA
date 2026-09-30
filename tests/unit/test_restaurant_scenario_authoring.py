import math

from pxr import Gf, Usd, UsdGeom
import pytest

from robot.scene.restaurant_risk_scenarios import restaurant_risk_scenario
from robot.scene.restaurant_scenario_authoring import (
    DEFAULT_SCENARIO_ROOT_PATH,
    author_restaurant_risk_scenario,
)


CHAIR_PATH = (
    "/World/Environment/Restaurant/assets_restaurant/"
    "Assets_Banquettes_Right/Geo_TableSet_09/Chair_Array"
)


def _stage_with_chair() -> Usd.Stage:
    stage = Usd.Stage.CreateInMemory()
    chair = UsdGeom.Xform.Define(stage, CHAIR_PATH)
    chair.AddTranslateOp().Set(Gf.Vec3d(10.0, 20.0, 0.0))
    geometry = UsdGeom.Cube.Define(stage, f"{CHAIR_PATH}/Geometry")
    geometry.CreateSizeAttr(1.0)
    geometry.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.4))
    geometry.AddScaleOp().Set(Gf.Vec3f(0.5, 0.6, 0.8))
    return stage


def test_existing_chair_places_bounds_base_and_preserves_root_layer():
    stage = _stage_with_chair()
    root_before = stage.GetRootLayer().ExportToString()

    result = author_restaurant_risk_scenario(
        stage,
        restaurant_risk_scenario("chair"),
        robot_pose_world_xy_yaw=(10.0, 20.0, math.pi / 2.0),
    )

    chair = result.actor("chair")
    assert chair.target_anchor_world_xyz_m == pytest.approx((10.0, 21.7, 0.0))
    assert chair.placed_anchor_world_xyz_m == pytest.approx((10.0, 21.7, 0.0))
    assert chair.bounds_world.minimum[2] == pytest.approx(0.0)
    assert chair.stationary
    assert not result.robot_motion_enabled
    assert not result.actor_motion_enabled
    assert stage.GetRootLayer().ExportToString() == root_before
    assert "Chair_Array" in stage.GetSessionLayer().ExportToString()


def test_elevated_object_uses_support_and_landing_geometry_for_fall_height():
    stage = Usd.Stage.CreateInMemory()
    root_before = stage.GetRootLayer().ExportToString()

    result = author_restaurant_risk_scenario(
        stage,
        restaurant_risk_scenario("elevated_object"),
        robot_pose_world_xy_yaw=(4.0, -2.0, 0.0),
    )

    support = result.actor("support")
    target = result.actor("elevated_target")
    support_height = support.bounds_world.maximum[2]
    landing_height = support.bounds_world.minimum[2]
    fall_height = support_height - landing_height
    object_height = target.bounds_world.size[2]
    assert target.bounds_world.minimum[2] == pytest.approx(support_height)
    assert fall_height == pytest.approx(0.8)
    assert object_height == pytest.approx(0.3)
    assert fall_height != pytest.approx(object_height)
    assert stage.GetRootLayer().ExportToString() == root_before
    session_text = stage.GetSessionLayer().ExportToString()
    assert DEFAULT_SCENARIO_ROOT_PATH.rsplit("/", 1)[1] in session_text
    assert "semantic" not in session_text.casefold()


def test_procedural_pose_orientation_and_configured_velocity_are_transformed():
    stage = Usd.Stage.CreateInMemory()

    result = author_restaurant_risk_scenario(
        stage,
        restaurant_risk_scenario("crossing_person"),
        robot_pose_world_xy_yaw=(1.0, 2.0, math.pi / 2.0),
    )

    person = result.actor("crossing_person_proxy")
    assert person.target_anchor_world_xyz_m == pytest.approx((2.2, 4.2, 0.85))
    assert person.configured_velocity_world_xy_mps == pytest.approx((-0.6, 0.0))
    assert person.bounds_world.size == pytest.approx((0.5, 0.5, 1.7))
    assert person.stationary


def test_duplicate_authoring_root_is_rejected_without_overwriting_fixture():
    stage = Usd.Stage.CreateInMemory()
    scenario = restaurant_risk_scenario("exposed_edge")
    author_restaurant_risk_scenario(
        stage,
        scenario,
        robot_pose_world_xy_yaw=(0.0, 0.0, 0.0),
    )

    with pytest.raises(RuntimeError, match="root already exists"):
        author_restaurant_risk_scenario(
            stage,
            scenario,
            robot_pose_world_xy_yaw=(0.0, 0.0, 0.0),
        )

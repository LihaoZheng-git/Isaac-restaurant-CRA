"""Stationary, session-layer authoring for restaurant risk scenarios.

The authored geometry is an experiment fixture.  Actor roles, dimensions,
velocities, and bounds are oracle information for setup and scoring; this
module does not expose them as runtime camera, LiDAR, or CRA evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

from pxr import Gf, Sdf, Usd, UsdGeom

from robot.risk.cra import PhysicalRiskFactor
from robot.scene.controlled_semantic_obstacle import (
    WorldBounds3D,
    place_existing_prim_world_origin,
    read_world_bounds,
)
from robot.scene.restaurant_risk_scenarios import (
    RestaurantRiskScenarioDefinition,
    RestaurantScenarioActor,
    ScenarioGeometry,
    ScenarioPlacementAnchor,
)


DEFAULT_SCENARIO_ROOT_PATH = "/World/CRARestaurantScenario"


def _finite_planar_pose(value: object) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)):
        raise TypeError("robot_pose_world_xy_yaw must contain three numbers")
    try:
        items = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError(
            "robot_pose_world_xy_yaw must contain three numbers"
        ) from error
    if len(items) != 3:
        raise ValueError("robot_pose_world_xy_yaw must contain three values")
    numbers: list[float] = []
    for item in items:
        if isinstance(item, bool) or not isinstance(item, Real):
            raise TypeError(
                "robot_pose_world_xy_yaw must contain real numbers"
            )
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(
                "robot_pose_world_xy_yaw must contain finite values"
            )
        numbers.append(number)
    return numbers[0], numbers[1], numbers[2]


def _validate_root_path(root_prim_path: object) -> str:
    if not isinstance(root_prim_path, str) or not root_prim_path:
        raise ValueError("root_prim_path must be a nonempty absolute prim path")
    path = Sdf.Path(root_prim_path)
    if not path.IsAbsolutePath() or not path.IsPrimPath():
        raise ValueError("root_prim_path must be an absolute USD prim path")
    return path.pathString


def _robot_position_to_world(
    position_robot_xyz_m: tuple[float, float, float],
    robot_pose_world_xy_yaw: tuple[float, float, float],
) -> tuple[float, float, float]:
    robot_x, robot_y, robot_yaw = robot_pose_world_xy_yaw
    relative_x, relative_y, world_z = position_robot_xyz_m
    cosine = math.cos(robot_yaw)
    sine = math.sin(robot_yaw)
    return (
        robot_x + cosine * relative_x - sine * relative_y,
        robot_y + sine * relative_x + cosine * relative_y,
        world_z,
    )


def _robot_velocity_to_world(
    velocity_robot_xy_mps: tuple[float, float],
    robot_yaw: float,
) -> tuple[float, float]:
    forward, left = velocity_robot_xy_mps
    cosine = math.cos(robot_yaw)
    sine = math.sin(robot_yaw)
    return (
        cosine * forward - sine * left,
        sine * forward + cosine * left,
    )


def _anchor_from_bounds(
    bounds: WorldBounds3D,
    anchor: ScenarioPlacementAnchor,
) -> tuple[float, float, float]:
    center = bounds.center
    if anchor is ScenarioPlacementAnchor.GEOMETRIC_CENTER:
        return center
    if anchor is ScenarioPlacementAnchor.BOUNDS_BASE_CENTER:
        return center[0], center[1], bounds.minimum[2]
    raise AssertionError(f"unsupported placement anchor: {anchor}")


@dataclass(frozen=True)
class AuthoredRestaurantScenarioActor:
    """Oracle-only placement diagnostics for one stationary actor."""

    actor_id: str
    prim_path: str
    geometry: ScenarioGeometry
    placement_anchor: ScenarioPlacementAnchor
    target_anchor_world_xyz_m: tuple[float, float, float]
    placed_anchor_world_xyz_m: tuple[float, float, float]
    bounds_world: WorldBounds3D
    configured_velocity_world_xy_mps: tuple[float, float]
    motion_enabled: bool = False

    @property
    def stationary(self) -> bool:
        return not self.motion_enabled


@dataclass(frozen=True)
class AuthoredRestaurantRiskScenario:
    """Result of authoring one scenario without advancing actor motion."""

    scenario_id: str
    factor: PhysicalRiskFactor
    root_prim_path: str
    robot_pose_world_xy_yaw: tuple[float, float, float]
    actors: tuple[AuthoredRestaurantScenarioActor, ...]
    runtime_sensor_claims: tuple[str, ...]
    robot_motion_enabled: bool = False
    actor_motion_enabled: bool = False

    def actor(self, actor_id: str) -> AuthoredRestaurantScenarioActor:
        """Return one authored actor by its exact catalog identifier."""

        for item in self.actors:
            if item.actor_id == actor_id:
                return item
        raise KeyError(f"unknown authored scenario actor: {actor_id!r}")


def _procedural_center(
    actor: RestaurantScenarioActor,
    target_anchor_world: tuple[float, float, float],
) -> tuple[float, float, float]:
    if actor.placement_anchor is ScenarioPlacementAnchor.GEOMETRIC_CENTER:
        return target_anchor_world
    if actor.dimensions_m is None:
        raise AssertionError("procedural actor dimensions were not validated")
    return (
        target_anchor_world[0],
        target_anchor_world[1],
        target_anchor_world[2] + 0.5 * actor.dimensions_m[2],
    )


def _define_procedural_actor(
    stage: Usd.Stage,
    actor: RestaurantScenarioActor,
    *,
    prim_path: str,
    target_anchor_world: tuple[float, float, float],
    world_yaw_radians: float,
) -> WorldBounds3D:
    dimensions = actor.dimensions_m
    if dimensions is None:
        raise AssertionError("procedural actor dimensions were not validated")
    center = _procedural_center(actor, target_anchor_world)

    if actor.geometry is ScenarioGeometry.CUBE:
        shape = UsdGeom.Cube.Define(stage, prim_path)
        shape.CreateSizeAttr(1.0)
        shape.CreateExtentAttr(
            [Gf.Vec3f(-0.5), Gf.Vec3f(0.5)]
        )
        scale = dimensions
    elif actor.geometry is ScenarioGeometry.CYLINDER:
        shape = UsdGeom.Cylinder.Define(stage, prim_path)
        shape.CreateAxisAttr(UsdGeom.Tokens.z)
        shape.CreateRadiusAttr(0.5)
        shape.CreateHeightAttr(1.0)
        shape.CreateExtentAttr(
            [
                Gf.Vec3f(-0.5, -0.5, -0.5),
                Gf.Vec3f(0.5, 0.5, 0.5),
            ]
        )
        scale = dimensions
    elif actor.geometry is ScenarioGeometry.CAPSULE:
        shape = UsdGeom.Capsule.Define(stage, prim_path)
        shape.CreateAxisAttr(UsdGeom.Tokens.z)
        shape.CreateRadiusAttr(0.5)
        shape.CreateHeightAttr(1.0)
        shape.CreateExtentAttr(
            [Gf.Vec3f(-0.5, -0.5, -1.0), Gf.Vec3f(0.5, 0.5, 1.0)]
        )
        scale = (dimensions[0], dimensions[1], 0.5 * dimensions[2])
    elif actor.geometry is ScenarioGeometry.CONE:
        shape = UsdGeom.Cone.Define(stage, prim_path)
        shape.CreateAxisAttr(UsdGeom.Tokens.z)
        shape.CreateRadiusAttr(0.5)
        shape.CreateHeightAttr(1.0)
        shape.CreateExtentAttr(
            [
                Gf.Vec3f(-0.5, -0.5, -0.5),
                Gf.Vec3f(0.5, 0.5, 0.5),
            ]
        )
        scale = dimensions
    else:
        raise AssertionError(f"unsupported procedural geometry: {actor.geometry}")

    shape.CreateDisplayColorAttr([Gf.Vec3f(*actor.display_color_rgb)])
    shape.AddTranslateOp().Set(Gf.Vec3d(*center))
    shape.AddRotateZOp().Set(math.degrees(world_yaw_radians))
    shape.AddScaleOp().Set(Gf.Vec3f(*scale))
    return read_world_bounds(stage, prim_path)


def _place_existing_actor(
    stage: Usd.Stage,
    actor: RestaurantScenarioActor,
    *,
    target_anchor_world: tuple[float, float, float],
) -> tuple[str, WorldBounds3D]:
    prim_path = actor.existing_prim_path
    if prim_path is None:
        raise AssertionError("existing actor path was not validated")
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"scenario actor prim does not exist: {prim_path}")
    original_bounds = read_world_bounds(stage, prim_path)
    original_anchor = _anchor_from_bounds(
        original_bounds,
        actor.placement_anchor,
    )
    original_origin = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    ).Transform(Gf.Vec3d(0.0))
    target_origin = tuple(
        float(original_origin[index])
        + target_anchor_world[index]
        - original_anchor[index]
        for index in range(3)
    )
    placement = place_existing_prim_world_origin(
        stage,
        prim_path=prim_path,
        target_world_origin=target_origin,
    )
    return prim_path, placement.placed_bounds


def author_restaurant_risk_scenario(
    stage: Usd.Stage,
    scenario: RestaurantRiskScenarioDefinition,
    *,
    robot_pose_world_xy_yaw: object,
    root_prim_path: str = DEFAULT_SCENARIO_ROOT_PATH,
) -> AuthoredRestaurantRiskScenario:
    """Author one scenario in the session layer and keep every actor static.

    The planar robot pose transforms catalog ``+x forward, +y left`` positions
    and configured velocities into world coordinates.  Catalog Z coordinates
    remain world heights so a robot chassis offset is never mistaken for a
    support, landing, or fall height.
    """

    if not isinstance(stage, Usd.Stage):
        raise TypeError("stage must be a pxr.Usd.Stage")
    if not isinstance(scenario, RestaurantRiskScenarioDefinition):
        raise TypeError("scenario must be a RestaurantRiskScenarioDefinition")
    pose = _finite_planar_pose(robot_pose_world_xy_yaw)
    root_path = _validate_root_path(root_prim_path)
    if stage.GetPrimAtPath(root_path).IsValid():
        raise RuntimeError(f"scenario authoring root already exists: {root_path}")
    session_layer = stage.GetSessionLayer()
    if session_layer is None:
        raise RuntimeError("The USD stage has no session layer")
    root_before = stage.GetRootLayer().ExportToString()

    authored: list[AuthoredRestaurantScenarioActor] = []
    with Usd.EditContext(stage, session_layer):
        UsdGeom.Xform.Define(stage, root_path)
        for actor in scenario.actors:
            if not Sdf.Path.IsValidIdentifier(actor.actor_id):
                raise ValueError(
                    "actor_id must be a valid USD identifier for authoring: "
                    f"{actor.actor_id!r}"
                )
            target_anchor = _robot_position_to_world(
                actor.position_robot_xyz_m,
                pose,
            )
            if actor.geometry is ScenarioGeometry.EXISTING_PRIM:
                prim_path, bounds = _place_existing_actor(
                    stage,
                    actor,
                    target_anchor_world=target_anchor,
                )
            else:
                prim_path = f"{root_path}/{actor.actor_id}"
                bounds = _define_procedural_actor(
                    stage,
                    actor,
                    prim_path=prim_path,
                    target_anchor_world=target_anchor,
                    world_yaw_radians=pose[2],
                )
            authored.append(
                AuthoredRestaurantScenarioActor(
                    actor_id=actor.actor_id,
                    prim_path=prim_path,
                    geometry=actor.geometry,
                    placement_anchor=actor.placement_anchor,
                    target_anchor_world_xyz_m=target_anchor,
                    placed_anchor_world_xyz_m=_anchor_from_bounds(
                        bounds,
                        actor.placement_anchor,
                    ),
                    bounds_world=bounds,
                    configured_velocity_world_xy_mps=(
                        _robot_velocity_to_world(
                            actor.linear_velocity_robot_xy_mps,
                            pose[2],
                        )
                    ),
                )
            )

    if stage.GetRootLayer().ExportToString() != root_before:
        raise RuntimeError("scenario authoring unexpectedly changed the root layer")
    return AuthoredRestaurantRiskScenario(
        scenario_id=scenario.scenario_id,
        factor=scenario.factor,
        root_prim_path=root_path,
        robot_pose_world_xy_yaw=pose,
        actors=tuple(authored),
        runtime_sensor_claims=scenario.runtime_sensor_claims,
    )

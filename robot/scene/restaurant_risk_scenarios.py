"""Reusable restaurant scenario geometry with an explicit oracle boundary.

The catalog defines what should be placed in the scene and what may be read
after a trial for scoring.  Authored actor roles, semantics, trajectories, and
cue labels are never runtime perception evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from numbers import Real

from robot.risk.cra import PhysicalRiskFactor


RESTAURANT_RISK_SCENARIO_SCHEMA_VERSION = 1


class ScenarioGeometry(str, Enum):
    """How an actor is represented in the session layer."""

    EXISTING_PRIM = "existing_prim"
    CUBE = "cube"
    CYLINDER = "cylinder"
    CAPSULE = "capsule"
    CONE = "cone"


class ScenarioAppearanceBoundary(str, Enum):
    """Whether appearance is a real scene asset or only a procedural proxy."""

    RENDERED_ASSET = "rendered_asset"
    PROCEDURAL_ORACLE_PROXY = "procedural_oracle_proxy"


class ScenarioPlacementAnchor(str, Enum):
    """Which physical point an actor position identifies."""

    GEOMETRIC_CENTER = "geometric_center"
    BOUNDS_BASE_CENTER = "bounds_base_center"


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value.strip()


def _finite_tuple(value: object, size: int, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be an iterable of numbers")
    try:
        items = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError(f"{name} must be an iterable of numbers") from error
    if len(items) != size:
        raise ValueError(f"{name} must contain {size} values")
    result = []
    for item in items:
        if isinstance(item, bool) or not isinstance(item, Real):
            raise TypeError(f"{name} must contain real numbers")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{name} must contain finite values")
        result.append(number)
    return tuple(result)


@dataclass(frozen=True)
class RestaurantScenarioActor:
    """One authored actor positioned in the robot's initial planar frame.

    ``position_robot_xyz_m`` is interpreted using ``placement_anchor``.  This
    keeps a floor placement distinct from a geometric center for reused USD
    assets whose dimensions come from their rendered bounds.
    """

    actor_id: str
    role: str
    geometry: ScenarioGeometry
    position_robot_xyz_m: tuple[float, float, float]
    oracle_semantic_class: str
    appearance_boundary: ScenarioAppearanceBoundary
    dimensions_m: tuple[float, float, float] | None = None
    existing_prim_path: str | None = None
    linear_velocity_robot_xy_mps: tuple[float, float] = (0.0, 0.0)
    display_color_rgb: tuple[float, float, float] = (0.5, 0.5, 0.5)
    oracle_cues: tuple[str, ...] = ()
    placement_anchor: ScenarioPlacementAnchor = (
        ScenarioPlacementAnchor.GEOMETRIC_CENTER
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "actor_id", _text(self.actor_id, "actor_id"))
        object.__setattr__(self, "role", _text(self.role, "role"))
        object.__setattr__(
            self,
            "oracle_semantic_class",
            _text(self.oracle_semantic_class, "oracle_semantic_class"),
        )
        object.__setattr__(self, "geometry", ScenarioGeometry(self.geometry))
        object.__setattr__(
            self,
            "appearance_boundary",
            ScenarioAppearanceBoundary(self.appearance_boundary),
        )
        object.__setattr__(
            self,
            "placement_anchor",
            ScenarioPlacementAnchor(self.placement_anchor),
        )
        position = _finite_tuple(
            self.position_robot_xyz_m,
            3,
            "position_robot_xyz_m",
        )
        velocity = _finite_tuple(
            self.linear_velocity_robot_xy_mps,
            2,
            "linear_velocity_robot_xy_mps",
        )
        color = _finite_tuple(self.display_color_rgb, 3, "display_color_rgb")
        if position[2] < 0:
            raise ValueError("actor position height cannot be negative")
        if any(not 0 <= item <= 1 for item in color):
            raise ValueError("display_color_rgb must lie in [0, 1]")
        if self.geometry is ScenarioGeometry.EXISTING_PRIM:
            if self.dimensions_m is not None:
                raise ValueError("existing prim actors cannot prescribe dimensions")
            path = _text(self.existing_prim_path, "existing_prim_path")
            if not path.startswith("/"):
                raise ValueError("existing_prim_path must be an absolute USD path")
            object.__setattr__(self, "existing_prim_path", path)
        else:
            if self.existing_prim_path is not None:
                raise ValueError("procedural actors cannot reference an existing prim")
            dimensions = _finite_tuple(self.dimensions_m, 3, "dimensions_m")
            if any(item <= 0 for item in dimensions):
                raise ValueError("procedural actor dimensions must be positive")
            object.__setattr__(self, "dimensions_m", dimensions)
        cues = tuple(_text(item, "oracle cue") for item in self.oracle_cues)
        object.__setattr__(self, "position_robot_xyz_m", position)
        object.__setattr__(self, "linear_velocity_robot_xy_mps", velocity)
        object.__setattr__(self, "display_color_rgb", color)
        object.__setattr__(self, "oracle_cues", cues)

    @property
    def moves(self) -> bool:
        return math.hypot(*self.linear_velocity_robot_xy_mps) > 0


@dataclass(frozen=True)
class RestaurantRiskScenarioDefinition:
    """One factor-focused scene definition and its evaluation boundary."""

    scenario_id: str
    factor: PhysicalRiskFactor
    description: str
    actors: tuple[RestaurantScenarioActor, ...]
    oracle_scoring_fields: tuple[str, ...]
    sensor_requirements: tuple[str, ...]
    limitations: tuple[str, ...]
    runtime_sensor_claims: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "scenario_id", _text(self.scenario_id, "scenario_id")
        )
        object.__setattr__(self, "factor", PhysicalRiskFactor(self.factor))
        object.__setattr__(
            self, "description", _text(self.description, "description")
        )
        actors = tuple(self.actors)
        if not actors or any(
            not isinstance(item, RestaurantScenarioActor) for item in actors
        ):
            raise ValueError("actors must contain RestaurantScenarioActor values")
        actor_ids = tuple(item.actor_id for item in actors)
        if len(set(actor_ids)) != len(actor_ids):
            raise ValueError("scenario actor IDs must be unique")
        object.__setattr__(self, "actors", actors)
        for name in (
            "oracle_scoring_fields",
            "sensor_requirements",
            "limitations",
            "runtime_sensor_claims",
        ):
            values = tuple(_text(item, name) for item in getattr(self, name))
            object.__setattr__(self, name, values)

    @property
    def requires_actor_motion(self) -> bool:
        return any(actor.moves for actor in self.actors)

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "factor": self.factor.value,
            "description": self.description,
            "actors": [
                {
                    "actor_id": actor.actor_id,
                    "role": actor.role,
                    "geometry": actor.geometry.value,
                    "position_robot_xyz_m": actor.position_robot_xyz_m,
                    "placement_anchor": actor.placement_anchor.value,
                    "dimensions_m": actor.dimensions_m,
                    "existing_prim_path": actor.existing_prim_path,
                    "linear_velocity_robot_xy_mps": (
                        actor.linear_velocity_robot_xy_mps
                    ),
                    "display_color_rgb": actor.display_color_rgb,
                    "oracle_semantic_class": actor.oracle_semantic_class,
                    "appearance_boundary": actor.appearance_boundary.value,
                    "oracle_cues": actor.oracle_cues,
                }
                for actor in self.actors
            ],
            "oracle_scoring_fields": self.oracle_scoring_fields,
            "sensor_requirements": self.sensor_requirements,
            "limitations": self.limitations,
            "runtime_sensor_claims": self.runtime_sensor_claims,
            "requires_actor_motion": self.requires_actor_motion,
        }


_EXISTING_CHAIR_PATH = (
    "/World/Environment/Restaurant/assets_restaurant/"
    "Assets_Banquettes_Right/Geo_TableSet_09/Chair_Array"
)


def primary_restaurant_risk_scenarios() -> tuple[
    RestaurantRiskScenarioDefinition, ...
]:
    """Return the fixed primary scenario catalog in evaluation order."""

    procedural = ScenarioAppearanceBoundary.PROCEDURAL_ORACLE_PROXY
    return (
        RestaurantRiskScenarioDefinition(
            "chair",
            PhysicalRiskFactor.POTENTIAL,
            "ordinary chair on floor",
            (
                RestaurantScenarioActor(
                    "chair",
                    "ordinary_obstacle",
                    ScenarioGeometry.EXISTING_PRIM,
                    (1.7, 0.0, 0.0),
                    "chair",
                    ScenarioAppearanceBoundary.RENDERED_ASSET,
                    existing_prim_path=_EXISTING_CHAIR_PATH,
                    placement_anchor=(
                        ScenarioPlacementAnchor.BOUNDS_BASE_CENTER
                    ),
                ),
            ),
            ("clearance_m",),
            ("camera--LiDAR association", "mass evidence"),
            ("chair fall height remains unavailable on a continuous floor",),
        ),
        RestaurantRiskScenarioDefinition(
            "crossing_person",
            PhysicalRiskFactor.KINETIC,
            "person crossing nominal route",
            (
                RestaurantScenarioActor(
                    "crossing_person_proxy",
                    "crossing_actor",
                    ScenarioGeometry.CAPSULE,
                    (2.2, -1.2, 0.85),
                    "person",
                    procedural,
                    dimensions_m=(0.50, 0.50, 1.70),
                    linear_velocity_robot_xy_mps=(0.0, 0.60),
                    display_color_rgb=(0.10, 0.65, 0.80),
                    oracle_cues=("authored crossing trajectory",),
                ),
            ),
            ("trajectory_robot_xy_m", "speed_mps", "clearance_m"),
            ("LiDAR tracking", "camera person detection", "route interaction"),
            (
                "procedural capsule is not sensor evidence of a person",
                "authored velocity is scoring truth only",
            ),
        ),
        RestaurantRiskScenarioDefinition(
            "elevated_object",
            PhysicalRiskFactor.POTENTIAL,
            "object above a localized landing surface",
            (
                RestaurantScenarioActor(
                    "support",
                    "support_surface",
                    ScenarioGeometry.CUBE,
                    (2.0, 0.0, 0.40),
                    "support_fixture",
                    procedural,
                    dimensions_m=(1.00, 0.70, 0.80),
                    display_color_rgb=(0.35, 0.25, 0.15),
                ),
                RestaurantScenarioActor(
                    "elevated_target",
                    "falling_object_candidate",
                    ScenarioGeometry.CUBE,
                    (2.0, 0.0, 0.95),
                    "object",
                    procedural,
                    dimensions_m=(0.25, 0.25, 0.30),
                    display_color_rgb=(0.85, 0.20, 0.12),
                    oracle_cues=("authored support and landing heights",),
                ),
            ),
            (
                "support_height_m",
                "landing_height_m",
                "support_to_landing_fall_height_m",
            ),
            (
                "LiDAR support surface",
                "LiDAR landing surface",
                "harmful-fall interaction",
                "mass evidence",
            ),
            ("object height is not fall height",),
        ),
        RestaurantRiskScenarioDefinition(
            "exposed_edge",
            PhysicalRiskFactor.CONTACT,
            "fixed-label exposed blade edge",
            (
                RestaurantScenarioActor(
                    "exposed_edge_proxy",
                    "dangerous_part",
                    ScenarioGeometry.CUBE,
                    (2.0, 0.0, 0.75),
                    "exposed_edge",
                    procedural,
                    dimensions_m=(0.55, 0.05, 0.10),
                    display_color_rgb=(0.70, 0.72, 0.75),
                    oracle_cues=("authored exposed-edge geometry",),
                ),
            ),
            ("edge_pose_robot_xyz_m", "clearance_m"),
            (
                "part segmentation",
                "physical sharpness",
                "material hardness",
                "3D route orientation",
            ),
            (
                "proxy geometry is not physical sharpness or hardness",
                "current trained baseline supports exposed edges only",
            ),
        ),
        RestaurantRiskScenarioDefinition(
            "carried_hot_soup",
            PhysicalRiskFactor.THERMAL,
            "open carried soup container with a visible-plume cue",
            (
                RestaurantScenarioActor(
                    "open_container_proxy",
                    "container",
                    ScenarioGeometry.CYLINDER,
                    (2.0, 0.0, 0.65),
                    "open_container",
                    procedural,
                    dimensions_m=(0.34, 0.34, 0.30),
                    display_color_rgb=(0.25, 0.28, 0.32),
                    oracle_cues=("authored open-container role",),
                ),
                RestaurantScenarioActor(
                    "visible_plume_proxy",
                    "plume_visualization",
                    ScenarioGeometry.CONE,
                    (2.0, 0.0, 1.10),
                    "visible_plume",
                    procedural,
                    dimensions_m=(0.28, 0.28, 0.60),
                    display_color_rgb=(0.80, 0.85, 0.90),
                    oracle_cues=("authored visible-plume proxy",),
                ),
            ),
            ("plume_pose_robot_xyz_m", "container_pose_robot_xyz_m"),
            (
                "plume segmentation",
                "source context",
                "plume--container association",
                "route interaction",
            ),
            (
                "procedural plume is oracle visualization, not measured steam",
                "temperature and burn severity remain unavailable",
            ),
        ),
    )


def restaurant_risk_scenario(
    scenario_id: str,
) -> RestaurantRiskScenarioDefinition:
    """Look up one primary scenario by normalized identifier."""

    key = "_".join(_text(scenario_id, "scenario_id").casefold().split())
    for scenario in primary_restaurant_risk_scenarios():
        if scenario.scenario_id == key:
            return scenario
    raise KeyError(f"unknown restaurant risk scenario: {scenario_id!r}")


def restaurant_risk_scenario_manifest() -> dict[str, object]:
    """Return a JSON-compatible catalog with no runtime sensor assertions."""

    return {
        "schema_version": RESTAURANT_RISK_SCENARIO_SCHEMA_VERSION,
        "coordinate_frame": "robot initial frame: +x forward, +y left, +z up",
        "oracle_boundary": (
            "authored roles, semantics, trajectories, and cues are offline scoring "
            "truth only and are never supplied to runtime perception or control"
        ),
        "scenarios": [
            item.to_dict() for item in primary_restaurant_risk_scenarios()
        ],
    }

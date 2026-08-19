"""Session-only placement and validation for controlled scene obstacles.

The helpers in this module are intended for repeatable simulation experiments.
They reposition an existing rendered USD prim in the stage's anonymous session
layer, so the source restaurant scene is never modified.  Ground-truth bounds
returned by the placement helper are suitable for independent validation and
emergency monitoring, but must not be supplied to perception or control.
"""

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np
from pxr import Gf, Usd, UsdGeom


def _finite_real(value: object, *, name: str) -> float:
    """Return one finite real value while rejecting booleans."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _finite_vector(value: object, *, size: int, name: str) -> np.ndarray:
    """Return one finite vector with a fixed size."""

    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},).")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values.")
    return np.array(result, dtype=np.float64, copy=True)


@dataclass(frozen=True)
class WorldBounds3D:
    """Axis-aligned world bounds for one controlled obstacle."""

    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]

    def __post_init__(self) -> None:
        minimum = _finite_vector(self.minimum, size=3, name="minimum")
        maximum = _finite_vector(self.maximum, size=3, name="maximum")
        if np.any(maximum < minimum):
            raise ValueError("maximum cannot be smaller than minimum.")
        object.__setattr__(self, "minimum", tuple(float(v) for v in minimum))
        object.__setattr__(self, "maximum", tuple(float(v) for v in maximum))

    @property
    def center(self) -> tuple[float, float, float]:
        """Return the center of the world-aligned bounds."""

        return tuple(
            0.5 * (lower + upper)
            for lower, upper in zip(self.minimum, self.maximum, strict=True)
        )

    @property
    def size(self) -> tuple[float, float, float]:
        """Return the size of the bounds along world X, Y, and Z."""

        return tuple(
            upper - lower
            for lower, upper in zip(self.minimum, self.maximum, strict=True)
        )


@dataclass(frozen=True)
class ControlledObstaclePlacement:
    """Diagnostics from one session-layer obstacle placement."""

    prim_path: str
    original_world_origin: tuple[float, float, float]
    target_world_origin: tuple[float, float, float]
    original_bounds: WorldBounds3D
    placed_bounds: WorldBounds3D

    @property
    def world_translation(self) -> tuple[float, float, float]:
        """Return the applied world-space translation."""

        return tuple(
            target - original
            for original, target in zip(
                self.original_world_origin,
                self.target_world_origin,
                strict=True,
            )
        )


@dataclass(frozen=True)
class PlanarFootprint:
    """Asymmetric rectangular footprint in a robot body frame."""

    front_extent: float
    rear_extent: float
    left_extent: float
    right_extent: float

    def __post_init__(self) -> None:
        for name in (
            "front_extent",
            "rear_extent",
            "left_extent",
            "right_extent",
        ):
            value = _finite_real(getattr(self, name), name=name)
            if value <= 0.0:
                raise ValueError(f"{name} must be greater than zero.")
            object.__setattr__(self, name, value)


def read_world_bounds(
    stage: Usd.Stage,
    prim_path: str,
) -> WorldBounds3D:
    """Return rendered world bounds for one existing USD prim."""

    prim = _get_valid_prim(stage, prim_path)
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
            UsdGeom.Tokens.proxy,
        ],
        useExtentsHint=True,
    )
    aligned = cache.ComputeWorldBound(prim).ComputeAlignedBox()
    if aligned.IsEmpty():
        raise RuntimeError(f"USD prim has no renderable world bounds: {prim_path}")
    return WorldBounds3D(
        minimum=tuple(float(v) for v in aligned.GetMin()),
        maximum=tuple(float(v) for v in aligned.GetMax()),
    )


def place_existing_prim_world_origin(
    stage: Usd.Stage,
    *,
    prim_path: str,
    target_world_origin: object,
    tolerance: float = 1e-5,
) -> ControlledObstaclePlacement:
    """Move an existing prim through a session-layer translation override.

    Rotation, scale, materials, geometry, and hierarchy are preserved.  The
    target is the prim origin, not the center of its axis-aligned bounds.
    """

    if not isinstance(stage, Usd.Stage):
        raise TypeError("stage must be a pxr.Usd.Stage.")
    prim = _get_valid_prim(stage, prim_path)
    target = _finite_vector(
        target_world_origin,
        size=3,
        name="target_world_origin",
    )
    resolved_tolerance = _finite_real(tolerance, name="tolerance")
    if resolved_tolerance <= 0.0:
        raise ValueError("tolerance must be greater than zero.")

    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise RuntimeError(f"USD prim is not transformable: {prim_path}")
    translate_ops = [
        operation
        for operation in xformable.GetOrderedXformOps()
        if operation.GetOpType() == UsdGeom.XformOp.TypeTranslate
        and not operation.IsInverseOp()
    ]
    if len(translate_ops) != 1:
        raise RuntimeError(
            "Controlled placement requires exactly one non-inverse translate "
            f"operation; found {len(translate_ops)} for {prim_path}."
        )

    time_code = Usd.TimeCode.Default()
    world_transform = xformable.ComputeLocalToWorldTransform(time_code)
    original_origin = np.asarray(
        world_transform.Transform(Gf.Vec3d(0.0, 0.0, 0.0)),
        dtype=np.float64,
    )
    original_bounds = read_world_bounds(stage, prim_path)
    parent = prim.GetParent()
    parent_to_world = UsdGeom.Xformable(parent).ComputeLocalToWorldTransform(
        time_code
    )
    target_parent = parent_to_world.GetInverse().Transform(
        Gf.Vec3d(*target.tolist())
    )

    previous_edit_target = stage.GetEditTarget()
    session_layer = stage.GetSessionLayer()
    if session_layer is None:
        raise RuntimeError("The USD stage has no session layer.")
    try:
        stage.SetEditTarget(session_layer)
        if not translate_ops[0].Set(Gf.Vec3d(*target_parent), time_code):
            raise RuntimeError(
                f"Failed to author the session transform for {prim_path}."
            )
    finally:
        stage.SetEditTarget(previous_edit_target)

    placed_transform = xformable.ComputeLocalToWorldTransform(time_code)
    placed_origin = np.asarray(
        placed_transform.Transform(Gf.Vec3d(0.0, 0.0, 0.0)),
        dtype=np.float64,
    )
    origin_error = float(np.linalg.norm(placed_origin - target))
    if origin_error > resolved_tolerance:
        raise RuntimeError(
            "Session obstacle placement missed its target by "
            f"{origin_error:.6e} m."
        )
    placed_bounds = read_world_bounds(stage, prim_path)
    return ControlledObstaclePlacement(
        prim_path=prim_path,
        original_world_origin=tuple(float(v) for v in original_origin),
        target_world_origin=tuple(float(v) for v in placed_origin),
        original_bounds=original_bounds,
        placed_bounds=placed_bounds,
    )


def oriented_footprint_clearance_to_bounds(
    robot_pose_world: object,
    *,
    footprint: PlanarFootprint,
    obstacle_bounds: WorldBounds3D,
) -> float:
    """Return planar separation between a robot rectangle and obstacle AABB.

    Zero means the conservative rectangles touch or overlap.  This function is
    intended only for an independent experiment abort guard and reporting.
    """

    if not isinstance(footprint, PlanarFootprint):
        raise TypeError("footprint must be a PlanarFootprint.")
    if not isinstance(obstacle_bounds, WorldBounds3D):
        raise TypeError("obstacle_bounds must be a WorldBounds3D.")
    pose = _finite_vector(robot_pose_world, size=3, name="robot_pose_world")
    local_corners = np.array(
        [
            [footprint.front_extent, footprint.left_extent],
            [-footprint.rear_extent, footprint.left_extent],
            [-footprint.rear_extent, -footprint.right_extent],
            [footprint.front_extent, -footprint.right_extent],
        ],
        dtype=np.float64,
    )
    cosine = math.cos(float(pose[2]))
    sine = math.sin(float(pose[2]))
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    robot_polygon = local_corners @ rotation.T + pose[:2]
    minimum = obstacle_bounds.minimum
    maximum = obstacle_bounds.maximum
    obstacle_polygon = np.array(
        [
            [minimum[0], minimum[1]],
            [maximum[0], minimum[1]],
            [maximum[0], maximum[1]],
            [minimum[0], maximum[1]],
        ],
        dtype=np.float64,
    )
    return _convex_polygon_clearance(robot_polygon, obstacle_polygon)


def line_segment_intersects_planar_bounds(
    start_world_xy: object,
    end_world_xy: object,
    *,
    obstacle_bounds: WorldBounds3D,
) -> bool:
    """Return whether a planar segment intersects an obstacle's world AABB."""

    if not isinstance(obstacle_bounds, WorldBounds3D):
        raise TypeError("obstacle_bounds must be a WorldBounds3D.")
    start = _finite_vector(start_world_xy, size=2, name="start_world_xy")
    end = _finite_vector(end_world_xy, size=2, name="end_world_xy")
    minimum = np.asarray(obstacle_bounds.minimum[:2], dtype=np.float64)
    maximum = np.asarray(obstacle_bounds.maximum[:2], dtype=np.float64)
    direction = end - start
    lower_time = 0.0
    upper_time = 1.0
    for axis in range(2):
        if abs(float(direction[axis])) <= 1e-12:
            if start[axis] < minimum[axis] or start[axis] > maximum[axis]:
                return False
            continue
        first = (minimum[axis] - start[axis]) / direction[axis]
        second = (maximum[axis] - start[axis]) / direction[axis]
        entry = min(float(first), float(second))
        exit_ = max(float(first), float(second))
        lower_time = max(lower_time, entry)
        upper_time = min(upper_time, exit_)
        if lower_time > upper_time:
            return False
    return True


def _get_valid_prim(stage: Usd.Stage, prim_path: str) -> Usd.Prim:
    """Return one valid prim after stable input validation."""

    if not isinstance(stage, Usd.Stage):
        raise TypeError("stage must be a pxr.Usd.Stage.")
    if not isinstance(prim_path, str) or not prim_path.startswith("/"):
        raise ValueError("prim_path must be an absolute USD prim path.")
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"USD prim does not exist: {prim_path}")
    return prim


def _convex_polygon_clearance(first: np.ndarray, second: np.ndarray) -> float:
    """Return Euclidean separation for two convex planar polygons."""

    if _polygons_overlap(first, second):
        return 0.0
    minimum_distance = math.inf
    for vertices, edges in ((first, second), (second, first)):
        for vertex in vertices:
            for edge_index in range(edges.shape[0]):
                edge_start = edges[edge_index]
                edge_end = edges[(edge_index + 1) % edges.shape[0]]
                minimum_distance = min(
                    minimum_distance,
                    _point_to_segment_distance(vertex, edge_start, edge_end),
                )
    return float(minimum_distance)


def _polygons_overlap(first: np.ndarray, second: np.ndarray) -> bool:
    """Return overlap under the separating-axis theorem."""

    for polygon in (first, second):
        for index in range(polygon.shape[0]):
            edge = polygon[(index + 1) % polygon.shape[0]] - polygon[index]
            axis = np.array([-edge[1], edge[0]], dtype=np.float64)
            norm = float(np.linalg.norm(axis))
            if norm <= 1e-12:
                continue
            axis /= norm
            first_projection = first @ axis
            second_projection = second @ axis
            if (
                float(np.max(first_projection))
                < float(np.min(second_projection)) - 1e-12
                or float(np.max(second_projection))
                < float(np.min(first_projection)) - 1e-12
            ):
                return False
    return True


def _point_to_segment_distance(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> float:
    """Return Euclidean distance from one point to a finite segment."""

    edge = end - start
    squared_length = float(edge @ edge)
    if squared_length <= 1e-24:
        return float(np.linalg.norm(point - start))
    parameter = float((point - start) @ edge) / squared_length
    parameter = min(1.0, max(0.0, parameter))
    projection = start + parameter * edge
    return float(np.linalg.norm(point - projection))

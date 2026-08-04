"""Measure Nova Carter's collision footprint in the chassis frame.

This one-time diagnostic loads the composed restaurant robot scene, finds
every enabled USD collider below ``/World/Robot``, and unions their bounds in
the ``/World/Robot/chassis_link`` coordinate frame. The resulting planar
extents can be copied directly into ``RectangularRobotFootprint``.

The report is based on collision geometry rather than render geometry. It
therefore represents the geometry used by physics for the selected Nova
Carter USD variants.

Run from the project root:

    python -u scripts/diagnostics/measure_nova_carter_footprint.py
"""

import math
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from robot.config import (
    ROBOT_BASE_PRIM_PATH,
    ROBOT_PRIM_PATH,
    ROBOT_SCENE_PATH,
)

from isaacsim import SimulationApp


simulation_app = SimulationApp({"headless": True})


from pxr import Usd, UsdGeom, UsdPhysics

from robot.scene.loader import load_robot_scene


RECOMMENDED_SAFETY_MARGIN_METERS = 0.05
MAX_REASONABLE_COLLIDER_SPAN_METERS = 2.0
BOUND_TOLERANCE = 1e-12


@dataclass(frozen=True)
class ColliderBounds:
    """One collider's axis-aligned chassis-frame bounds."""

    prim_path: str
    minimum_xyz: tuple[float, float, float]
    maximum_xyz: tuple[float, float, float]


@dataclass(frozen=True)
class CollisionFootprintReport:
    """Union of all enabled robot collision bounds."""

    colliders: tuple[ColliderBounds, ...]
    rejected_colliders: tuple[ColliderBounds, ...]
    minimum_xyz: tuple[float, float, float]
    maximum_xyz: tuple[float, float, float]

    @property
    def front_extent(self) -> float:
        """Return the maximum ``+X`` collision extent."""

        return max(0.0, self.maximum_xyz[0])

    @property
    def rear_extent(self) -> float:
        """Return the maximum ``-X`` collision extent magnitude."""

        return max(0.0, -self.minimum_xyz[0])

    @property
    def left_extent(self) -> float:
        """Return the maximum ``+Y`` collision extent."""

        return max(0.0, self.maximum_xyz[1])

    @property
    def right_extent(self) -> float:
        """Return the maximum ``-Y`` collision extent magnitude."""

        return max(0.0, -self.minimum_xyz[1])

    @property
    def length(self) -> float:
        """Return the total collision length along chassis X."""

        return self.maximum_xyz[0] - self.minimum_xyz[0]

    @property
    def width(self) -> float:
        """Return the total collision width along chassis Y."""

        return self.maximum_xyz[1] - self.minimum_xyz[1]

    @property
    def height(self) -> float:
        """Return the total collision height along chassis Z."""

        return self.maximum_xyz[2] - self.minimum_xyz[2]


def measure_collision_footprint(
    stage,
    *,
    robot_prim_path: str = ROBOT_PRIM_PATH,
    chassis_prim_path: str = ROBOT_BASE_PRIM_PATH,
) -> CollisionFootprintReport:
    """Return enabled collider bounds expressed in the chassis frame."""

    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    chassis_prim = stage.GetPrimAtPath(chassis_prim_path)
    if not robot_prim.IsValid():
        raise ValueError(f"Robot prim does not exist: {robot_prim_path}")
    if not chassis_prim.IsValid():
        raise ValueError(
            f"Chassis prim does not exist: {chassis_prim_path}"
        )

    bounds_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.proxy,
        ],
        # Compute directly from the leaf geometry. Some Nova Carter caster
        # cylinders carry stale 100-unit authored extents that must not be
        # substituted through model extent hints.
        useExtentsHint=False,
    )
    collider_bounds: list[ColliderBounds] = []
    rejected_bounds: list[ColliderBounds] = []

    for prim in _traverse_with_instance_proxies(robot_prim):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue

        collision_api = UsdPhysics.CollisionAPI(prim)
        collision_enabled = collision_api.GetCollisionEnabledAttr().Get()
        if collision_enabled is False:
            continue

        relative_bound = bounds_cache.ComputeRelativeBound(
            prim,
            chassis_prim,
        )
        aligned_range = relative_bound.ComputeAlignedBox()
        if aligned_range.IsEmpty():
            continue

        minimum = _finite_vector(
            aligned_range.GetMin(),
            context=f"minimum bound for {prim.GetPath()}",
        )
        maximum = _finite_vector(
            aligned_range.GetMax(),
            context=f"maximum bound for {prim.GetPath()}",
        )
        bounds = ColliderBounds(
            prim_path=str(prim.GetPath()),
            minimum_xyz=minimum,
            maximum_xyz=maximum,
        )
        spans = tuple(
            maximum[axis] - minimum[axis]
            for axis in range(3)
        )
        if max(spans) > MAX_REASONABLE_COLLIDER_SPAN_METERS:
            rejected_bounds.append(bounds)
            continue

        collider_bounds.append(bounds)

    if not collider_bounds:
        raise RuntimeError(
            f"No enabled collision prims were found below {robot_prim_path}."
        )

    minimum_xyz = tuple(
        min(bounds.minimum_xyz[axis] for bounds in collider_bounds)
        for axis in range(3)
    )
    maximum_xyz = tuple(
        max(bounds.maximum_xyz[axis] for bounds in collider_bounds)
        for axis in range(3)
    )
    report = CollisionFootprintReport(
        colliders=tuple(collider_bounds),
        rejected_colliders=tuple(rejected_bounds),
        minimum_xyz=minimum_xyz,
        maximum_xyz=maximum_xyz,
    )
    _validate_planar_extents(report)
    return report


def _traverse_with_instance_proxies(root_prim):
    """Yield a subtree while descending through USD instances."""

    pending_prims = [root_prim]
    traversal_predicate = Usd.TraverseInstanceProxies()

    while pending_prims:
        prim = pending_prims.pop()
        yield prim
        children = prim.GetFilteredChildren(traversal_predicate)
        pending_prims.extend(reversed(children))


def _finite_vector(value, *, context: str) -> tuple[float, float, float]:
    """Return one finite three-dimensional vector."""

    components = tuple(float(component) for component in value)
    if len(components) != 3 or not all(
        math.isfinite(component) for component in components
    ):
        raise ValueError(f"Invalid {context}: {components}")
    return components


def _validate_planar_extents(report: CollisionFootprintReport) -> None:
    """Reject empty or reference-point-excluding collision envelopes."""

    if report.length <= BOUND_TOLERANCE:
        raise RuntimeError("Measured collision length is zero.")
    if report.width <= BOUND_TOLERANCE:
        raise RuntimeError("Measured collision width is zero.")
    if report.front_extent <= BOUND_TOLERANCE:
        raise RuntimeError(
            "Collision bounds do not extend in chassis +X. Verify the "
            "configured chassis frame."
        )
    if report.rear_extent <= BOUND_TOLERANCE:
        raise RuntimeError(
            "Collision bounds do not extend in chassis -X. Verify the "
            "configured chassis frame."
        )
    if report.left_extent <= BOUND_TOLERANCE:
        raise RuntimeError(
            "Collision bounds do not extend in chassis +Y. Verify the "
            "configured chassis frame."
        )
    if report.right_extent <= BOUND_TOLERANCE:
        raise RuntimeError(
            "Collision bounds do not extend in chassis -Y. Verify the "
            "configured chassis frame."
        )


def print_report(report: CollisionFootprintReport) -> None:
    """Print measured bounds and copy-ready footprint configuration."""

    print("\n" + "=" * 70)
    print("NOVA CARTER COLLISION FOOTPRINT REPORT")
    print("=" * 70)
    print("\nMeasurement basis:")
    print(f"  Robot subtree:       {ROBOT_PRIM_PATH}")
    print(f"  Reference frame:     {ROBOT_BASE_PRIM_PATH}")
    print(f"  Enabled colliders:   {len(report.colliders)}")
    print(
        "  Rejected outliers:   "
        f"{len(report.rejected_colliders)}"
    )

    print("\nChassis-frame collision bounds:")
    print(
        "  X interval:          "
        f"[{report.minimum_xyz[0]:.6f}, "
        f"{report.maximum_xyz[0]:.6f}] m"
    )
    print(
        "  Y interval:          "
        f"[{report.minimum_xyz[1]:.6f}, "
        f"{report.maximum_xyz[1]:.6f}] m"
    )
    print(
        "  Z interval:          "
        f"[{report.minimum_xyz[2]:.6f}, "
        f"{report.maximum_xyz[2]:.6f}] m"
    )

    print("\nPlanar footprint:")
    print(f"  Front extent (+X):   {report.front_extent:.6f} m")
    print(f"  Rear extent (-X):    {report.rear_extent:.6f} m")
    print(f"  Left extent (+Y):    {report.left_extent:.6f} m")
    print(f"  Right extent (-Y):   {report.right_extent:.6f} m")
    print(f"  Total length:        {report.length:.6f} m")
    print(f"  Total width:         {report.width:.6f} m")
    print(f"  Collision height:    {report.height:.6f} m")

    print("\nCopy-ready safety configuration:")
    print("  RectangularRobotFootprint(")
    print(f"      front_extent={report.front_extent:.6f},")
    print(f"      rear_extent={report.rear_extent:.6f},")
    print(f"      left_extent={report.left_extent:.6f},")
    print(f"      right_extent={report.right_extent:.6f},")
    print(
        "      safety_margin="
        f"{RECOMMENDED_SAFETY_MARGIN_METERS:.6f},"
    )
    print("  )")

    print("\nCollider details:")
    for collider in report.colliders:
        print(f"  - {collider.prim_path}")
        print(
            "      min="
            f"({collider.minimum_xyz[0]:.4f}, "
            f"{collider.minimum_xyz[1]:.4f}, "
            f"{collider.minimum_xyz[2]:.4f})"
        )
        print(
            "      max="
            f"({collider.maximum_xyz[0]:.4f}, "
            f"{collider.maximum_xyz[1]:.4f}, "
            f"{collider.maximum_xyz[2]:.4f})"
        )

    if report.rejected_colliders:
        print("\nRejected collider-bound outliers:")
        print(
            "  These bounds exceeded the loose "
            f"{MAX_REASONABLE_COLLIDER_SPAN_METERS:.1f} m corruption "
            "guard and were not used."
        )
        for collider in report.rejected_colliders:
            spans = tuple(
                collider.maximum_xyz[axis]
                - collider.minimum_xyz[axis]
                for axis in range(3)
            )
            print(f"  - {collider.prim_path}")
            print(
                "      span="
                f"({spans[0]:.4f}, {spans[1]:.4f}, "
                f"{spans[2]:.4f}) m"
            )

    print("\nInterpretation:")
    if report.rejected_colliders:
        print(
            "  [PASS WITH WARNINGS] Valid robot-scale colliders produced "
            "the footprint."
        )
        print(
            "  Review the rejected collider list before fixing the "
            "configuration."
        )
    else:
        print(
            "  [PASS] Enabled collision geometry produced a valid "
            "footprint."
        )
    print("  The 0.05 m margin is a policy choice, not measured geometry.")
    print("  Keep the measured extents fixed for reproducible experiments.")
    print("=" * 70)


def main() -> None:
    """Load the scene, measure the robot, and print the report."""

    loaded_scene = load_robot_scene(
        ROBOT_SCENE_PATH,
        simulation_app,
    )
    report = measure_collision_footprint(loaded_scene.stage)
    print_report(report)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\n" + "=" * 70)
        print("ERROR DURING NOVA CARTER FOOTPRINT MEASUREMENT")
        print("=" * 70)
        print(error)
        traceback.print_exc()
    finally:
        simulation_app.close()

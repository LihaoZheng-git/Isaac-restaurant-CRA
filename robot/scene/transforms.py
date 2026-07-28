"""USD coordinate-frame transformation helpers.

Import this module only after ``SimulationApp`` has been created.
"""

import math
import numpy as np
import isaacsim.core.experimental.utils.transform as transform_utils
from pxr import Gf, Usd, UsdGeom

def get_prim_world_transform(
    stage: Usd.Stage,
    prim_path: str,
) -> Gf.Matrix4d:
    """Return a prim's transform from its local frame to the world frame."""
    prim = stage.GetPrimAtPath(prim_path)

    if not prim.IsValid():
        raise ValueError(f"USD prim does not exist: {prim_path}")

    xformable = UsdGeom.Xformable(prim)

    if not xformable:
        raise ValueError(f"USD prim is not transformable: {prim_path}")

    return xformable.ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )


def get_prim_planar_pose(
    stage: Usd.Stage,
    prim_path: str,
) -> tuple[float, float, float]:
    """Return a prim's world-frame planar pose as (x, y, yaw).

    The prim's local +X axis is treated as its forward direction.
    Yaw is measured counterclockwise around the world +Z axis.
    """
    world_transform = get_prim_world_transform(stage, prim_path)

    translation = world_transform.ExtractTranslation()
    forward_world = world_transform.TransformDir(
        Gf.Vec3d(1.0, 0.0, 0.0)
    )

    if math.hypot(forward_world[0], forward_world[1]) < 1e-9:
        raise ValueError(
            f"Cannot determine planar heading for prim: {prim_path}"
        )

    yaw = math.atan2(forward_world[1], forward_world[0])

    return (
        float(translation[0]),
        float(translation[1]),
        float(yaw),
    )

def compute_relative_prim_transform(
    stage: Usd.Stage,
    *,
    source_prim_path: str,
    target_prim_path: str,
) -> np.ndarray:
    """Compute a column-vector transform between two USD prim frames.

    The returned matrix transforms points from the source local frame
    into the target local frame:

        point_target = target_from_source @ point_source

    For the CBF pipeline:

        source = PandarXT-32 LiDAR
        target = Nova Carter chassis

    Args:
        stage:
            Currently loaded USD stage.
        source_prim_path:
            Absolute path of the source coordinate frame.
        target_prim_path:
            Absolute path of the target coordinate frame.

    Returns:
        A contiguous ``float64`` homogeneous matrix with shape ``(4, 4)``.
    """

    if stage is None:
        raise ValueError("stage cannot be None.")

    source_xformable = _get_xformable(
        stage,
        source_prim_path,
        role="source",
    )

    target_xformable = _get_xformable(
        stage,
        target_prim_path,
        role="target",
    )

    time_code = Usd.TimeCode.Default()

    source_to_world_usd = np.asarray(
        source_xformable.ComputeLocalToWorldTransform(
            time_code
        ),
        dtype=np.float64,
    )

    target_to_world_usd = np.asarray(
        target_xformable.ComputeLocalToWorldTransform(
            time_code
        ),
        dtype=np.float64,
    )

    target_from_source = (
        transform_utils.compute_relative_transform(
            source_to_world_usd,
            target_to_world_usd,
        )
    )

    target_from_source = np.asarray(
        target_from_source,
        dtype=np.float64,
    )

    if target_from_source.shape != (4, 4):
        raise RuntimeError(
            "Isaac Sim returned an unexpected relative-transform "
            f"shape: {target_from_source.shape}."
        )

    if not np.isfinite(target_from_source).all():
        raise RuntimeError(
            "Isaac Sim returned a non-finite relative transform."
        )

    expected_last_row = np.array(
        [0.0, 0.0, 0.0, 1.0],
        dtype=np.float64,
    )

    if not np.allclose(
        target_from_source[3],
        expected_last_row,
        atol=1e-8,
        rtol=0.0,
    ):
        raise RuntimeError(
            "The relative transform is not in the expected "
            "column-vector homogeneous format."
        )

    return np.ascontiguousarray(
        target_from_source,
        dtype=np.float64,
    )


def _get_xformable(
    stage: Usd.Stage,
    prim_path: str,
    *,
    role: str,
) -> UsdGeom.Xformable:
    """Get and validate one transformable USD prim."""

    if not isinstance(prim_path, str) or not prim_path.startswith("/"):
        raise ValueError(
            f"{role}_prim_path must be an absolute USD prim path."
        )

    prim = stage.GetPrimAtPath(prim_path)

    if not prim.IsValid():
        raise RuntimeError(
            f"The {role} prim was not found: {prim_path}"
        )

    xformable = UsdGeom.Xformable(prim)

    if not xformable:
        raise RuntimeError(
            f"The {role} prim is not transformable: {prim_path}"
        )

    return xformable
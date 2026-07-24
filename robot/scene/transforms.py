"""USD coordinate-frame transformation helpers.

Import this module only after ``SimulationApp`` has been created.
"""

import numpy as np
import isaacsim.core.experimental.utils.transform as transform_utils
from pxr import Usd, UsdGeom


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
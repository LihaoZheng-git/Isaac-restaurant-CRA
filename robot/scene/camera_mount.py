"""Explicit experimental RGB mount pitch, without changing lens calibration."""

import math

import numpy as np


def pitched_camera_transform(parent_from_camera, upward_degrees):
    """Pitch about the USD camera's local right (+X) axis at its own origin.

    USD cameras look along -Z with +Y up. This is a physical mount hypothesis,
    not image rectification or a change to the stock Nova Carter configuration.
    """
    original = np.asarray(parent_from_camera, dtype=float)
    if (
        original.shape != (4, 4)
        or not np.isfinite(original).all()
        or not np.allclose(original[3], [0, 0, 0, 1])
        or not math.isfinite(upward_degrees)
        or not 0 <= upward_degrees <= 25
    ):
        raise ValueError("finite homogeneous camera transform and pitch 0..25 required")
    angle = math.radians(upward_degrees)
    c, s = math.cos(angle), math.sin(angle)
    pitch = np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])
    return original @ pitch


def apply_experimental_camera_pitch(
    stage, camera_path, upward_degrees, base_path, *, raise_metres=0.0
):
    """Session-only mount edit; call before initializing the camera sensor.

    Intrinsics are untouched. Callers must read actual extrinsics after this
    edit and retain the audit alongside results. Import after SimulationApp.
    """
    from pxr import Gf, Usd, UsdGeom

    if not math.isfinite(raise_metres) or not 0 <= raise_metres <= 0.6:
        raise ValueError("experimental RGB mount raise must be within 0...6 metres")
    camera = UsdGeom.Xformable(stage.GetPrimAtPath(camera_path))
    base = UsdGeom.Xformable(stage.GetPrimAtPath(base_path))
    if not camera or not base:
        raise ValueError("camera and base prims must exist")
    before_local = np.asarray(camera.GetLocalTransformation(), dtype=float).T
    before_world = np.asarray(
        camera.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    ).T
    base_world = np.asarray(base.ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
    new_local = pitched_camera_transform(before_local, upward_degrees)
    # Resolve base +Z into the camera parent's coordinates; rotation and height
    # are physical mount changes, not a post-hoc projection adjustment.
    parent_world = before_world @ np.linalg.inv(before_local)
    parent_from_base = np.linalg.solve(parent_world, base_world)
    new_local[:3, 3] += parent_from_base[:3, 2] * raise_metres
    if upward_degrees or raise_metres:
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            camera.MakeMatrixXform().Set(Gf.Matrix4d(new_local.T.tolist()))
    after_world = np.asarray(
        camera.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    ).T
    expected_origin = before_world[:3, 3] + base_world[:3, 2] * raise_metres
    if not np.allclose(expected_origin, after_world[:3, 3], atol=1e-7):
        raise RuntimeError(
            "camera optical origin differs from the declared mount change"
        )
    before_base = np.linalg.solve(base_world, before_world)
    after_base = np.linalg.solve(base_world, after_world)
    if upward_degrees and after_base[2, 2] >= before_base[2, 2]:
        raise RuntimeError("camera -Z forward axis did not tilt upward in base frame")
    return {
        "upward_degrees": upward_degrees,
        "raise_metres": raise_metres,
        "camera_prim": camera_path,
        "before_parent_from_camera": before_local.tolist(),
        "after_parent_from_camera": new_local.tolist(),
        "before_base_forward": (-before_base[:3, 2]).tolist(),
        "after_base_forward": (-after_base[:3, 2]).tolist(),
        "before_base_origin": before_base[:3, 3].tolist(),
        "after_base_origin": after_base[:3, 3].tolist(),
        "scope": "explicit experimental camera mount; session only; unchanged lens; extrinsics re-read after edit; not stock-camera validation",
    }

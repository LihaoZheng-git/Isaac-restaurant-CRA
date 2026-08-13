"""Read authored Isaac Sim camera calibration from a USD prim.

Import this module only after ``SimulationApp`` has been created.
"""

from numbers import Real

from pxr import Usd

from robot.perception.camera_lidar_fusion import FThetaCameraCalibration


def read_fisheye_polynomial_calibration(
    stage: Usd.Stage,
    *,
    camera_prim_path: str,
    output_resolution: tuple[int, int],
) -> FThetaCameraCalibration:
    """Return the camera's authored f-theta calibration.

    Args:
        stage: Loaded USD stage.
        camera_prim_path: Absolute path of the camera prim.
        output_resolution: Render-product ``(height, width)``.
    """

    if stage is None:
        raise ValueError("stage cannot be None.")
    if not isinstance(camera_prim_path, str) or not camera_prim_path.startswith("/"):
        raise ValueError("camera_prim_path must be an absolute USD prim path.")
    if len(output_resolution) != 2:
        raise ValueError("output_resolution must contain (height, width).")
    output_height = int(output_resolution[0])
    output_width = int(output_resolution[1])
    prim = stage.GetPrimAtPath(camera_prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Camera prim was not found: {camera_prim_path}")
    projection_type = _read_attribute(prim, "cameraProjectionType")
    if str(projection_type) != "fisheyePolynomial":
        raise RuntimeError(
            "Camera--LiDAR fusion currently requires the authored "
            "fisheyePolynomial model, received "
            f"{projection_type!r}."
        )
    return FThetaCameraCalibration(
        nominal_width=int(_read_real_attribute(prim, "fthetaWidth")),
        nominal_height=int(_read_real_attribute(prim, "fthetaHeight")),
        optical_center_x=_read_real_attribute(prim, "fthetaCx"),
        optical_center_y=_read_real_attribute(prim, "fthetaCy"),
        maximum_fov_degrees=_read_real_attribute(prim, "fthetaMaxFov"),
        polynomial_coefficients=tuple(
            _read_real_attribute(prim, f"fthetaPoly{suffix}")
            for suffix in "ABCDE"
        ),
        output_width=output_width,
        output_height=output_height,
    )


def _read_attribute(prim: Usd.Prim, name: str) -> object:
    """Read one required authored USD attribute."""

    attribute = prim.GetAttribute(name)
    if not attribute.IsValid() or not attribute.HasAuthoredValueOpinion():
        raise RuntimeError(
            f"Camera calibration attribute is missing or unauthored: {name}"
        )
    value = attribute.Get()
    if value is None:
        raise RuntimeError(f"Camera calibration attribute has no value: {name}")
    return value


def _read_real_attribute(prim: Usd.Prim, name: str) -> float:
    """Read one required finite-number-compatible USD attribute."""

    value = _read_attribute(prim, name)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise RuntimeError(f"Camera calibration attribute is not numeric: {name}")
    return float(value)

"""Reusable RGB-camera interface for Nova Carter.

This module wraps Nova Carter's existing camera prims using the Isaac Sim
6.0 experimental RTX camera API. It does not create a replacement camera
and does not modify the camera's authored pose or calibration.

Create SimulationApp before importing this module.

Resolution follows the OpenCV/NumPy convention:
    (height, width)
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import omni.timeline
import omni.usd
from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera
from pxr import Usd, UsdGeom


from robot.config import ROBOT_PRIM_PATH


FRONT_HAWK_LEFT_CAMERA_PATH = (
    f"{ROBOT_PRIM_PATH}/chassis_link/sensors/"
    "front_hawk/left/camera_left"
)

FRONT_HAWK_RIGHT_CAMERA_PATH = (
    f"{ROBOT_PRIM_PATH}/chassis_link/sensors/"
    "front_hawk/right/camera_right"
)

FRONT_OWL_CAMERA_PATH = (
    f"{ROBOT_PRIM_PATH}/chassis_link/sensors/front_owl/camera"
)

CAMERA_PRIM_PATHS = {
    "front_hawk_left": FRONT_HAWK_LEFT_CAMERA_PATH,
    "front_hawk_right": FRONT_HAWK_RIGHT_CAMERA_PATH,
    "front_owl": FRONT_OWL_CAMERA_PATH,
}

DEFAULT_CAMERA_PRIM_PATH = FRONT_HAWK_LEFT_CAMERA_PATH
DEFAULT_RGB_RESOLUTION = (480, 640)


@dataclass(frozen=True)
class CameraFrameInfo:
    """Metadata for the most recently retrieved RGB frame."""

    camera_prim_path: str
    read_index: int
    simulation_time_seconds: float
    height: int
    width: int
    channels: int
    dtype: str


def find_camera_prim_paths(
    robot_prim_path: str = ROBOT_PRIM_PATH,
) -> list[str]:
    """Return all USD Camera prim paths below the robot."""

    stage = omni.usd.get_context().get_stage()

    if stage is None:
        raise RuntimeError("No USD stage is currently open.")

    robot_prim = stage.GetPrimAtPath(robot_prim_path)

    if not robot_prim.IsValid():
        raise RuntimeError(
            f"No robot prim was found at {robot_prim_path}."
        )

    camera_paths = [
        str(prim.GetPath())
        for prim in Usd.PrimRange(robot_prim)
        if prim.IsA(UsdGeom.Camera)
    ]

    return sorted(camera_paths)


def _camera_label(camera_prim_path: str) -> str:
    """Return a readable name for a known Nova Carter camera."""

    for name, path in CAMERA_PRIM_PATHS.items():
        if camera_prim_path == path:
            return name

    return camera_prim_path.rsplit("/", maxsplit=1)[-1]


def _to_numpy(data: Any) -> np.ndarray:
    """Copy a camera buffer into a NumPy array."""

    if isinstance(data, np.ndarray):
        return np.array(data, copy=True)

    if hasattr(data, "numpy"):
        return np.array(data.numpy(), copy=True)

    return np.array(data, copy=True)


class NovaCarterCamera:
    """Read RGB images from one existing Nova Carter camera.

    The default camera is the left image sensor of the front Hawk stereo
    camera. Pass another camera path to use the right Hawk or front Owl.

    This wrapper deliberately preserves:

    - The camera's existing transform
    - The camera's existing optical parameters
    - The camera's existing distortion model
    - The camera's existing tick rate
    """

    def __init__(
        self,
        *,
        camera_prim_path: str = DEFAULT_CAMERA_PRIM_PATH,
        resolution: tuple[int, int] = DEFAULT_RGB_RESOLUTION,
    ) -> None:
        if not camera_prim_path.startswith("/"):
            raise ValueError(
                "camera_prim_path must be an absolute USD prim path."
            )

        if len(resolution) != 2:
            raise ValueError(
                "resolution must contain exactly (height, width)."
            )

        height = int(resolution[0])
        width = int(resolution[1])

        if height <= 0 or width <= 0:
            raise ValueError(
                "Camera height and width must both be positive."
            )

        self.camera_prim_path = camera_prim_path
        self.resolution = (height, width)

        self._rtx_camera: RtxCamera | None = None
        self._camera_sensor: CameraSensor | None = None

        self._last_frame_info: CameraFrameInfo | None = None
        self._last_annotator_info: dict[str, Any] = {}
        self._read_index = 0

    @property
    def is_initialized(self) -> bool:
        """Whether the RTX camera capture pipeline has been created."""

        return (
            self._rtx_camera is not None
            and self._camera_sensor is not None
        )

    def initialize(self) -> None:
        """Wrap the existing camera and create its RGB render product."""

        if self.is_initialized:
            return

        stage = omni.usd.get_context().get_stage()

        if stage is None:
            raise RuntimeError("No USD stage is currently open.")

        camera_prim = stage.GetPrimAtPath(self.camera_prim_path)

        if not camera_prim.IsValid():
            available_paths = find_camera_prim_paths()

            available_text = (
                "\n".join(
                    f"  - {path}" for path in available_paths
                )
                if available_paths
                else "  (no Camera prims found under /World/Robot)"
            )

            raise RuntimeError(
                "The configured camera prim was not found:\n"
                f"  {self.camera_prim_path}\n\n"
                "Available robot cameras:\n"
                f"{available_text}"
            )

        if not camera_prim.IsA(UsdGeom.Camera):
            raise RuntimeError(
                "The configured prim exists but is not a USD Camera:\n"
                f"  {self.camera_prim_path}"
            )

        # tick_rate=None preserves the rate authored in Nova Carter.
        #
        # reset_xform_op_properties=False is important because this camera
        # is already mounted and calibrated inside the robot asset.
        self._rtx_camera = RtxCamera(
            self.camera_prim_path,
            tick_rate=None,
            reset_xform_op_properties=False,
        )

        self._camera_sensor = CameraSensor(
            self._rtx_camera,
            resolution=self.resolution,
            annotators=["rgb"],
        )

        height, width = self.resolution

        print("\nNova Carter RGB camera initialized:")
        print(f"  Camera name: {self.camera_name}")
        print(f"  Camera prim: {self.camera_prim_path}")
        print(
            "  Resolution:  "
            f"{height} x {width} (height x width)"
        )
        print("  Camera pose: preserved")
        print("  Calibration: preserved")
        print("  Tick rate:   preserved")

    @property
    def camera_name(self) -> str:
        """Readable identifier for the selected camera."""

        return _camera_label(self.camera_prim_path)

    def get_rgb(self) -> np.ndarray:
        """Return the latest RGB image as a NumPy array.

        Returns:
            RGB image with shape ``(height, width, 3)``.

        Raises:
            RuntimeError: If the camera is not initialized or no rendered
                frame is available yet.
        """

        if self._camera_sensor is None:
            raise RuntimeError(
                "Camera is not initialized. Call initialize() first."
            )

        data, annotator_info = self._camera_sensor.get_data("rgb")

        if data is None:
            raise RuntimeError(
                "No RGB frame is available yet. Advance the simulation "
                "and try again."
            )

        rgb = _to_numpy(data)

        # Some camera implementations return a one-element batch.
        if rgb.ndim == 4 and rgb.shape[0] == 1:
            rgb = rgb[0]

        if rgb.ndim != 3:
            raise RuntimeError(
                "Unexpected RGB frame shape from CameraSensor: "
                f"{rgb.shape}"
            )

        # Be tolerant if a backend returns RGBA for the RGB annotator.
        if rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]
        elif rgb.shape[2] != 3:
            raise RuntimeError(
                "Expected 3 or 4 image channels, but received "
                f"{rgb.shape[2]}."
            )

        rgb = np.ascontiguousarray(rgb)

        self._read_index += 1
        self._last_annotator_info = dict(annotator_info or {})

        timeline = omni.timeline.get_timeline_interface()

        self._last_frame_info = CameraFrameInfo(
            camera_prim_path=self.camera_prim_path,
            read_index=self._read_index,
            simulation_time_seconds=float(
                timeline.get_current_time()
            ),
            height=int(rgb.shape[0]),
            width=int(rgb.shape[1]),
            channels=int(rgb.shape[2]),
            dtype=str(rgb.dtype),
        )

        return rgb

    def get_frame_info(self) -> CameraFrameInfo:
        """Return metadata for the latest successful RGB read."""

        if self._last_frame_info is None:
            raise RuntimeError(
                "No RGB frame has been read yet. Call get_rgb() first."
            )

        return self._last_frame_info

    def get_annotator_info(self) -> dict[str, Any]:
        """Return annotator metadata from the latest RGB read."""

        return dict(self._last_annotator_info)
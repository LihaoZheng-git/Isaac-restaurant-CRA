"""Reusable RGB-camera interface for Nova Carter.

Import this module only after ``SimulationApp`` has been created.
Camera resolutions follow the OpenCV/NumPy convention:
``(height, width)``.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from pxr import Usd, UsdGeom


DEFAULT_ROBOT_PRIM_PATH = "/World/Robot"
DEFAULT_FRONT_LEFT_CAMERA_PRIM_PATH = (
    "/World/Robot/chassis_link/sensors/front_hawk/left/camera_left"
)
DEFAULT_RGB_RESOLUTION = (480, 640)
DEFAULT_RT_SUBFRAMES = 16


@dataclass(frozen=True)
class CameraFrameInfo:
    """Metadata recorded for the most recently retrieved RGB frame."""

    camera_prim_path: str
    read_index: int
    simulation_time_seconds: float
    height: int
    width: int
    channels: int
    dtype: str


def find_camera_prim_paths(
    robot_prim_path: str = DEFAULT_ROBOT_PRIM_PATH,
) -> list[str]:
    """Return all USD Camera prim paths below the robot prim."""

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


class NovaCarterCamera:
    """Capture RGB frames from one of Nova Carter's existing cameras."""

    def __init__(
        self,
        *,
        camera_prim_path: str = DEFAULT_FRONT_LEFT_CAMERA_PRIM_PATH,
        resolution: tuple[int, int] = DEFAULT_RGB_RESOLUTION,
        rt_subframes: int = DEFAULT_RT_SUBFRAMES,
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

        if rt_subframes <= 0:
            raise ValueError("rt_subframes must be positive.")

        self.camera_prim_path = camera_prim_path
        self.resolution = (height, width)
        self.rt_subframes = int(rt_subframes)

        self._render_product: Any | None = None
        self._rgb_annotator: Any | None = None
        self._last_frame_info: CameraFrameInfo | None = None
        self._last_annotator_info: dict[str, Any] = {}
        self._read_index = 0

    @property
    def is_initialized(self) -> bool:
        """Whether the Replicator RGB pipeline has been created."""

        return (
            self._render_product is not None
            and self._rgb_annotator is not None
        )

    def initialize(self) -> None:
        """Validate the camera prim and create its RGB capture pipeline."""

        if self.is_initialized:
            return

        stage = omni.usd.get_context().get_stage()

        if stage is None:
            raise RuntimeError("No USD stage is currently open.")

        camera_prim = stage.GetPrimAtPath(self.camera_prim_path)

        if not camera_prim.IsValid():
            available_paths = find_camera_prim_paths()
            available_text = (
                "\n".join(f"  - {path}" for path in available_paths)
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

        # Nova Carter's authored Hawk cameras need autotriggering to remain
        # synchronized with Replicator's explicit capture steps.
        if camera_prim.HasAttribute("omni:sensor:tickRate"):
            camera_prim.GetAttribute("omni:sensor:tickRate").Set(0.0)

        # A zero f-stop disables depth of field and produces a sharper image.
        if camera_prim.HasAttribute("fStop"):
            camera_prim.GetAttribute("fStop").Set(0.0)

        # Manual GUI Play should control the simulation timeline. RGB capture
        # is requested explicitly in get_rgb().
        rep.orchestrator.set_capture_on_play(False)

        height, width = self.resolution
        render_product = rep.create.render_product(
            self.camera_prim_path,
            resolution=(width, height),
            name="nova_carter_front_left_rgb",
            force_new=True,
        )
        rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb_annotator.attach(render_product)

        # Retain both objects for the lifetime of this wrapper. Releasing the
        # render product would leave the annotator without a Hydra texture.
        self._render_product = render_product
        self._rgb_annotator = rgb_annotator

        print("\nNova Carter RGB camera initialized:")
        print(f"  Camera prim: {self.camera_prim_path}")
        print(
            "  Resolution:  "
            f"{height} x {width} (height x width)"
        )
        print("  Tick rate:   autotrigger every simulation frame")
        print(f"  RT subframes: {self.rt_subframes}")

    def get_rgb(self, *, copy: bool = True) -> np.ndarray:
        """Render and return one RGB frame as a ``(H, W, 3)`` array."""

        if self._rgb_annotator is None:
            raise RuntimeError(
                "Camera is not initialized. Call initialize() first."
            )

        # Match NVIDIA's Nova Carter capture sequence: pause the running
        # timeline, execute one explicit Replicator capture step, read the
        # annotator while the timeline remains paused, and then resume.
        timeline = omni.timeline.get_timeline_interface()
        was_playing = timeline.is_playing()

        if was_playing:
            timeline.pause()

        try:
            rep.orchestrator.step(rt_subframes=self.rt_subframes)

            annotator_data = self._rgb_annotator.get_data(
                device="cpu",
                do_array_copy=True,
                use_legacy_structure=False,
            )
        finally:
            if was_playing:
                timeline.play()

        annotator_info: dict[str, Any] = {}

        if isinstance(annotator_data, dict):
            frame = annotator_data.get("data")

            nested_info = annotator_data.get("info")
            if isinstance(nested_info, dict):
                annotator_info.update(nested_info)

            annotator_info.update(
                {
                    key: value
                    for key, value in annotator_data.items()
                    if key not in {"data", "info"}
                }
            )
        else:
            # This fallback supports Replicator builds that still return the
            # legacy raw-array structure despite the option above.
            frame = annotator_data

        if frame is None:
            raise RuntimeError(
                "No RGB frame is available yet. Keep the simulation "
                "playing and try another rendered update."
            )

        rgb = np.asarray(frame)

        if rgb.size == 0:
            raise RuntimeError(
                "No RGB frame is available yet. Replicator returned an "
                "empty RGB array."
            )

        if rgb.ndim != 3:
            raise RuntimeError(
                "Unexpected RGB frame shape: "
                f"{rgb.shape}. Expected (height, width, channels)."
            )

        if rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]

        if rgb.shape[2] != 3:
            raise RuntimeError(
                "Unexpected RGB channel count: "
                f"{rgb.shape[2]}. Expected 3 or 4 channels."
            )

        if copy:
            rgb = np.ascontiguousarray(rgb).copy()

        self._read_index += 1
        self._last_annotator_info = annotator_info
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
        """Return metadata for the most recent successful ``get_rgb`` call."""

        if self._last_frame_info is None:
            raise RuntimeError(
                "No frame metadata is available. Call get_rgb() first."
            )

        return self._last_frame_info

    def get_annotator_info(self) -> dict[str, Any]:
        """Return metadata from the latest RGB annotator read."""

        return dict(self._last_annotator_info)
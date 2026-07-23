"""RTX LiDAR access for the Nova Carter robot.

This module wraps LiDAR sensors already included in the Nova Carter USD asset.
It does not create new sensors or modify their authored transforms, scan
profiles, accumulation settings, or tick rates.

The default sensor is the 3D Hesai PandarXT-32.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import omni.timeline
import omni.usd
from isaacsim.sensors.experimental.rtx import (
    Lidar,
    LidarSensor,
    parse_generic_model_output_data,
)
from isaacsim.sensors.experimental.rtx.generic_model_output import (
    CoordsType,
)


from robot.config import ROBOT_PRIM_PATH
from robot.sensor.lidar_geometry import (
    filter_valid_points,
    spherical_to_cartesian,
)


# ---------------------------------------------------------------------------
# Nova Carter LiDAR prim paths
# ---------------------------------------------------------------------------

PANDAR_XT32_LIDAR_PRIM_PATH = (
    f"{ROBOT_PRIM_PATH}/chassis_link/sensors/"
    "XT_32/PandarXT_32_10hz"
)

FRONT_RPLIDAR_PRIM_PATH = (
    f"{ROBOT_PRIM_PATH}/chassis_link/sensors/front_RPLidar/RPLidar_S2E"
)

REAR_RPLIDAR_PRIM_PATH = (
    f"{ROBOT_PRIM_PATH}/chassis_link/sensors/rear_RPLidar/RPLidar_S2E"
)

DEFAULT_LIDAR_PRIM_PATH = PANDAR_XT32_LIDAR_PRIM_PATH

GENERIC_MODEL_OUTPUT_ANNOTATOR = "generic-model-output"


@dataclass(frozen=True)
class LidarFrameInfo:
    """Metadata describing the latest successful LiDAR scan."""

    lidar_prim_path: str
    read_index: int
    simulation_time_seconds: float
    sensor_frame_id: int
    sensor_timestamp_nanoseconds: int
    source_point_count: int
    returned_point_count: int
    source_coordinate_type: str


def _copy_numeric_buffer(
    buffer: Any,
    element_count: int,
) -> np.ndarray:
    """Copy one GenericModelOutput numeric buffer into NumPy memory."""

    if element_count <= 0:
        return np.empty((0,), dtype=np.float32)

    if buffer is None:
        raise RuntimeError(
            "GenericModelOutput reported points, but one of its "
            "coordinate buffers is missing."
        )

    try:
        # GenericModelOutput normally exposes coordinate data as ctypes
        # pointers. Copying is important because Isaac Sim may reuse the
        # underlying sensor buffer on the next rendered update.
        array = np.ctypeslib.as_array(
            buffer,
            shape=(element_count,),
        )
    except (TypeError, ValueError):
        if hasattr(buffer, "numpy"):
            array = buffer.numpy()
        else:
            array = np.asarray(buffer)

    array = np.asarray(array).reshape(-1)

    if array.size < element_count:
        raise RuntimeError(
            "A GenericModelOutput coordinate buffer is shorter than "
            f"expected: expected {element_count}, received {array.size}."
        )

    return np.array(
        array[:element_count],
        dtype=np.float32,
        copy=True,
    )


def _coordinate_type_name(coordinate_type: Any) -> str:
    """Return a readable GenericModelOutput coordinate type."""

    try:
        return CoordsType(coordinate_type).name
    except (TypeError, ValueError):
        name = getattr(coordinate_type, "name", None)

        if name is not None:
            return str(name)

        return str(coordinate_type)


def _is_coordinate_type(
    coordinate_type: Any,
    expected_type: Any,
) -> bool:
    """Compare a parsed coordinate type with a CoordsType member."""

    if coordinate_type == expected_type:
        return True

    try:
        return CoordsType(coordinate_type) == expected_type
    except (TypeError, ValueError):
        return (
            _coordinate_type_name(coordinate_type).upper()
            == expected_type.name.upper()
        )




class NovaCarterLidar:
    """Read 3D point clouds from an existing Nova Carter RTX LiDAR.

    By default, this class uses the PandarXT-32 mounted on top of Nova
    Carter. Returned points are expressed in the LiDAR sensor coordinate
    frame, with units of meters.

    The wrapper deliberately preserves:

    - The sensor's existing transform
    - The sensor's existing LiDAR model and scan profile
    - The sensor's existing accumulation setting
    - The sensor's existing tick rate
    """

    def __init__(
        self,
        *,
        lidar_prim_path: str = DEFAULT_LIDAR_PRIM_PATH,
        filter_invalid_points: bool = True,
    ) -> None:
        if not lidar_prim_path.startswith("/"):
            raise ValueError(
                "lidar_prim_path must be an absolute USD prim path."
            )

        self.lidar_prim_path = lidar_prim_path
        self.filter_invalid_points = bool(
            filter_invalid_points
        )

        self._rtx_lidar: Lidar | None = None
        self._lidar_sensor: LidarSensor | None = None

        self._last_frame_info: LidarFrameInfo | None = None
        self._last_annotator_info: dict[str, Any] = {}
        self._read_index = 0

    @property
    def is_initialized(self) -> bool:
        """Whether the RTX LiDAR capture pipeline has been created."""

        return (
            self._rtx_lidar is not None
            and self._lidar_sensor is not None
        )

    @property
    def lidar_name(self) -> str:
        """Readable name of the selected LiDAR prim."""

        return self.lidar_prim_path.rsplit("/", maxsplit=1)[-1]

    def initialize(self) -> None:
        """Wrap the existing LiDAR and create its runtime pipeline."""

        if self.is_initialized:
            return

        stage = omni.usd.get_context().get_stage()

        if stage is None:
            raise RuntimeError("No USD stage is currently open.")

        lidar_prim = stage.GetPrimAtPath(
            self.lidar_prim_path
        )

        if not lidar_prim.IsValid():
            raise RuntimeError(
                "The configured LiDAR prim was not found:\n"
                f"  {self.lidar_prim_path}\n\n"
                "Run scripts/diagnostics/inspect_lidar.py to list "
                "the available Nova Carter LiDAR prims."
            )

        lidar_type = lidar_prim.GetTypeName()

        if lidar_type != "OmniLidar":
            raise RuntimeError(
                "The configured prim exists but is not an "
                "OmniLidar prim:\n"
                f"  Path: {self.lidar_prim_path}\n"
                f"  Type: {lidar_type or '<untyped>'}"
            )

        # accumulate_outputs=None preserves the value already authored
        # by the PandarXT-32 asset.
        #
        # tick_rate=None preserves its authored 10 Hz sensor rate.
        #
        # reset_xform_op_properties=False preserves its mounting pose
        # within Nova Carter.
        self._rtx_lidar = Lidar(
            self.lidar_prim_path,
            accumulate_outputs=None,
            aux_output_level="NONE",
            tick_rate=None,
            reset_xform_op_properties=False,
        )

        self._lidar_sensor = LidarSensor(
            self._rtx_lidar,
            annotators=[
                GENERIC_MODEL_OUTPUT_ANNOTATOR
            ],
        )

        tick_rate_attribute = lidar_prim.GetAttribute(
            "omni:sensor:tickRate"
        )
        scan_rate_attribute = lidar_prim.GetAttribute(
            "omni:sensor:Core:scanRateBaseHz"
        )
        accumulation_attribute = lidar_prim.GetAttribute(
            "omni:sensor:Core:accumulateOutputs"
        )

        tick_rate = (
            tick_rate_attribute.Get()
            if tick_rate_attribute.IsValid()
            else None
        )
        scan_rate = (
            scan_rate_attribute.Get()
            if scan_rate_attribute.IsValid()
            else None
        )
        accumulate_outputs = (
            accumulation_attribute.Get()
            if accumulation_attribute.IsValid()
            else None
        )

        print("\nNova Carter RTX LiDAR initialized:")
        print(f"  LiDAR name:       {self.lidar_name}")
        print(f"  LiDAR prim:       {self.lidar_prim_path}")
        print(f"  USD type:         {lidar_type}")
        print(f"  Tick rate:        {tick_rate}")
        print(f"  Scan rate:        {scan_rate}")
        print(f"  Accumulate scans: {accumulate_outputs}")
        print("  LiDAR pose:       preserved")
        print("  LiDAR profile:    preserved")
        print("  Point frame:      sensor")
        print("  Point units:      meters")

    def get_point_cloud(self) -> np.ndarray:
        """Return the latest completed point cloud as an ``(N, 3)`` array.

        Points use the LiDAR sensor coordinate frame:

        - ``+X`` points forward
        - ``+Y`` points left
        - ``+Z`` points upward

        Returns:
            A copied ``float32`` NumPy array with shape ``(N, 3)``.

        Raises:
            RuntimeError: If the LiDAR is not initialized or the latest
                rendered update does not contain a completed, nonempty scan.
        """

        if self._lidar_sensor is None:
            raise RuntimeError(
                "LiDAR is not initialized. Call initialize() first."
            )

        data, annotator_info = self._lidar_sensor.get_data(
            GENERIC_MODEL_OUTPUT_ANNOTATOR
        )

        if data is None:
            raise RuntimeError(
                "No LiDAR data is available yet. Advance the "
                "simulation while the timeline is playing and try again."
            )

        generic_output = parse_generic_model_output_data(
            data
        )

        if generic_output is None:
            raise RuntimeError(
                "The GenericModelOutput LiDAR buffer could not "
                "be parsed."
            )

        source_point_count = int(
            generic_output.numElements
        )

        if source_point_count <= 0:
            raise RuntimeError(
                "The latest LiDAR update contains no points. "
                "The PandarXT-32 produces a completed scan at 10 Hz, "
                "so advance the simulation and try again."
            )

        first = np.array(
            np.asarray(generic_output.x).reshape(-1)[
                :source_point_count
            ],
            dtype=np.float32,
            copy=True,
        )

        second = np.array(
            np.asarray(generic_output.y).reshape(-1)[
                :source_point_count
            ],
            dtype=np.float32,
            copy=True,
        )

        third = np.array(
            np.asarray(generic_output.z).reshape(-1)[
                :source_point_count
            ],
            dtype=np.float32,
            copy=True,
        )

        coordinate_type = generic_output.elementsCoordsType


        coordinate_type_name = _coordinate_type_name(
            coordinate_type
        )

        if _is_coordinate_type(
            coordinate_type,
            CoordsType.CARTESIAN,
        ):
            points = np.column_stack(
                (first, second, third)
            ).astype(np.float32, copy=False)

        elif _is_coordinate_type(
            coordinate_type,
            CoordsType.SPHERICAL,
        ):
            # For spherical GenericModelOutput:
            #
            #   x = azimuth in degrees
            #   y = elevation in degrees
            #   z = distance in meters
            points = spherical_to_cartesian(
                azimuth_degrees=first,
                elevation_degrees=second,
                distance_meters=third,
            )

        else:
            raise RuntimeError(
                "Unsupported LiDAR coordinate type in "
                f"GenericModelOutput: {coordinate_type_name}"
            )

        if self.filter_invalid_points:
            points = filter_valid_points(points)
        else:
            points = np.ascontiguousarray(
                points,
                dtype=np.float32,
            )

        if points.shape[0] == 0:
            raise RuntimeError(
                "The LiDAR scan contained data, but no valid XYZ "
                "points remained after filtering."
            )

        self._read_index += 1
        self._last_annotator_info = dict(
            annotator_info or {}
        )

        timeline = omni.timeline.get_timeline_interface()

        self._last_frame_info = LidarFrameInfo(
            lidar_prim_path=self.lidar_prim_path,
            read_index=self._read_index,
            simulation_time_seconds=float(
                timeline.get_current_time()
            ),
            sensor_frame_id=int(
                generic_output.frameId
            ),
            sensor_timestamp_nanoseconds=int(
                generic_output.timestampNs
            ),
            source_point_count=source_point_count,
            returned_point_count=int(points.shape[0]),
            source_coordinate_type=coordinate_type_name,
        )

        return points

    def get_frame_info(self) -> LidarFrameInfo:
        """Return metadata for the latest successful point-cloud read."""

        if self._last_frame_info is None:
            raise RuntimeError(
                "No LiDAR point cloud has been read yet. "
                "Call get_point_cloud() first."
            )

        return self._last_frame_info

    def get_annotator_info(self) -> dict[str, Any]:
        """Return annotator metadata from the latest successful read."""

        return dict(self._last_annotator_info)
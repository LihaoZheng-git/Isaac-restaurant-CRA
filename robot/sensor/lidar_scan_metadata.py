"""Copied RTX scan metadata; importing this module does not require Isaac.

GMO element times are offsets from its header timestamp. FrameAtTime uses
quaternions in (x, y, z, w) order. These values describe the simulator sensor,
not measured odometry, and are exposed for auditing before any deskew.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LidarScanPose:
    timestamp_nanoseconds: int
    position_m: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]


@dataclass(frozen=True)
class LidarScanMetadata:
    sensor_timestamp_nanoseconds: int
    frame_of_reference: str
    motion_compensation_state: str
    frame_start: LidarScanPose | None
    frame_end: LidarScanPose | None
    model_to_app_transform: tuple[float, ...] | None
    point_time_offsets_nanoseconds: np.ndarray | None
    unavailable_reasons: tuple[str, ...]

    def summary(self):
        """Small JSON-ready audit; point arrays stay in numeric snapshots."""

        def pose(value):
            if value is None:
                return None
            return {
                "timestamp_nanoseconds": value.timestamp_nanoseconds,
                "position_m": value.position_m,
                "orientation_xyzw": value.orientation_xyzw,
            }

        offsets = self.point_time_offsets_nanoseconds
        return {
            "timestamp_nanoseconds": self.sensor_timestamp_nanoseconds,
            "frame_of_reference": self.frame_of_reference,
            "motion_compensation_state": self.motion_compensation_state,
            "frame_start": pose(self.frame_start),
            "frame_end": pose(self.frame_end),
            "model_to_app_transform": self.model_to_app_transform,
            "offset_count": None if offsets is None else len(offsets),
            "offset_min_ns": None
            if offsets is None or not len(offsets)
            else int(offsets.min()),
            "offset_max_ns": None
            if offsets is None or not len(offsets)
            else int(offsets.max()),
            "unavailable_reasons": self.unavailable_reasons,
        }


def copy_scan_metadata(generic_output, valid_point_mask):
    """Own all metadata and apply the exact XYZ filter to per-point times.

    Missing metadata remains explicitly unavailable; it must not be replaced by
    read time, a zero offset, or an identity transform. Legacy point-cloud callers
    may still use the cloud without claiming time alignment or motion correction.
    """
    count = int(generic_output.numElements)
    valid = np.asarray(valid_point_mask)
    if valid.shape != (count,) or valid.dtype != np.bool_:
        raise ValueError("scan metadata requires the matching boolean XYZ mask")
    reasons = []

    def enum_name(name):
        value = getattr(generic_output, name, None)
        if value is None:
            reasons.append(f"{name} unavailable")
            return "UNAVAILABLE"
        return str(getattr(value, "name", value))

    def frame(name):
        value = getattr(generic_output, name, None)
        try:
            position = np.asarray(value.posM, dtype=float)
            orientation = np.asarray(value.orientation, dtype=float)
            timestamp = int(value.timestampNs)
            if (
                position.shape != (3,)
                or orientation.shape != (4,)
                or not np.isfinite(position).all()
                or not np.isfinite(orientation).all()
                or not np.isclose(np.linalg.norm(orientation), 1.0, atol=1e-4)
                or timestamp < 0
            ):
                raise ValueError("invalid frame pose")
            return LidarScanPose(timestamp, tuple(position), tuple(orientation))
        except (AttributeError, TypeError, ValueError):
            reasons.append(f"{name} unavailable or invalid")
            return None

    offsets = getattr(generic_output, "timeOffsetNs", None)
    if offsets is None:
        reasons.append("per-point time offsets unavailable")
    else:
        values = np.asarray(offsets).reshape(-1)
        if values.size < count or values.dtype.kind not in "iu":
            offsets = None
            reasons.append("per-point time offsets invalid or shorter than XYZ")
        else:
            offsets = np.array(values[:count][valid], dtype=np.int64, copy=True)
            offsets.setflags(write=False)
    transform = getattr(generic_output, "modelToAppTransform", None)
    if transform is not None:
        values = np.asarray(transform, dtype=float).reshape(-1)
        if values.shape == (16,) and np.isfinite(values).all():
            transform = tuple(values)
        else:
            transform = None
    if transform is None:
        reasons.append("model-to-app transform unavailable or invalid")
    reference = enum_name("frameOfReference")
    compensation = enum_name("motionCompensationState")
    start, end = frame("frameStart"), frame("frameEnd")
    return LidarScanMetadata(
        int(generic_output.timestampNs),
        reference,
        compensation,
        start,
        end,
        transform,
        offsets,
        tuple(reasons),
    )

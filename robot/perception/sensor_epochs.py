"""Align moving camera/LiDAR observations to their acquisition-time ego poses.

Pose references retain their provenance. No extrapolation, completion-time
timestamp substitution, or unavailable-to-zero conversion is permitted.
"""

from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from robot.perception.physical_attributes import EstimateStatus


def rigid_transform(value):
    matrix = np.array(value, dtype=float, copy=True)
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8)
        or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(matrix[:3, :3]), 1.0, atol=1e-6)
    ):
        raise ValueError("pose must be a finite proper rigid 4x4 transform")
    matrix.setflags(write=False)
    return matrix


def timestamp(value):
    if isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError("timestamp must be finite and nonnegative")
    return float(value)


@dataclass(frozen=True)
class EgoPoseSample:
    timestamp_seconds: float
    world_from_robot: np.ndarray
    status: EstimateStatus
    source: str

    def __post_init__(self):
        object.__setattr__(self, "timestamp_seconds", timestamp(self.timestamp_seconds))
        object.__setattr__(
            self, "world_from_robot", rigid_transform(self.world_from_robot)
        )
        status = EstimateStatus(self.status)
        if status in {EstimateStatus.UNAVAILABLE, EstimateStatus.FALLBACK}:
            raise ValueError(
                "an unavailable/fallback ego pose cannot align sensor evidence"
            )
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("ego pose source is required")
        object.__setattr__(self, "status", status)


def gravity_aligned_robot_transform(world_from_robot):
    """Rotate robot points into the scan-time heading frame with world Z up.

    The reference origin stays at the chassis. Pose provenance belongs to the
    caller; this geometry operation does not turn oracle poses into odometry.
    """
    pose = rigid_transform(world_from_robot)
    yaw = math.atan2(pose[1, 0], pose[0, 0])
    c, s = math.cos(yaw), math.sin(yaw)
    result = np.eye(4)
    result[:3, :3] = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1.0]]) @ pose[:3, :3]
    return rigid_transform(result)


class EgoPoseHistory:
    def __init__(self, *, maximum_samples=256, maximum_interpolation_gap_seconds=0.05):
        if (
            isinstance(maximum_samples, bool)
            or not isinstance(maximum_samples, int)
            or maximum_samples < 2
        ):
            raise ValueError("maximum_samples must be an integer >= 2")
        if (
            not math.isfinite(maximum_interpolation_gap_seconds)
            or maximum_interpolation_gap_seconds <= 0
        ):
            raise ValueError("maximum interpolation gap must be positive")
        self.maximum_samples = maximum_samples
        self.maximum_gap = float(maximum_interpolation_gap_seconds)
        self.samples = []

    def append(self, sample: EgoPoseSample):
        if not isinstance(sample, EgoPoseSample):
            raise TypeError("history requires EgoPoseSample")
        if (
            self.samples
            and sample.timestamp_seconds <= self.samples[-1].timestamp_seconds
        ):
            raise ValueError("ego pose samples must be strictly increasing")
        self.samples.append(sample)
        self.samples = self.samples[-self.maximum_samples :]

    def sample(self, at_seconds):
        at = timestamp(at_seconds)
        for sample in self.samples:
            if abs(at - sample.timestamp_seconds) <= 1e-9:
                return sample
        for left, right in zip(self.samples, self.samples[1:]):
            if left.timestamp_seconds < at < right.timestamp_seconds:
                gap = right.timestamp_seconds - left.timestamp_seconds
                if gap > self.maximum_gap:
                    raise ValueError("ego pose interpolation crosses an excessive gap")
                if left.status != right.status or left.source != right.source:
                    raise ValueError("ego pose provenance changed inside the interval")
                alpha = (at - left.timestamp_seconds) / gap
                matrix = np.eye(4)
                matrix[:3, 3] = (1 - alpha) * left.world_from_robot[
                    :3, 3
                ] + alpha * right.world_from_robot[:3, 3]
                rotations = Rotation.from_matrix(
                    np.stack(
                        (left.world_from_robot[:3, :3], right.world_from_robot[:3, :3])
                    )
                )
                matrix[:3, :3] = Slerp(
                    [left.timestamp_seconds, right.timestamp_seconds], rotations
                )(at).as_matrix()
                return EgoPoseSample(at, matrix, left.status, f"{left.source}")
        raise ValueError(
            "sensor timestamp is outside retained ego pose history; no extrapolation"
        )

    def velocity_world_xy(self, at_seconds, *, window_seconds=0.1):
        if not math.isfinite(window_seconds) or window_seconds <= 0:
            raise ValueError("velocity window must be positive")
        current = self.sample(at_seconds)
        previous = self.sample(at_seconds - window_seconds)
        if (current.status, current.source) != (previous.status, previous.source):
            raise ValueError("velocity window mixes ego pose provenance")
        return tuple(
            (current.world_from_robot[:2, 3] - previous.world_from_robot[:2, 3])
            / window_seconds
        )


@dataclass(frozen=True)
class AlignedSensorEpoch:
    lidar_pose: EgoPoseSample
    camera_pose: EgoPoseSample
    camera_from_scan_robot: np.ndarray
    sensor_age_seconds: float


def align_camera_lidar_epoch(
    history,
    *,
    lidar_timestamp_seconds,
    camera_timestamp_seconds,
    camera_from_robot,
    maximum_sensor_age_seconds=0.15,
):
    """Project scan-time chassis returns into the camera-time optical frame."""
    lidar_time, camera_time = (
        timestamp(lidar_timestamp_seconds),
        timestamp(camera_timestamp_seconds),
    )
    if not math.isfinite(maximum_sensor_age_seconds) or maximum_sensor_age_seconds <= 0:
        raise ValueError("maximum sensor age must be positive")
    age = camera_time - lidar_time
    if age < -1e-9 or age > maximum_sensor_age_seconds:
        raise ValueError("LiDAR epoch is future or stale for the camera observation")
    lidar_pose, camera_pose = history.sample(lidar_time), history.sample(camera_time)
    transform = (
        rigid_transform(camera_from_robot)
        @ np.linalg.inv(camera_pose.world_from_robot)
        @ lidar_pose.world_from_robot
    )
    return AlignedSensorEpoch(
        lidar_pose, camera_pose, rigid_transform(transform), max(0.0, age)
    )

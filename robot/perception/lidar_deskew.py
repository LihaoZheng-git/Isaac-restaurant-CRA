"""Ego-motion correction of uncompensated, individually timed RTX returns.

The output is a cloud in one retained chassis epoch. The caller must retain
pose provenance and declare the static-scene scope: this corrects sensor motion,
not independently moving objects. Unsupported coordinate/compensation contracts
and missing pose history are rejected rather than guessed.
"""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from robot.perception.sensor_epochs import rigid_transform, timestamp
from robot.sensor.lidar_scan_metadata import LidarScanMetadata


def deskew_uncompensated_lidar(
    sensor_xyz,
    metadata,
    history,
    *,
    robot_from_sensor,
    reference_timestamp_seconds,
    maximum_scan_duration_seconds=0.15,
):
    if (
        isinstance(maximum_scan_duration_seconds, bool)
        or not np.isfinite(maximum_scan_duration_seconds)
        or maximum_scan_duration_seconds <= 0
    ):
        raise ValueError("maximum scan duration must be finite and positive")
    if not isinstance(metadata, LidarScanMetadata):
        raise TypeError("copied LiDAR scan metadata is required")
    if metadata.frame_of_reference != "SENSOR":
        raise ValueError("deskew requires audited SENSOR-frame coordinates")
    if metadata.motion_compensation_state != "NONCOMPENSATED":
        raise ValueError("deskew requires NONCOMPENSATED output; never double-correct")
    app_transform = metadata.model_to_app_transform
    if app_transform is None or not np.allclose(
        np.asarray(app_transform).reshape(4, 4), np.eye(4), atol=1e-7
    ):
        raise ValueError("deskew requires the audited identity model-to-app transform")
    points = np.asarray(sensor_xyz, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or not np.isfinite(points).all()
        or not len(points)
    ):
        raise ValueError("deskew requires nonempty finite Nx3 sensor returns")
    offsets = metadata.point_time_offsets_nanoseconds
    if (
        offsets is None
        or offsets.shape != (len(points),)
        or offsets.dtype.kind not in "iu"
    ):
        raise ValueError("deskew requires per-return integer acquisition times")
    if np.any(offsets < 0):
        raise ValueError(
            "negative return offsets are outside the audited RTX scan contract"
        )
    if int(offsets.max()) * 1e-9 > maximum_scan_duration_seconds:
        raise ValueError("return times exceed the configured scan-duration scope")
    mount = rigid_transform(robot_from_sensor)
    reference_time = timestamp(reference_timestamp_seconds)
    reference = history.sample(reference_time)
    times_ns, lookup = np.unique(
        offsets + metadata.sensor_timestamp_nanoseconds, return_inverse=True
    )
    times = times_ns.astype(float) * 1e-9
    first, last = history.sample(float(times[0])), history.sample(float(times[-1]))
    if any(
        (item.status, item.source) != (reference.status, reference.source)
        for item in (first, last)
    ):
        raise ValueError("deskew cannot mix ego pose provenance")
    retained = history.samples
    history_times = np.array([item.timestamp_seconds for item in retained])
    # Only floating-point clock roundoff is clipped; sample() rejected actual
    # out-of-history timestamps above. No extrapolation or read-time substitute.
    times = np.clip(times, history_times[0], history_times[-1])
    right = np.clip(
        np.searchsorted(history_times, times, side="right"), 1, len(retained) - 1
    )
    left = right - 1
    for index in np.unique(np.concatenate((left, right))):
        item = retained[index]
        if (item.status, item.source) != (reference.status, reference.source):
            raise ValueError("deskew cannot mix ego pose provenance")
    gaps = history_times[right] - history_times[left]
    if np.any(gaps > history.maximum_gap + 1e-9):
        raise ValueError("deskew crosses an excessive ego pose history gap")
    used_left, used_right = int(left.min()), int(right.max())
    poses = np.stack(
        [item.world_from_robot for item in retained[used_left : used_right + 1]]
    )
    rotations = Slerp(
        history_times[used_left : used_right + 1],
        Rotation.from_matrix(poses[:, :3, :3]),
    )(times).as_matrix()
    alpha = (times - history_times[left]) / gaps
    translation = (1 - alpha[:, None]) * np.stack(
        [retained[i].world_from_robot[:3, 3] for i in left]
    ) + alpha[:, None] * np.stack([retained[i].world_from_robot[:3, 3] for i in right])
    body = points @ mount[:3, :3].T + mount[:3, 3]
    world = np.einsum("nij,nj->ni", rotations[lookup], body) + translation[lookup]
    inv_reference = np.linalg.inv(reference.world_from_robot)
    corrected = world @ inv_reference[:3, :3].T + inv_reference[:3, 3]
    return np.ascontiguousarray(corrected, dtype=np.float32), {
        "method": "per_return_ego_motion_deskew",
        "point_count": len(points),
        "distinct_return_times": len(times),
        "acquisition_start_seconds": float(times[0]),
        "acquisition_end_seconds": float(times[-1]),
        "reference_timestamp_seconds": reference_time,
        "maximum_scan_duration_seconds": maximum_scan_duration_seconds,
        "pose_source": reference.source,
        "pose_status": reference.status.value,
        "scope": "ego motion only; no independently moving object compensation",
    }

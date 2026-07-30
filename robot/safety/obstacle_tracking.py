"""Pure NumPy obstacle clustering and velocity tracking.

Input obstacle points must be expressed in the current robot base frame:

    +X forward
    +Y left

The tracker converts cluster centroids into the world frame before estimating
velocity. This compensates for robot translation and rotation. Estimated
physical obstacle velocities are then resolved back into the current robot
base-frame axes for direct use by ``CBFSafetyFilter``.

The tracker must be updated once per new LiDAR frame, using the LiDAR sensor
timestamp rather than once per physics step.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ObstacleTrackerConfig:
    """Configuration for Euclidean clustering and centroid tracking."""

    cluster_radius: float = 0.35
    minimum_cluster_points: int = 2
    maximum_association_distance: float = 0.50
    velocity_smoothing_factor: float = 0.50
    maximum_missed_updates: int = 3
    minimum_time_delta: float = 1e-6

    def __post_init__(self) -> None:
        positive_values = {
            "cluster_radius": self.cluster_radius,
            "maximum_association_distance": (
                self.maximum_association_distance
            ),
            "minimum_time_delta": self.minimum_time_delta,
        }

        for name, value in positive_values.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")

            if value <= 0.0:
                raise ValueError(
                    f"{name} must be greater than zero."
                )

        if (
            not isinstance(self.minimum_cluster_points, int)
            or isinstance(self.minimum_cluster_points, bool)
        ):
            raise TypeError(
                "minimum_cluster_points must be an integer."
            )

        if self.minimum_cluster_points <= 0:
            raise ValueError(
                "minimum_cluster_points must be greater than zero."
            )

        if not math.isfinite(
            self.velocity_smoothing_factor
        ):
            raise ValueError(
                "velocity_smoothing_factor must be finite."
            )

        if not (
            0.0
            < self.velocity_smoothing_factor
            <= 1.0
        ):
            raise ValueError(
                "velocity_smoothing_factor must be in (0, 1]."
            )

        if (
            not isinstance(self.maximum_missed_updates, int)
            or isinstance(self.maximum_missed_updates, bool)
        ):
            raise TypeError(
                "maximum_missed_updates must be an integer."
            )

        if self.maximum_missed_updates < 0:
            raise ValueError(
                "maximum_missed_updates cannot be negative."
            )


@dataclass(frozen=True)
class TrackedObstacle:
    """One obstacle cluster observed in the current LiDAR frame."""

    track_id: int
    centroid_world_xy: tuple[float, float]
    centroid_robot_xy: tuple[float, float]
    velocity_world_xy: tuple[float, float]
    velocity_robot_xy: tuple[float, float]
    point_count: int
    observation_count: int
    matched_previous_track: bool


@dataclass(frozen=True)
class ObstacleTrackingResult:
    """Tracking result aligned with the current obstacle-point array."""

    timestamp_seconds: float
    point_velocities_robot_xy: np.ndarray
    point_track_ids: np.ndarray
    tracked_obstacles: tuple[TrackedObstacle, ...]
    cluster_count: int
    matched_track_count: int
    new_track_count: int


@dataclass
class _TrackState:
    """Mutable state retained internally between LiDAR frames."""

    track_id: int
    centroid_world_xy: np.ndarray
    velocity_world_xy: np.ndarray
    last_observation_timestamp: float
    observation_count: int
    missed_updates: int


class ObstacleTracker:
    """Cluster obstacle points and estimate physical obstacle velocities."""

    def __init__(
        self,
        config: ObstacleTrackerConfig | None = None,
    ) -> None:
        self.config = config or ObstacleTrackerConfig()
        self._tracks: dict[int, _TrackState] = {}
        self._next_track_id = 0
        self._last_update_timestamp: float | None = None

    @property
    def active_track_count(self) -> int:
        """Return the number of retained track states."""

        return len(self._tracks)

    def reset(self) -> None:
        """Remove all track history and restart track identifiers."""

        self._tracks.clear()
        self._next_track_id = 0
        self._last_update_timestamp = None

    def update(
        self,
        obstacle_points_robot_xy: np.ndarray,
        *,
        robot_position_world_xy: np.ndarray,
        robot_yaw: float,
        timestamp_seconds: float,
    ) -> ObstacleTrackingResult:
        """Process one new LiDAR frame.

        Args:
            obstacle_points_robot_xy:
                Obstacle points with shape ``(N, 2)`` in the current
                robot base frame.
            robot_position_world_xy:
                Current planar robot position ``[x, y]`` in the world
                frame.
            robot_yaw:
                Current robot world-frame yaw in radians.
            timestamp_seconds:
                Monotonic LiDAR sensor timestamp in seconds.

        Returns:
            Per-point physical obstacle velocities resolved into the
            current robot frame, track identifiers, and cluster summaries.
        """

        points_robot = self._validate_points(
            obstacle_points_robot_xy
        )
        robot_position_world = self._validate_vector(
            robot_position_world_xy,
            name="robot_position_world_xy",
        )
        yaw = self._validate_scalar(
            robot_yaw,
            name="robot_yaw",
        )
        timestamp = self._validate_timestamp(
            timestamp_seconds
        )

        clusters = _cluster_point_indices(
            points_robot,
            cluster_radius=self.config.cluster_radius,
            minimum_cluster_points=(
                self.config.minimum_cluster_points
            ),
        )

        point_velocities_robot = np.zeros(
            points_robot.shape,
            dtype=np.float64,
        )
        point_track_ids = np.full(
            points_robot.shape[0],
            -1,
            dtype=np.int64,
        )

        centroid_positions_robot = [
            np.mean(
                points_robot[indices],
                axis=0,
                dtype=np.float64,
            )
            for indices in clusters
        ]
        centroid_positions_world = [
            _robot_point_to_world(
                centroid,
                robot_position_world_xy=(
                    robot_position_world
                ),
                robot_yaw=yaw,
            )
            for centroid in centroid_positions_robot
        ]

        cluster_to_track, matched_track_count = (
            self._update_tracks(
                centroid_positions_world,
                timestamp_seconds=timestamp,
            )
        )

        tracked_obstacles: list[TrackedObstacle] = []

        for cluster_index, point_indices in enumerate(
            clusters
        ):
            track = cluster_to_track[cluster_index]
            velocity_robot = _world_vector_to_robot(
                track.velocity_world_xy,
                robot_yaw=yaw,
            )

            point_velocities_robot[point_indices] = (
                velocity_robot
            )
            point_track_ids[point_indices] = (
                track.track_id
            )

            centroid_robot = (
                centroid_positions_robot[cluster_index]
            )

            tracked_obstacles.append(
                TrackedObstacle(
                    track_id=track.track_id,
                    centroid_world_xy=(
                        float(track.centroid_world_xy[0]),
                        float(track.centroid_world_xy[1]),
                    ),
                    centroid_robot_xy=(
                        float(centroid_robot[0]),
                        float(centroid_robot[1]),
                    ),
                    velocity_world_xy=(
                        float(track.velocity_world_xy[0]),
                        float(track.velocity_world_xy[1]),
                    ),
                    velocity_robot_xy=(
                        float(velocity_robot[0]),
                        float(velocity_robot[1]),
                    ),
                    point_count=int(point_indices.size),
                    observation_count=(
                        track.observation_count
                    ),
                    matched_previous_track=(
                        track.observation_count > 1
                    ),
                )
            )

        self._last_update_timestamp = timestamp

        cluster_count = len(clusters)

        return ObstacleTrackingResult(
            timestamp_seconds=timestamp,
            point_velocities_robot_xy=(
                np.ascontiguousarray(
                    point_velocities_robot,
                    dtype=np.float64,
                )
            ),
            point_track_ids=np.ascontiguousarray(
                point_track_ids,
                dtype=np.int64,
            ),
            tracked_obstacles=tuple(
                tracked_obstacles
            ),
            cluster_count=cluster_count,
            matched_track_count=matched_track_count,
            new_track_count=(
                cluster_count - matched_track_count
            ),
        )

    def _update_tracks(
        self,
        centroid_positions_world: list[np.ndarray],
        *,
        timestamp_seconds: float,
    ) -> tuple[dict[int, _TrackState], int]:
        """Associate current centroids with retained track states."""

        associations = self._associate_centroids(
            centroid_positions_world,
            timestamp_seconds=timestamp_seconds,
        )

        matched_track_ids = set(associations.values())
        next_tracks: dict[int, _TrackState] = {}
        cluster_to_track: dict[int, _TrackState] = {}

        for cluster_index, centroid_world in enumerate(
            centroid_positions_world
        ):
            previous_track_id = associations.get(
                cluster_index
            )

            if previous_track_id is None:
                track = self._create_track(
                    centroid_world,
                    timestamp_seconds=timestamp_seconds,
                )
            else:
                previous_track = self._tracks[
                    previous_track_id
                ]
                track = self._update_matched_track(
                    previous_track,
                    centroid_world,
                    timestamp_seconds=timestamp_seconds,
                )

            next_tracks[track.track_id] = track
            cluster_to_track[cluster_index] = track

        for track_id, track in self._tracks.items():
            if track_id in matched_track_ids:
                continue

            missed_updates = track.missed_updates + 1

            if (
                missed_updates
                <= self.config.maximum_missed_updates
            ):
                track.missed_updates = missed_updates
                next_tracks[track_id] = track

        self._tracks = next_tracks

        return cluster_to_track, len(associations)

    def _associate_centroids(
        self,
        centroid_positions_world: list[np.ndarray],
        *,
        timestamp_seconds: float,
    ) -> dict[int, int]:
        """Greedily associate nearest predicted tracks and centroids."""

        candidates: list[tuple[float, int, int]] = []

        for cluster_index, centroid_world in enumerate(
            centroid_positions_world
        ):
            for track_id, track in self._tracks.items():
                elapsed_since_observation = (
                    timestamp_seconds
                    - track.last_observation_timestamp
                )
                predicted_centroid = (
                    track.centroid_world_xy
                    + track.velocity_world_xy
                    * elapsed_since_observation
                )
                association_distance = float(
                    np.linalg.norm(
                        centroid_world
                        - predicted_centroid
                    )
                )

                if (
                    association_distance
                    <= self.config.maximum_association_distance
                ):
                    candidates.append(
                        (
                            association_distance,
                            cluster_index,
                            track_id,
                        )
                    )

        candidates.sort(key=lambda item: item[0])

        associations: dict[int, int] = {}
        used_track_ids: set[int] = set()

        for _, cluster_index, track_id in candidates:
            if cluster_index in associations:
                continue

            if track_id in used_track_ids:
                continue

            associations[cluster_index] = track_id
            used_track_ids.add(track_id)

        return associations

    def _create_track(
        self,
        centroid_world_xy: np.ndarray,
        *,
        timestamp_seconds: float,
    ) -> _TrackState:
        """Create a stationary-initialized track."""

        track = _TrackState(
            track_id=self._next_track_id,
            centroid_world_xy=np.array(
                centroid_world_xy,
                dtype=np.float64,
                copy=True,
            ),
            velocity_world_xy=np.zeros(
                2,
                dtype=np.float64,
            ),
            last_observation_timestamp=(
                timestamp_seconds
            ),
            observation_count=1,
            missed_updates=0,
        )

        self._next_track_id += 1

        return track

    def _update_matched_track(
        self,
        previous_track: _TrackState,
        centroid_world_xy: np.ndarray,
        *,
        timestamp_seconds: float,
    ) -> _TrackState:
        """Update one associated track and estimate its velocity."""

        elapsed_seconds = (
            timestamp_seconds
            - previous_track.last_observation_timestamp
        )

        if elapsed_seconds < self.config.minimum_time_delta:
            raise ValueError(
                "The elapsed time between matched observations "
                "is too small for velocity estimation."
            )

        measured_velocity = (
            centroid_world_xy
            - previous_track.centroid_world_xy
        ) / elapsed_seconds

        if previous_track.observation_count == 1:
            filtered_velocity = measured_velocity
        else:
            smoothing = (
                self.config.velocity_smoothing_factor
            )
            filtered_velocity = (
                smoothing * measured_velocity
                + (1.0 - smoothing)
                * previous_track.velocity_world_xy
            )

        return _TrackState(
            track_id=previous_track.track_id,
            centroid_world_xy=np.array(
                centroid_world_xy,
                dtype=np.float64,
                copy=True,
            ),
            velocity_world_xy=np.array(
                filtered_velocity,
                dtype=np.float64,
                copy=True,
            ),
            last_observation_timestamp=(
                timestamp_seconds
            ),
            observation_count=(
                previous_track.observation_count + 1
            ),
            missed_updates=0,
        )

    def _validate_timestamp(
        self,
        timestamp_seconds: float,
    ) -> float:
        """Validate a strictly increasing LiDAR timestamp."""

        timestamp = self._validate_scalar(
            timestamp_seconds,
            name="timestamp_seconds",
        )

        if self._last_update_timestamp is not None:
            elapsed_seconds = (
                timestamp
                - self._last_update_timestamp
            )

            if (
                elapsed_seconds
                < self.config.minimum_time_delta
            ):
                raise ValueError(
                    "timestamp_seconds must increase by at least "
                    f"{self.config.minimum_time_delta} seconds."
                )

        return timestamp

    @staticmethod
    def _validate_points(
        obstacle_points_robot_xy: np.ndarray,
    ) -> np.ndarray:
        """Return obstacle points as a validated float64 array."""

        points = np.asarray(
            obstacle_points_robot_xy,
            dtype=np.float64,
        )

        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(
                "obstacle_points_robot_xy must have shape (N, 2), "
                f"received {points.shape}."
            )

        if not np.isfinite(points).all():
            raise ValueError(
                "obstacle_points_robot_xy must contain only finite "
                "values."
            )

        return points

    @staticmethod
    def _validate_vector(
        value: np.ndarray,
        *,
        name: str,
    ) -> np.ndarray:
        """Return one validated planar float64 vector."""

        vector = np.asarray(
            value,
            dtype=np.float64,
        )

        if vector.shape != (2,):
            raise ValueError(
                f"{name} must have shape (2,), "
                f"received {vector.shape}."
            )

        if not np.isfinite(vector).all():
            raise ValueError(
                f"{name} must contain only finite values."
            )

        return vector

    @staticmethod
    def _validate_scalar(
        value: float,
        *,
        name: str,
    ) -> float:
        """Return one validated finite scalar."""

        converted_value = float(value)

        if not math.isfinite(converted_value):
            raise ValueError(f"{name} must be finite.")

        return converted_value


def _cluster_point_indices(
    points_xy: np.ndarray,
    *,
    cluster_radius: float,
    minimum_cluster_points: int,
) -> tuple[np.ndarray, ...]:
    """Return connected Euclidean point clusters.

    The quadratic pairwise-distance implementation is intentional: obstacle
    detection already reduces each scan to a small angular-binned point set,
    and this form keeps the baseline deterministic and dependency-free.
    """

    point_count = points_xy.shape[0]

    if point_count == 0:
        return ()

    offsets = (
        points_xy[:, np.newaxis, :]
        - points_xy[np.newaxis, :, :]
    )
    distance_squared = np.einsum(
        "ijk,ijk->ij",
        offsets,
        offsets,
    )
    neighbor_matrix = (
        distance_squared
        <= cluster_radius**2
    )

    visited = np.zeros(
        point_count,
        dtype=bool,
    )
    clusters: list[np.ndarray] = []

    for seed_index in range(point_count):
        if visited[seed_index]:
            continue

        pending = [seed_index]
        visited[seed_index] = True
        component: list[int] = []

        while pending:
            point_index = pending.pop()
            component.append(point_index)

            neighbor_indices = np.flatnonzero(
                neighbor_matrix[point_index]
                & ~visited
            )

            visited[neighbor_indices] = True
            pending.extend(
                int(index)
                for index in neighbor_indices
            )

        if len(component) >= minimum_cluster_points:
            clusters.append(
                np.array(
                    sorted(component),
                    dtype=np.int64,
                )
            )

    return tuple(clusters)


def _robot_point_to_world(
    point_robot_xy: np.ndarray,
    *,
    robot_position_world_xy: np.ndarray,
    robot_yaw: float,
) -> np.ndarray:
    """Transform one planar point from robot to world coordinates."""

    cosine_yaw = math.cos(robot_yaw)
    sine_yaw = math.sin(robot_yaw)

    return np.array(
        [
            (
                robot_position_world_xy[0]
                + cosine_yaw * point_robot_xy[0]
                - sine_yaw * point_robot_xy[1]
            ),
            (
                robot_position_world_xy[1]
                + sine_yaw * point_robot_xy[0]
                + cosine_yaw * point_robot_xy[1]
            ),
        ],
        dtype=np.float64,
    )


def _world_vector_to_robot(
    vector_world_xy: np.ndarray,
    *,
    robot_yaw: float,
) -> np.ndarray:
    """Resolve one world-frame vector into current robot-frame axes."""

    cosine_yaw = math.cos(robot_yaw)
    sine_yaw = math.sin(robot_yaw)

    return np.array(
        [
            (
                cosine_yaw * vector_world_xy[0]
                + sine_yaw * vector_world_xy[1]
            ),
            (
                -sine_yaw * vector_world_xy[0]
                + cosine_yaw * vector_world_xy[1]
            ),
        ],
        dtype=np.float64,
    )

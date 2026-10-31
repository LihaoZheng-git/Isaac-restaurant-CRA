"""Positive geometry uncertainty, not renewed identity or physical evidence.

The configured walking-speed bound covers every direction over source age plus
the entire checked controller horizon. Current sensor occupancy is never removed.
"""

from dataclasses import dataclass
import math

import numpy as np

SOURCE = "missing_track_reachable_geometry_prior"
ID_OFFSET = 6_000_000_000


@dataclass(frozen=True)
class ReachableTrackRegion:
    track_id: int
    center_world_xy: tuple
    original_lidar_timestamp: float
    original_camera_timestamp: float
    original_camera_frame: int
    evaluation_timestamp: float
    body_radius_m: float
    maximum_speed_mps: float
    checked_horizon_seconds: float
    reason: str

    def __post_init__(self):
        if (
            isinstance(self.track_id, bool)
            or not isinstance(self.track_id, (int, np.integer))
            or self.track_id < 0
            or not self.reason
        ):
            raise ValueError("valid original track and reason required")
        center = np.asarray(self.center_world_xy, float)
        if center.shape != (2,) or not np.isfinite(center).all():
            raise ValueError("finite original center required")
        object.__setattr__(self, "center_world_xy", tuple(float(v) for v in center))
        values = (
            self.original_lidar_timestamp,
            self.original_camera_timestamp,
            self.evaluation_timestamp,
            self.body_radius_m,
            self.maximum_speed_mps,
            self.checked_horizon_seconds,
        )
        if any(isinstance(v, bool) or not math.isfinite(v) for v in values):
            raise ValueError("finite source and motion bounds required")
        if (
            not 0 <= self.original_lidar_timestamp <= self.evaluation_timestamp
            or not 0
            <= self.original_camera_timestamp
            <= self.evaluation_timestamp + 0.25
            or self.body_radius_m <= 0
            or self.maximum_speed_mps < 1.0
            or self.checked_horizon_seconds < 1.55
            or isinstance(self.original_camera_frame, bool)
            or not isinstance(self.original_camera_frame, (int, np.integer))
            or self.original_camera_frame < 0
        ):
            raise ValueError("original epochs and full walking/horizon bounds required")

    @property
    def source_age(self):
        return self.evaluation_timestamp - self.original_lidar_timestamp

    @property
    def radius_m(self):
        return self.body_radius_m + self.maximum_speed_mps * (
            self.source_age + self.checked_horizon_seconds
        )

    @property
    def geometry_id(self):
        return ID_OFFSET + self.track_id

    def diagnostic(self):
        return dict(
            track_id=self.track_id,
            geometry_track_id=self.geometry_id,
            source=SOURCE,
            status="prior",
            reason=self.reason,
            center_world_xy=list(self.center_world_xy),
            radius_m=self.radius_m,
            original_lidar_timestamp_seconds=self.original_lidar_timestamp,
            original_person_camera_timestamp=self.original_camera_timestamp,
            original_person_camera_frame_id=self.original_camera_frame,
            evaluation_timestamp_seconds=self.evaluation_timestamp,
            source_age_seconds=self.source_age,
            body_radius_prior_m=self.body_radius_m,
            maximum_speed_prior_mps=self.maximum_speed_mps,
            checked_horizon_seconds=self.checked_horizon_seconds,
            current_identity_verified=False,
            current_motion_measured=False,
            scope="all-direction reach from last verified positive geometry; no source renewal, factor transfer or free-space inference",
        )

    def cells(self, config, anchor_xy, anchor_yaw):
        c, s = math.cos(anchor_yaw), math.sin(anchor_yaw)
        center = (np.asarray(self.center_world_xy) - anchor_xy) @ np.array(
            [[c, -s], [s, c]]
        )
        # Work is bounded by the local grid even after a long source absence.
        cols, rows = np.meshgrid(
            np.arange(config.column_count), np.arange(config.row_count)
        )
        xy = np.column_stack((cols.ravel(), rows.ravel())) * config.resolution
        xy += np.array([config.minimum_x, config.minimum_y]) + 0.5 * config.resolution
        selected = np.linalg.norm(
            xy - center, axis=1
        ) <= self.radius_m + config.resolution / math.sqrt(2)
        return xy[selected], rows.ravel()[selected], cols.ravel()[selected], center


@dataclass(frozen=True)
class TrackingUncertaintySnapshot:
    source_time: float
    regions_and_guidance: tuple = ()

    def coverage(
        self,
        key,
        original_time,
        original_points,
        original_risk,
        grid,
        anchor_xy,
        anchor_yaw,
    ):
        """Replace a bookkeeping stop only with whole current-field coverage."""
        matches = [(r, g) for r, g in self.regions_and_guidance if r.track_id == key]
        if len(matches) != 1 or original_points is None:
            return None
        region, guidance = matches[0]
        points = np.asarray(original_points, float)
        if (
            region.evaluation_timestamp != self.source_time
            or original_time != region.original_lidar_timestamp
            or points.ndim != 2
            or points.shape[1] != 2
            or not len(points)
            or not np.isfinite(points).all()
            or not math.isfinite(original_risk)
            or guidance.risk < original_risk
            or np.any(
                np.linalg.norm(points - region.center_world_xy, axis=1)
                > region.body_radius_m
            )
        ):
            return None
        _, rows, cols, center = region.cells(grid.config, anchor_xy, anchor_yaw)
        cfg = grid.config
        # Nearby lost geometry cannot be excused by an out-of-domain raster.
        if (
            np.any(center - region.radius_m < [cfg.minimum_x, cfg.minimum_y])
            or np.any(center + region.radius_m >= [cfg.maximum_x, cfg.maximum_y])
            or not len(rows)
            or not np.all(grid.occupied_mask[rows, cols])
        ):
            return None
        # The grid may choose a stronger overlapping source; identity is not a
        # reason to reject equally conservative physical occupancy.
        if np.any(grid.cell_risks[rows, cols] < guidance.risk - 1e-12) or np.any(
            grid.cell_boundary_fluxes[rows, cols] > guidance.boundary_flux + 1e-12
        ):
            return None
        return dict(
            track_id=key,
            original_source_time=original_time,
            current_source_time=self.source_time,
            status="prior",
            source=SOURCE,
            radius_m=region.radius_m,
            original_risk=original_risk,
            applied_risk=guidance.risk,
            whole_region_current_field_covered=True,
        )

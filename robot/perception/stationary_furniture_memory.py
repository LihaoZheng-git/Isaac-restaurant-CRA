"""Bounded positive-surface PRIOR for stationary furniture in a mixed scene.

Repeated world-space support is geometric stationarity evidence, not a furniture
classifier. The experiment assumes separated stationary background assets;
unknown people, newly moving furniture and crowded occlusion are outside scope.
No free-space clearing, factor refresh, actor truth or planner command lives here.
"""

from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy.spatial import cKDTree


SOURCE = "bounded_stationary_furniture_surface_prior"
ID_OFFSET = 5_000_000_000


@dataclass(frozen=True)
class FurnitureMemoryConfig:
    maximum_age_seconds: float = 10.0
    minimum_support_seconds: float = .2
    maximum_sample_gap_seconds: float = .35
    support_distance_m: float = .06
    uncertainty_radius_m: float = .05
    uncertainty_growth_mps: float = .005
    human_exclusion_radius_m: float = 1.0
    moving_surface_exclusion_radius_m: float = .06
    moving_track_speed_mps: float = .2
    enrollment_distance_m: float = 2.5

    def __post_init__(self):
        for value in vars(self).values():
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("furniture memory limits must be finite and positive")
        if self.maximum_age_seconds > 10 or self.minimum_support_seconds < .2:
            raise ValueError("bounded furniture memory requires <=10s age and >=.2s support")


def raster_surface(points, radius, grid_config, anchor_xy, anchor_yaw):
    """All cells intersecting a disk about each ORIGINAL positive return.

    Return in-domain cell centers and a clipping flag. The guard rejects a
    clipped proof; a partial raster can still add conservative field occupancy.
    """
    c, s = math.cos(anchor_yaw), math.sin(anchor_yaw)
    local = (np.asarray(points) - anchor_xy) @ np.array([[c, -s], [s, c]])
    origin = np.array([grid_config.minimum_x, grid_config.minimum_y])
    step = grid_config.resolution
    base = np.floor((local - origin) / step).astype(np.int64)
    extent = int(math.ceil(radius / step)) + 1
    offsets = np.array([(x, y) for x in range(-extent, extent + 1)
                        for y in range(-extent, extent + 1)])
    # Work per unique original cell/subcell without an unbounded dense grid.
    cells = base[:, None, :] + offsets
    centers = origin + (cells + .5) * step
    delta = np.maximum(np.abs(centers - local[:, None, :]) - step / 2, 0)
    selected = cells[np.linalg.norm(delta, axis=2) <= radius + 1e-12]
    selected = np.unique(selected, axis=0)
    inside = ((selected[:, 0] >= 0) & (selected[:, 0] < grid_config.column_count)
              & (selected[:, 1] >= 0) & (selected[:, 1] < grid_config.row_count))
    return selected[inside], bool(not inside.all())


@dataclass(frozen=True)
class RememberedSurface:
    track_id: int
    source_time: float
    source_frame: int
    support_start_time: float
    centroid: np.ndarray
    points: np.ndarray
    guidance: object


@dataclass(frozen=True)
class FurnitureSnapshot:
    source_time: float
    config: FurnitureMemoryConfig
    surfaces: tuple

    def radius(self, record):
        return self.config.uncertainty_radius_m + self.config.uncertainty_growth_mps * (
            self.source_time - record.source_time)

    def placements(self, grid_config, anchor_xy, anchor_yaw):
        points, ids = [], []
        for record in self.surfaces:
            cells, _ = raster_surface(record.points, self.radius(record), grid_config, anchor_xy, anchor_yaw)
            points.append(np.array([grid_config.minimum_x, grid_config.minimum_y])
                          + (cells + .5) * grid_config.resolution)
            ids.extend([ID_OFFSET + record.track_id] * len(cells))
        return (np.concatenate(points) if points else np.empty((0, 2)), np.asarray(ids, dtype=np.int64))

    def coverage(self, track_id, last_seen, centroid, grid, anchor_xy, anchor_yaw):
        """A guard exception needs exact old identity/epoch AND actual grid cover."""
        records = [r for r in self.surfaces if r.track_id == track_id]
        if len(records) != 1:
            return None
        r = records[0]
        age = self.source_time - r.source_time
        if (abs(r.source_time - last_seen) > 1e-9 or not 0 < age <= self.config.maximum_age_seconds
                or not np.allclose(r.centroid, centroid, atol=1e-9, rtol=0)):
            return None
        cells, clipped = raster_surface(r.points, self.radius(r), grid.config, anchor_xy, anchor_yaw)
        if clipped or not len(cells):
            return None
        cols, rows = cells.T
        if (not grid.occupied_mask[rows, cols].all()
                or not (grid.cell_risks[rows, cols] >= r.guidance.risk - 1e-12).all()
                or not (grid.cell_boundary_fluxes[rows, cols] <= r.guidance.boundary_flux + 1e-12).all()):
            return None
        return dict(track_id=int(track_id), source=SOURCE, status="prior",
                    original_source_time=r.source_time, original_source_frame=r.source_frame,
                    current_source_time=self.source_time, missing_seconds=age,
                    original_point_count=len(r.points), original_risk=r.guidance.risk,
                    uncertainty_radius_m=self.radius(r), checked_cell_count=len(cells),
                    support_start_time=r.support_start_time,
                    basis="whole_original_surface_in_current_field; unchanged_source_and_risk")

    def diagnostic(self):
        return dict(source=SOURCE, status="prior", source_time=self.source_time,
                    config=asdict(self.config), retained_count=len(self.surfaces),
                    retained=[dict(track_id=r.track_id, original_source_time=r.source_time,
                                   original_source_frame=r.source_frame, original_point_count=len(r.points),
                                   original_risk=r.guidance.risk, uncertainty_radius_m=self.radius(r),
                                   support_start_time=r.support_start_time) for r in self.surfaces])


class StationaryFurnitureMemory:
    def __init__(self, config=None):
        self.config = config or FurnitureMemoryConfig()
        self.history = []
        self.records = {}
        self.last_time = None

    def update(self, points, ids, tracks, guidance, *, source_time, source_frame,
               robot_xy, excluded_points=(), excluded_track_ids=()):
        """Original scans only; callers stage this object with field admission.

        At least three distinct scans spanning .2s must cover EVERY original
        point. Track IDs can change across those scans. Suspected human/moving
        returns purge nearby history and records, including previously idle ones.
        A current incomplete observation does not inherit old eligibility.
        """
        cfg = self.config
        points, ids = np.asarray(points, float), np.asarray(ids)
        if (not math.isfinite(source_time) or source_time < 0
                or self.last_time is not None and source_time <= self.last_time
                or points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all()
                or ids.shape != (len(points),) or not np.issubdtype(ids.dtype, np.integer)):
            raise ValueError("furniture memory requires aligned finite increasing observed sources")
        tracks = {t.track_id: t for t in tracks}
        if any(np.count_nonzero(ids == k) != t.point_count for k, t in tracks.items()):
            raise ValueError("furniture memory requires original point ownership")
        people = set(excluded_track_ids)
        moving = {k for k, t in tracks.items() if t.observation_count >= 3
                  and np.linalg.norm(t.velocity_world_xy) > cfg.moving_track_speed_mps}
        excluded = people | moving
        forbidden = [points[np.isin(ids, list(people))]]
        extra = np.asarray(excluded_points, float).reshape(-1, 2)
        if not np.isfinite(extra).all():
            raise ValueError("invalid exclusion geometry")
        forbidden.append(extra)
        forbidden = np.concatenate(forbidden)
        tree = cKDTree(forbidden) if len(forbidden) else None
        moving_points = points[np.isin(ids, list(moving))]
        moving_tree = cKDTree(moving_points) if len(moving_points) else None

        def allowed(p):
            human_clear = (np.ones(len(p), dtype=bool) if tree is None else
                           tree.query(p)[0] > cfg.human_exclusion_radius_m)
            motion_clear = (np.ones(len(p), dtype=bool) if moving_tree is None else
                            moving_tree.query(p)[0] > cfg.moving_surface_exclusion_radius_m)
            return human_clear & motion_clear

        usable = (ids >= 0) & ~np.isin(ids, list(excluded)) & allowed(points)
        # Nearby enrollment is a resource/scope limit, not a free-space claim.
        usable &= np.linalg.norm(points - robot_xy, axis=1) <= cfg.enrollment_distance_m
        history = [(t, p[allowed(p)]) for t, p in self.history
                   if 0 < source_time - t <= 2 * cfg.maximum_sample_gap_seconds + 1e-9]
        if self.last_time is not None and source_time - self.last_time > cfg.maximum_sample_gap_seconds + 1e-9:
            history = []
        support = np.zeros(len(points), dtype=int)
        earliest = np.full(len(points), source_time)
        for t, p in history:
            if len(p):
                matched = cKDTree(p).query(points)[0] <= cfg.support_distance_m
                support += matched
                earliest[matched] = np.minimum(earliest[matched], t)
        stable = usable & (support >= 2) & (source_time - earliest >= cfg.minimum_support_seconds - 1e-9)
        records = {k: r for k, r in self.records.items()
                   if 0 < source_time - r.source_time <= cfg.maximum_age_seconds
                   and k not in excluded and allowed(r.points).all()}
        enrolled = []
        for key, track in tracks.items():
            owned = ids == key
            records.pop(key, None)
            if owned.sum() < 3 or not stable[owned].all():
                continue
            value = guidance[key]
            if value.tangential_strength != 1.25 or not math.isclose(value.boundary_flux, -(.5 + 1.2 * value.risk), abs_tol=1e-12):
                raise ValueError("furniture memory requires unchanged primary lambda and CRA mapping")
            records[key] = RememberedSurface(int(key), float(source_time), int(source_frame),
                float(earliest[owned].max()), np.asarray(track.centroid_world_xy).copy(),
                points[owned].copy(), value)
            enrolled.append(int(key))
        self.records, self.last_time = records, float(source_time)
        self.history = (history + [(float(source_time), points[usable].copy())])[-8:]
        snapshot = FurnitureSnapshot(float(source_time), cfg,
                                     tuple(r for k, r in records.items() if k not in tracks))
        audit = snapshot.diagnostic()
        audit.update(enrolled_track_ids=enrolled, stable_point_count=int(stable.sum()),
                     excluded_track_ids=sorted(excluded), exclusion_point_count=len(forbidden),
                     current_observed_point_count=len(points),
                     scope="repeated geometric support plus stationary-background prior; not semantic furniture recognition")
        return snapshot, audit


def person_exclusion_points(world, projection, person_frame, camera_frame, camera_time, envelopes):
    """A suspect person mask EXCLUDES memory; it does not assert association."""
    points = []
    if person_frame is not None:
        if person_frame.source_frame_id != camera_frame or person_frame.source_timestamp_seconds != camera_time:
            raise ValueError("person exclusion requires the actual current RGB source")
        ix, pixels = projection.source_indices, projection.pixel_indices_xy
        for d in person_frame.detections:
            if d.semantic_class == "person" and d.confidence >= .5:
                points.append(world[ix[d.mask[pixels[:, 1], pixels[:, 0]]], :2])
    for row in envelopes:
        points.append(np.asarray(row["center_world_xy"]).reshape(1, 2))
    return np.concatenate(points) if points else np.empty((0, 2))

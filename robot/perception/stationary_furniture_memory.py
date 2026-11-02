"""Bounded positive-surface PRIOR for stationary furniture in a mixed scene.

Repeated world-space support is geometric stationarity evidence, not a furniture
classifier. The experiment assumes separated stationary background assets;
unknown people, newly moving furniture and crowded occlusion are outside scope.
No free-space clearing, factor refresh, actor truth or planner command lives here.
"""

from dataclasses import asdict, dataclass, replace
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
    fragment_level_proof: bool = False

    def __post_init__(self):
        if not isinstance(self.fragment_level_proof, bool):
            raise ValueError("fragment-level proof must be an explicit boolean")
        for name, value in vars(self).items():
            if name == "fragment_level_proof":
                continue
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
    qualification_source_time: float | None = None


@dataclass(frozen=True)
class ObservedSurfaceCoverage:
    original: RememberedSurface
    current_source_frame: int
    current_track_id: int
    points: np.ndarray
    maximum_correspondence_distance_m: float
    fragment_level_proof: bool = False
    support_source_times: tuple = ()


@dataclass(frozen=True)
class FurnitureSnapshot:
    source_time: float
    config: FurnitureMemoryConfig
    surfaces: tuple
    observed_surfaces: tuple = ()

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
        for observed in self.observed_surfaces:
            r = observed.original
            if (r.track_id != track_id or abs(r.source_time-last_seen) > 1e-9
                or not np.allclose(r.centroid, centroid, atol=1e-9, rtol=0)):
                continue
            # These are actual CURRENT returns, not the old uncertainty raster.
            c, s = math.cos(anchor_yaw), math.sin(anchor_yaw)
            local = (observed.points-anchor_xy) @ np.array([[c, -s], [s, c]])
            cells = np.unique(np.floor((local-[grid.config.minimum_x, grid.config.minimum_y])
                                      / grid.config.resolution).astype(int), axis=0)
            if (not len(cells) or np.any(cells < 0)
                or np.any(cells[:, 0] >= grid.config.column_count)
                or np.any(cells[:, 1] >= grid.config.row_count)):
                continue
            cols, rows = cells.T
            if (grid.occupied_mask[rows, cols].all()
                and (grid.cell_risks[rows, cols] >= r.guidance.risk-1e-12).all()
                and (grid.cell_boundary_fluxes[rows, cols] <= r.guidance.boundary_flux+1e-12).all()):
                return dict(track_id=int(track_id), source="current_whole_stationary_surface_coverage",
                    status="inferred", original_source_time=r.source_time,
                    original_source_frame=r.source_frame, current_source_time=self.source_time,
                    current_source_frame=observed.current_source_frame,
                    current_track_id=observed.current_track_id,
                    missing_seconds=self.source_time-r.source_time,
                    qualification_source_time=r.qualification_source_time,
                    support_start_time=r.support_start_time,
                    original_point_count=len(r.points), current_covering_point_count=len(observed.points),
                    original_risk=r.guidance.risk, checked_cell_count=len(cells),
                    maximum_correspondence_distance_m=observed.maximum_correspondence_distance_m,
                    fragment_level_proof=observed.fragment_level_proof,
                    support_source_times=observed.support_source_times,
                    basis="fresh_unique_whole_positive_surface_in_current_field; no_retained_geometry_or_source_renewal")
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
                    qualification_source_time=r.qualification_source_time,
                    basis="whole_original_surface_in_current_field; unchanged_source_and_risk")

    def diagnostic(self):
        return dict(source=SOURCE, status="prior", source_time=self.source_time,
                    observed_coverage_count=len(self.observed_surfaces),
                    observed_coverage=[dict(track_id=o.original.track_id,
                        original_source_time=o.original.source_time,
                        current_source_time=self.source_time, current_source_frame=o.current_source_frame,
                        current_track_id=o.current_track_id, current_point_count=len(o.points),
                        fragment_level_proof=o.fragment_level_proof,
                        support_source_times=o.support_source_times,
                        maximum_correspondence_distance_m=o.maximum_correspondence_distance_m)
                        for o in self.observed_surfaces],
                    config=asdict(self.config), retained_count=len(self.surfaces),
                    retained=[dict(track_id=r.track_id, original_source_time=r.source_time,
                                   original_source_frame=r.source_frame, original_point_count=len(r.points),
                                   original_risk=r.guidance.risk, uncertainty_radius_m=self.radius(r),
                                   support_start_time=r.support_start_time,
                                   qualification_source_time=r.qualification_source_time) for r in self.surfaces])


class StationaryFurnitureMemory:
    def __init__(self, config=None):
        self.config = config or FurnitureMemoryConfig()
        self.history = []
        self.records = {}
        self.pending_surfaces = {}
        self.observed_associations = {}
        # Separate history for proven pieces of an excluded parent. Never used
        # to qualify a new surface or the rest of that parent.
        self.fragment_history = {}
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
        high_centroid_speed = {k for k, t in tracks.items() if t.observation_count >= 3
                               and np.linalg.norm(t.velocity_world_xy) > cfg.moving_track_speed_mps}
        forbidden = [points[np.isin(ids, list(people))]]
        extra = np.asarray(excluded_points, float).reshape(-1, 2)
        if not np.isfinite(extra).all():
            raise ValueError("invalid exclusion geometry")
        forbidden.append(extra)
        forbidden = np.concatenate(forbidden)
        tree = cKDTree(forbidden) if len(forbidden) else None
        def human_clear(p):
            return (np.ones(len(p), dtype=bool) if tree is None else
                    tree.query(p)[0] > cfg.human_exclusion_radius_m)

        history = [(t, p[human_clear(p)]) for t, p in self.history
                   if 0 < source_time - t <= 2 * cfg.maximum_sample_gap_seconds + 1e-9]
        source_gap = (self.last_time is not None
                      and source_time - self.last_time > cfg.maximum_sample_gap_seconds + 1e-9)
        if source_gap:
            history = []

        support_trees = {}

        def support_tree(p):
            # Keep the array alive with its tree; several pending fragments
            # query the same actual scan without rebuilding its spatial index.
            if id(p) not in support_trees:
                support_trees[id(p)] = (p, cKDTree(p))
            return support_trees[id(p)][1]

        def support_from(samples, query=points, minimum_sources=2):
            support = np.zeros(len(query), dtype=int)
            earliest = np.full(len(query), source_time)
            for t, p in samples:
                if len(p):
                    matched = support_tree(p).query(query)[0] <= cfg.support_distance_m
                    support += matched
                    earliest[matched] = np.minimum(earliest[matched], t)
            stable = (support >= minimum_sources) & (source_time - earliest >= cfg.minimum_support_seconds - 1e-9)
            return stable, earliest

        # Centroid velocity can jump when a stationary cluster loses half its
        # visible surface. Prefer repeated support of EVERY actual world point;
        # leave physical/kinetic velocity evidence unchanged and log disagreement.
        world_supported, _ = support_from(history)
        disagreement = {k for k in high_centroid_speed if world_supported[ids == k].all()}
        moving = high_centroid_speed - disagreement
        excluded = people | moving
        moving_points = points[np.isin(ids, list(moving))]
        moving_tree = cKDTree(moving_points) if len(moving_points) else None

        raw_tree = support_tree(points) if len(points) else None

        def unique_owner(original):
            if raw_tree is None:
                return None
            neighborhoods = raw_tree.query_ball_point(original, cfg.support_distance_m)
            indices = np.unique(np.concatenate(neighborhoods)).astype(int)
            nearby_ids = ids[indices]
            if np.count_nonzero(nearby_ids == -1) >= 3:
                return None
            owners = []
            for owner in np.unique(nearby_ids[nearby_ids >= 0]):
                owned_indices = indices[nearby_ids == owner]
                if (len(np.unique(points[owned_indices], axis=0)) >= 3
                    and np.all(support_tree(points[owned_indices]).query(original)[0] <= cfg.support_distance_m)):
                    owners.append((int(owner), owned_indices))
            return owners[0] if len(owners) == 1 else None

        continuous = {k: r for k, (r, t) in self.observed_associations.items()
                      if not source_gap and 0 < source_time-t <= cfg.maximum_sample_gap_seconds+1e-9}
        qualified = dict(continuous)
        qualified.update({k: r for k, r in self.records.items()
                          if 0 < source_time-r.source_time <= cfg.maximum_age_seconds})
        fragment_proofs, fragment_history = {}, {}
        if cfg.fragment_level_proof and not source_gap:
            for key, r in qualified.items():
                if key in tracks or key in people or not human_clear(r.points).all():
                    continue
                match = unique_owner(r.points)
                if match is None:
                    continue
                owner, indices = match
                if owner in people or (owner not in moving and key not in self.fragment_history):
                    continue
                # A different moving surface near the old piece is still a
                # contradiction. Only the uniquely covering parent is exempt.
                other = points[np.isin(ids, list(moving - {owner}))]
                if len(other) and np.any(support_tree(other).query(r.points)[0] <= cfg.moving_surface_exclusion_radius_m):
                    continue
                samples = list(history)
                old = self.fragment_history.get(key)
                if old is not None and old[0] == r.source_time:
                    samples += [(t, p[human_clear(p)]) for t, p in old[1]
                                if 0 < source_time-t <= 2*cfg.maximum_sample_gap_seconds+1e-9]
                # Normal and fragment histories may share a scan. Count each
                # actual source once, never duplicate a source into three scans.
                by_time = {}
                for t, p in samples:
                    by_time.setdefault(t, []).append(p)
                samples = [(t, np.concatenate(parts)) for t, parts in sorted(by_time.items())]
                current = points[indices]
                supported, _ = support_from(samples, current)
                current = current[supported & human_clear(current)]
                original_supported, _ = support_from(samples, r.points)
                if not original_supported.all() or len(np.unique(current, axis=0)) < 3:
                    continue
                distances = support_tree(current).query(r.points)[0]
                if np.any(distances > cfg.support_distance_m):
                    continue
                proof = ObservedSurfaceCoverage(r, int(source_frame), owner,
                    current.copy(), float(distances.max()), True,
                    tuple(t for t, _ in samples) + (float(source_time),))
                fragment_proofs[key] = proof
                # Save only positively verified local returns from this source.
                # Failed proofs disappear; history cannot revive an old query.
                original_tree = support_tree(r.points)
                fragment_history[key] = (r.source_time,
                    ([(t, p[original_tree.query(p)[0] <= cfg.support_distance_m])
                      for t, p in samples]
                     + [(float(source_time), current.copy())])[-8:])

        def allowed(p):
            motion_clear = (np.ones(len(p), dtype=bool) if moving_tree is None else
                            moving_tree.query(p)[0] > cfg.moving_surface_exclusion_radius_m)
            return human_clear(p) & motion_clear

        usable = (ids >= 0) & ~np.isin(ids, list(excluded)) & allowed(points)
        history = [(t, p[allowed(p)]) for t, p in history]
        world_supported, earliest = support_from(history)
        stable = usable & world_supported
        records = {k: r for k, r in self.records.items()
                   if 0 < source_time - r.source_time <= cfg.maximum_age_seconds
                   and (k in fragment_proofs or k not in excluded and allowed(r.points).all())}
        # A fragment may change ID just before its third supporting scan. Keep
        # its original measured surface unpublished for at most TWO existing
        # source-gap intervals (the three-scan history window). Later actual
        # scans may complete the same three-
        # source/.2s/.06m proof, without renewing its geometry/policy epoch.
        pending = {k: r for k, r in self.pending_surfaces.items()
                   if not source_gap and k not in tracks and k not in excluded
                   and 0 < source_time-r.source_time <= 2*cfg.maximum_sample_gap_seconds+1e-9
                   and allowed(r.points).all()}
        current_points = points[usable]
        current_tree = support_tree(current_points) if len(current_points) else None
        confirmed = []
        for key, candidate in list(pending.items()):
            supported, first = support_from(history + [(source_time, current_points)],
                                            candidate.points, minimum_sources=3)
            if (supported.all() and current_tree is not None
                and np.all(current_tree.query(candidate.points)[0] <= cfg.support_distance_m)):
                records[key] = replace(candidate, support_start_time=float(first.max()),
                                       qualification_source_time=float(source_time))
                confirmed.append(key)
                del pending[key]
        enrolled, rejected = [], []
        for key, track in tracks.items():
            owned = ids == key
            records.pop(key, None)
            # Match the guard's centroid-based enrollment scope. Keep full raw
            # scan history so outer points have genuine earlier support before
            # a large furniture cluster's center enters the local watch region.
            if np.linalg.norm(np.asarray(track.centroid_world_xy) - robot_xy) > cfg.enrollment_distance_m:
                continue
            value = guidance[key]
            if value.tangential_strength != 1.25 or not math.isclose(value.boundary_flux, -(.5 + 1.2 * value.risk), abs_tol=1e-12):
                raise ValueError("furniture memory requires unchanged primary lambda and CRA mapping")
            candidate = RememberedSurface(int(key), float(source_time), int(source_frame),
                float(earliest[owned].max()), np.asarray(track.centroid_world_xy).copy(),
                points[owned].copy(), value)
            if owned.sum() < 3 or not stable[owned].all():
                rejected.append(dict(track_id=int(key), original_points=int(owned.sum()),
                    supported_points=int(stable[owned].sum()), excluded=key in excluded))
                if owned.sum() >= 3 and usable[owned].all():
                    pending[key] = candidate
                continue
            records[key] = replace(candidate, qualification_source_time=float(source_time))
            enrolled.append(int(key))
        self.pending_surfaces = pending
        # A freshly observed whole surface can resolve geometry continuity even
        # when its large parent cluster has a distant centroid. The old query
        # and policy stay unchanged; no old geometry is added after its expiry.
        # Beyond that expiry, a proof must be continuous across real sources.
        associations = dict(continuous)
        associations.update(records)
        observed, verified = [], {}
        for key, r in associations.items():
            if key in fragment_proofs:
                observed.append(fragment_proofs[key])
                verified[key] = (r, float(source_time))
                continue
            if key in tracks or key in excluded or not allowed(r.points).all() or raw_tree is None:
                continue
            match = unique_owner(r.points)
            if match is None:
                continue
            owner, owned_indices = match
            covered_points = points[owned_indices[stable[owned_indices]]]
            if len(np.unique(covered_points, axis=0)) < 3:
                continue
            distance = support_tree(covered_points).query(r.points)[0]
            if np.any(distance > cfg.support_distance_m):
                continue
            observed.append(ObservedSurfaceCoverage(r, int(source_frame), owner,
                                                   covered_points.copy(), float(distance.max())))
            verified[key] = (r, float(source_time))
        self.observed_associations = verified
        self.fragment_history = fragment_history
        self.records, self.last_time = records, float(source_time)
        self.history = (history + [(float(source_time), points[usable].copy())])[-8:]
        snapshot = FurnitureSnapshot(float(source_time), cfg,
                                     tuple(r for k, r in records.items() if k not in tracks), tuple(observed))
        audit = snapshot.diagnostic()
        audit.update(enrolled_track_ids=enrolled, stable_point_count=int(stable.sum()),
                     qualification_history_reset=bool(source_gap),
                     pending_surface_count=len(pending), confirmed_missing_track_ids=confirmed,
                     rejected_tracks=rejected, centroid_surface_disagreement_track_ids=sorted(disagreement),
                     fragment_proof_track_ids=sorted(fragment_proofs),
                     excluded_track_ids=sorted(excluded), exclusion_point_count=len(forbidden),
                     current_observed_point_count=len(points),
                     scope="repeated geometric support plus stationary-background prior; not semantic furniture recognition")
        return snapshot, audit


def person_exclusion_points(world, projection, person_frame, camera_frame, camera_time, envelopes):
    """Resolved person masks exclude without asserting instance association.

    A clipped detector box/mask is an unresolved cue, not evidence that all
    LiDAR rays behind the image edge belong to a person. Current validated body
    centers still exclude unconditionally. No rejected mask establishes absence
    of a person or clears occupancy; partial unrecognized humans remain outside
    this separated, visible-human experiment's acceptance.
    """
    points, detections = [], []
    if person_frame is not None:
        if person_frame.source_frame_id != camera_frame or person_frame.source_timestamp_seconds != camera_time:
            raise ValueError("person exclusion requires the actual current RGB source")
        ix, pixels = projection.source_indices, projection.pixel_indices_xy
        for index, d in enumerate(person_frame.detections):
            if d.semantic_class == "person" and d.confidence >= .5:
                mask, box = d.mask, d.bounding_box
                height, width = mask.shape
                # Include the detector's continuous box extent: mask resizing
                # can leave a one-pixel sliver even when its box hits the edge.
                clipped = (mask[0].any() or mask[-1].any() or mask[:, 0].any()
                    or mask[:, -1].any() or math.floor(box.x_min) <= 0
                    or math.floor(box.y_min) <= 0 or math.ceil(box.x_max) >= width
                    or math.ceil(box.y_max) >= height)
                hits = mask[pixels[:, 1], pixels[:, 0]]
                detections.append(dict(detection_index=index, confidence=float(d.confidence),
                    accepted=not bool(clipped), projected_return_count=int(hits.sum()),
                    reason="unresolved_image_boundary_person_cue" if clipped else "resolved_current_person_mask",
                    bounding_box=asdict(box)))
                if not clipped:
                    points.append(world[ix[hits], :2])
    for row in envelopes:
        points.append(np.asarray(row["center_world_xy"]).reshape(1, 2))
    return (np.concatenate(points) if points else np.empty((0, 2))), dict(
        camera_frame_id=camera_frame, camera_timestamp_seconds=camera_time,
        detections=detections, current_body_center_count=len(envelopes),
        scope="resolved raw masks plus current validated bodies; rejected boundary cues are unknown, not negative person evidence")

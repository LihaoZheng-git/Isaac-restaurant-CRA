"""Lost positive surfaces remain planning obstacles, independent of track IDs.

This is a geometry PRIOR, not renewed CRA evidence or inferred free space.
Stationary support and generic motion uncertainty have separate source clocks.
Only the explicit experimental mode uses this memory; legacy gates are intact.
"""

from dataclasses import asdict, dataclass, field, replace
import math

import numpy as np
from scipy.spatial import cKDTree

from robot.safety.class_safety_policy import ClassSafetyParameters
from robot.safety.safe_sage_guidance_policy import SafeSAGEGuidanceParameters

SOURCE = "unknown_obstacle_geometry_prior"
ID_OFFSET = 7_000_000_000


@dataclass(frozen=True)
class UnknownObstacleConfig:
    # Same numeric phi as four existing 0.25 factor priors, not a new estimate.
    unknown_phi: float = 0.68359375
    enrollment_distance_m: float = 2.5
    stationary_radius_m: float = .05
    stationary_growth_mps: float = .005
    stationary_support_distance_m: float = .10
    stationary_lease_seconds: float = 10.
    maximum_source_gap_seconds: float = .35
    unknown_speed_mps: float = 1.
    checked_horizon_seconds: float = 1.55
    supported_background_prior: bool = False
    background_support_fraction: float = .99
    background_maximum_gap_m: float = .25
    background_position_radius_m: float = .25
    persistent_background_prior: bool = False
    background_maximum_radius_m: float = .30

    def __post_init__(self):
        if (not isinstance(self.supported_background_prior, bool)
                or not isinstance(self.persistent_background_prior, bool)
                or self.persistent_background_prior and not self.supported_background_prior):
            raise ValueError("background prior requires explicit boolean opt-in")
        for name, value in vars(self).items():
            if name in ('supported_background_prior', 'persistent_background_prior'):
                continue
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("finite positive unknown-obstacle configuration required")
        if (self.unknown_phi > 1 or self.unknown_speed_mps < 1
                or self.checked_horizon_seconds < 1.55
                or self.stationary_lease_seconds > 10
                or self.maximum_source_gap_seconds > .35
                or not .99 <= self.background_support_fraction <= 1
                or self.background_maximum_gap_m > .25
                or self.background_position_radius_m < self.background_maximum_gap_m
                or self.persistent_background_prior and (
                    self.background_maximum_radius_m > .30
                    or self.background_maximum_radius_m < max(self.background_position_radius_m,
                        self.stationary_radius_m, self.stationary_support_distance_m))):
            raise ValueError("invalid risk or original motion/horizon/source bounds")

    def guidance(self, previous=None, retained=None):
        policies = [p for p in (previous, retained) if p is not None]
        risk = max([self.unknown_phi, *[p.risk for p in policies]])
        flux = max([.5 + 1.2*risk, *[p.boundary_flux_magnitude for p in policies]])
        return SafeSAGEGuidanceParameters(ClassSafetyParameters(risk, flux),
            previous.tangential_strength if previous is not None else 1.25,
            previous.passing_side if previous is not None else "right")


@dataclass(frozen=True)
class UnknownSurface:
    track_id: int
    source_time: float
    source_frame: int
    center: tuple
    points: np.ndarray
    guidance: object
    stationary_support_time: float | None = None
    stationary_qualified_time: float | None = None
    background_support: dict | None = None
    motion_support_time: float | None = None
    motion_position_radius_m: float | None = None

    def __post_init__(self):
        points = np.array(self.points, dtype=float, copy=True)
        if (isinstance(self.track_id, bool) or not isinstance(self.track_id, (int, np.integer)) or self.track_id < 0
                or not math.isfinite(self.source_time) or self.source_time < 0
                or isinstance(self.source_frame, bool) or not isinstance(self.source_frame, (int, np.integer)) or self.source_frame < 0
                or np.shape(self.center) != (2,) or not np.isfinite(self.center).all()
                or points.ndim != 2 or points.shape[1] != 2 or not len(points) or not np.isfinite(points).all()
                or not isinstance(self.guidance, SafeSAGEGuidanceParameters)
                or self.stationary_support_time is not None and (
                    not math.isfinite(self.stationary_support_time) or self.stationary_support_time < self.source_time)
                or (self.motion_support_time is None) != (self.motion_position_radius_m is None)
                or self.motion_support_time is not None and (
                    not math.isfinite(self.motion_support_time) or self.motion_support_time < self.source_time
                    or not math.isfinite(self.motion_position_radius_m) or self.motion_position_radius_m <= 0)):
            raise ValueError("valid original positive surface and provenance required")
        points.setflags(write=False)
        object.__setattr__(self, 'points', points)


@dataclass(frozen=True)
class UncertainSurface:
    original: UnknownSurface
    radius_m: float
    mode: str
    support_distance_m: float | None = None
    replacement_track_id: int | None = None
    _cell_cache: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        if not math.isfinite(self.radius_m) or self.radius_m < 0:
            raise ValueError("finite nonnegative geometry uncertainty required")

    @property
    def geometry_id(self):
        return ID_OFFSET + self.original.track_id

    def cells(self, cfg, anchor_xy, anchor_yaw):
        """Grid-bounded work even when a generic motion region grows large."""
        key = (cfg.minimum_x, cfg.maximum_x, cfg.minimum_y, cfg.maximum_y,
            cfg.resolution, *anchor_xy, anchor_yaw)
        if key in self._cell_cache:
            return self._cell_cache[key]
        c, s = math.cos(anchor_yaw), math.sin(anchor_yaw)
        points = (self.original.points-anchor_xy) @ np.array([[c, -s], [s, c]])
        cols, rows = np.meshgrid(np.arange(cfg.column_count), np.arange(cfg.row_count))
        xy = np.column_stack((cols.ravel(), rows.ravel()))*cfg.resolution
        xy += np.array([cfg.minimum_x, cfg.minimum_y]) + .5*cfg.resolution
        distances = cKDTree(points).query(xy)[0]
        selected = distances <= self.radius_m + cfg.resolution/math.sqrt(2)
        clipped = (np.any(points.min(axis=0)-self.radius_m < [cfg.minimum_x, cfg.minimum_y])
            or np.any(points.max(axis=0)+self.radius_m >= [cfg.maximum_x, cfg.maximum_y]))
        result = (xy[selected], rows.ravel()[selected], cols.ravel()[selected], bool(clipped))
        for array in result[:3]:
            array.setflags(write=False)
        self._cell_cache[key] = result
        return result


@dataclass(frozen=True)
class UnknownObstacleSnapshot:
    source_time: float
    config: UnknownObstacleConfig
    regions: tuple

    def coverage(self, key, source_time, center, grid, anchor_xy, anchor_yaw, *, original_points=None, original_risk=None):
        matches = [r for r in self.regions if r.original.track_id == key]
        if len(matches) != 1:
            return None
        r = matches[0]
        old = r.original
        if (old.source_time != source_time or self.source_time < source_time
                or not np.allclose(old.center, center, atol=1e-9, rtol=0)):
            return None
        if (original_points is not None and not np.array_equal(old.points, original_points)
                or original_risk is not None and old.guidance.risk < original_risk):
            return None
        _, rows, cols, clipped = r.cells(grid.config, anchor_xy, anchor_yaw)
        if (not grid.occupied_mask[rows, cols].all()
                or np.any(grid.cell_risks[rows, cols] < old.guidance.risk-1e-12)
                or np.any(grid.cell_boundary_fluxes[rows, cols] > old.guidance.boundary_flux+1e-12)):
            return None
        # Outside the local field is not declared free. Existing MPC sampling
        # rejects out-of-domain/invalid edge queries, including rolling edges.
        return dict(track_id=key, source=SOURCE, status="prior", mode=r.mode,
            original_source_time=old.source_time, original_source_frame=old.source_frame,
            current_source_time=self.source_time, original_point_count=len(old.points),
            stationary_support_time=old.stationary_support_time,
            background_support=old.background_support,
            motion_support_time=old.motion_support_time,
            motion_position_radius_m=old.motion_position_radius_m,
            radius_m=r.radius_m, fixed_unknown_phi=self.config.unknown_phi,
            applied_risk=old.guidance.risk, checked_cell_count=len(rows),
            replacement_track_id=r.replacement_track_id,
            region_clipped_to_local_field=clipped, local_region_current_field_covered=True,
            scope="positive geometry prior in current local field; no free-space, identity or factor renewal")

    def diagnostic(self):
        return dict(source=SOURCE, status="prior", source_time=self.source_time,
            config=asdict(self.config), regions=[dict(track_id=r.original.track_id,
                original_source_time=r.original.source_time, original_source_frame=r.original.source_frame,
                original_point_count=len(r.original.points), center_world_xy=r.original.center,
                stationary_support_time=r.original.stationary_support_time,
                stationary_qualified_time=r.original.stationary_qualified_time,
                background_support=r.original.background_support,
                motion_support_time=r.original.motion_support_time,
                motion_position_radius_m=r.original.motion_position_radius_m,
                radius_m=r.radius_m, mode=r.mode, support_distance_m=r.support_distance_m,
                replacement_track_id=r.replacement_track_id,
                fixed_unknown_phi=self.config.unknown_phi, applied_risk=r.original.guidance.risk)
                for r in self.regions])


class UnknownObstacleMemory:
    def __init__(self, config=None):
        self.config = config or UnknownObstacleConfig()
        if not isinstance(self.config, UnknownObstacleConfig):
            raise TypeError("expected UnknownObstacleConfig")
        self.records = {}
        self.last_time = None
        self.history = []

    def _background_support(self, points, stable, tree, source_times):
        """Dominant measured support establishes a hypothesis, not all-point observation.

        Every unsupported point remains in the original occupied geometry. The
        separate position allowance covers the bounded unmatched tail. Counting
        distinct coordinates prevents duplicate returns from fabricating support.
        """
        if not self.config.supported_background_prior or tree is None:
            return None
        distances, indices = tree.query(points)
        supported = distances <= .06
        fraction = float(supported.mean())
        distinct = len(np.unique(stable[indices[supported]], axis=0))
        if (fraction < self.config.background_support_fraction or distinct < 3
                or distances.max() > self.config.background_maximum_gap_m):
            return None
        return dict(basis='repeated fixed-world support; stationary-background PRIOR',
            support_source_times=tuple(source_times), original_point_count=len(points),
            supported_point_count=int(supported.sum()), unsupported_point_count=int((~supported).sum()),
            supported_fraction=fraction, distinct_support_points=distinct,
            observed_support_distance_m=.06, maximum_gap_m=float(distances.max()),
            position_allowance_m=self.config.background_position_radius_m,
            whole_original_geometry_retained=True)

    def update(self, points, ids, tracks, guidance, *, source_time, source_frame,
               robot_xy, furniture_memory=None, excluded_points=(), human_regions=(),
               covered_human_fragments=()):
        points, ids = np.asarray(points, float), np.asarray(ids)
        current = {t.track_id: t for t in tracks}
        if (not math.isfinite(source_time) or source_time < 0
                or self.last_time is not None and source_time <= self.last_time
                or points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all()
                or ids.shape != (len(points),) or not np.issubdtype(ids.dtype, np.integer)
                or len(current) != len(tracks) or np.any(ids < -1)
                or any(int(k) not in current for k in np.unique(ids) if k >= 0)
                or any(np.count_nonzero(ids == t.track_id) != t.point_count for t in tracks)
                or np.shape(robot_xy) != (2,) or not np.isfinite(robot_xy).all()):
            raise ValueError("increasing aligned positive geometry required")
        cfg = self.config
        records = dict(self.records)
        # Current geometry is a real new observation. Factors are not copied to
        # new IDs. A known higher policy is never lowered because a mask is lost.
        for key, t in current.items():
            if key not in records and np.linalg.norm(np.asarray(t.centroid_world_xy)-robot_xy) > cfg.enrollment_distance_m:
                continue
            old = records.get(key)
            policy = cfg.guidance(guidance[key], old.guidance if old is not None else None)
            qualified = None if furniture_memory is None else furniture_memory.records.get(key)
            stationary = qualified is not None and qualified.source_time == source_time
            records[key] = UnknownSurface(key, source_time, source_frame, tuple(t.centroid_world_xy),
                points[ids == key].copy(), policy, source_time if stationary else None,
                qualified.qualification_source_time if stationary else None)
        excluded = np.asarray(excluded_points, float).reshape((-1, 2))
        exclusion_tree = cKDTree(excluded) if len(excluded) else None
        # Own raw support history excludes people. Repeated fixed-world support
        # is evaluated locally, without mistaking a changing cluster centroid
        # for physical motion of every surface in that cluster.
        clear = np.ones(len(points), bool) if exclusion_tree is None else exclusion_tree.query(points)[0] > 1.
        for center, radius in human_regions:
            clear &= np.linalg.norm(points-center, axis=1) > radius
        history = [(t, p) for t, p in self.history if 0 < source_time-t <= 2*cfg.maximum_source_gap_seconds+1e-9]
        if self.last_time is not None and source_time-self.last_time > cfg.maximum_source_gap_seconds+1e-9:
            history = []
        history = [(t, p if exclusion_tree is None else p[exclusion_tree.query(p)[0] > 1.]) for t, p in history]
        for center, radius in human_regions:
            history = [(t, p[np.linalg.norm(p-center, axis=1) > radius]) for t, p in history]
        history.append((source_time, points[clear].copy()))
        stable = np.empty((0, 2))
        if history:
            latest_time, latest = history[-1]
            previous = [(t, p) for t, p in history[:-1]
                if 0 < source_time-t <= 2*cfg.maximum_source_gap_seconds+1e-9]
            if latest_time == source_time and len(previous) >= 2 and len(latest):
                support = [cKDTree(p).query(latest)[0] <= .06 if len(p) else np.zeros(len(latest), bool)
                           for _, p in previous]
                support = np.array(support)
                early = np.array([source_time-t >= .2-1e-9 for t, _ in previous])
                good = (support.sum(axis=0) >= 2) & np.any(support & early[:, None], axis=0)
                stable = latest[good]
        stable_tree = cKDTree(stable) if len(stable) >= 3 else None
        raw_tree = cKDTree(points) if len(points) else None
        # A high centroid velocity alone can be cluster resampling. Where its
        # current geometry also lacks dominant fixed-world support, retain the
        # dynamic interpretation and exclude it from the background hypothesis.
        background = {}
        moving_points = []
        if cfg.supported_background_prior:
            for key, t in current.items():
                subset = points[ids == key]
                proof = self._background_support(subset, stable, stable_tree, [t for t, _ in history])
                if not np.all(clear[ids == key]):
                    proof = None
                background[key] = proof
                if proof is None and np.linalg.norm(t.velocity_world_xy) > .2:
                    moving_points.extend(subset)
        moving_tree = cKDTree(moving_points) if len(moving_points) else None

        def excluded_surface(surface):
            excluded = bool(exclusion_tree is not None and np.any(exclusion_tree.query(surface)[0] <= 1.))
            for center, radius in human_regions:
                excluded |= bool(np.any(np.linalg.norm(surface-center, axis=1) <= radius))
            excluded |= bool(moving_tree is not None and np.any(moving_tree.query(surface)[0] <= .06))
            return excluded

        if cfg.supported_background_prior:
            for key in current:
                if key not in records or excluded_surface(records[key].points):
                    continue
                proof = background[key]
                if proof is not None:
                    records[key] = replace(records[key], stationary_support_time=source_time,
                        stationary_qualified_time=source_time, background_support=proof)
        regions = []
        for key, old in list(records.items()):
            if key in current:
                continue
            age = source_time-old.source_time
            support_distance = None
            human_proof = next((p for p in covered_human_fragments
                if p["track_id"] == key and p["original_source_time"] == old.source_time), None)
            if human_proof is not None:
                replacement = human_proof["parent_track_id"]
                if replacement not in current:
                    raise ValueError("fragment replacement requires current validated parent")
                regions.append(UncertainSurface(old, age*1., "current_body_prior_geometry_replacement",
                    None, replacement))
                records.pop(key)
                continue
            # A short-gap WHOLE positive surface may be represented by another
            # current cluster. This is geometry continuity, not semantic ID or
            # factor transfer; the independent guard retires its old query only
            # after verifying this exact surface's actual current-field raster.
            stable_match = False
            if old.stationary_support_time is not None and stable_tree is not None:
                dist, ix = stable_tree.query(old.points)
                stable_match = bool(np.all(dist <= .10) and len(np.unique(stable[ix], axis=0)) >= 3)
            if raw_tree is not None and (age <= cfg.maximum_source_gap_seconds+1e-9 or stable_match):
                distances, nearest = raw_tree.query(old.points)
                if np.all(distances <= .10) and len(np.unique(points[nearest], axis=0)) >= 3:
                    neighbors = raw_tree.query_ball_point(old.points, .10)
                    indices = np.unique(np.concatenate(neighbors)).astype(int)
                    owners = [int(k) for k in np.unique(ids[indices]) if k >= 0
                        and np.all(cKDTree(points[ids == k]).query(old.points)[0] <= .10)]
                    unowned = np.count_nonzero(ids[indices] < 0) >= 3
                    if len(owners) == 1 and not unowned:
                        replacement = owners[0]
                        regions.append(UncertainSurface(old, .10, "current_whole_surface_geometry_replacement",
                            float(distances.max()), replacement))
                        records.pop(key)
                        if replacement in records:
                            target = records[replacement]
                            records[replacement] = replace(target, guidance=cfg.guidance(target.guidance, old.guidance))
                        continue
            human_near = excluded_surface(old.points)
            # A real motion/person contradiction invalidates the stationary
            # assumption. It does not erase geometry or stop the mission.
            if human_near and old.stationary_support_time is not None:
                # The last real geometric support is newer than the original
                # query/factor source. Start motion uncertainty there, without
                # rewriting either clock or pretending the conflict is observed
                # free space. Retain the full position allowance at that epoch.
                old = replace(old, motion_support_time=old.stationary_support_time,
                    motion_position_radius_m=cfg.background_position_radius_m
                    if old.background_support is not None else cfg.stationary_support_distance_m,
                    stationary_support_time=None, background_support=None)
            background_proof = None if human_near else self._background_support(
                old.points, stable, stable_tree, [t for t, _ in history])
            if background_proof is not None:
                old = replace(old, stationary_support_time=source_time,
                    stationary_qualified_time=old.stationary_qualified_time or source_time,
                    background_support=background_proof, motion_support_time=None, motion_position_radius_m=None)
            stationary_age = math.inf if old.stationary_support_time is None else source_time-old.stationary_support_time
            if (not human_near and old.stationary_support_time is not None and stable_tree is not None
                    and old.background_support is None
                    and self.last_time is not None
                    and source_time-self.last_time <= cfg.maximum_source_gap_seconds+1e-9):
                distances, indices = stable_tree.query(old.points)
                support_distance = float(distances.max())
                # This is an explicit 10cm POSITION PRIOR backed by repeated
                # current points, not relaxation of the legacy 6cm observation
                # test. Original query/identity/factor epochs remain unchanged.
                if support_distance <= cfg.stationary_support_distance_m and len(np.unique(stable[indices], axis=0)) >= 3:
                    old = replace(old, stationary_support_time=source_time)
                    stationary_age = 0.
            persistent = (cfg.persistent_background_prior and old.background_support is not None
                and old.stationary_support_time is not None and not human_near)
            if stationary_age <= cfg.stationary_lease_seconds or persistent:
                radius = max(cfg.stationary_radius_m + cfg.stationary_growth_mps*stationary_age,
                    cfg.stationary_support_distance_m if old.stationary_support_time != old.source_time else 0.)
                mode = "stationary_positive_support_position_prior" if stationary_age == 0 else "retained_stationary_geometry_prior"
                if old.background_support is not None:
                    radius = max(radius, cfg.background_position_radius_m + cfg.stationary_growth_mps*stationary_age)
                    mode = 'supported_background_position_prior' if stationary_age == 0 else 'retained_supported_background_prior'
                    if persistent:
                        # An explicit static-map assumption, not an extended
                        # observation lease. No source/support clock advances.
                        radius = min(radius, cfg.background_maximum_radius_m)
                        if stationary_age > cfg.stationary_lease_seconds:
                            mode = 'persistent_stationary_background_prior'
            else:
                # Preserve geometry beyond a static lease; apply the explicit
                # unknown-motion bound, including the complete control horizon.
                motion_age = age if old.stationary_support_time is None else stationary_age-cfg.stationary_lease_seconds
                base_radius = (cfg.background_position_radius_m + cfg.stationary_growth_mps*cfg.stationary_lease_seconds
                    if old.background_support is not None else cfg.stationary_support_distance_m)
                if old.motion_support_time is not None:
                    motion_age = source_time-old.motion_support_time
                    base_radius = old.motion_position_radius_m
                radius = base_radius + cfg.unknown_speed_mps*(motion_age+cfg.checked_horizon_seconds)
                mode = "unknown_motion_reachable_surface_prior"
            records[key] = old
            regions.append(UncertainSurface(old, radius, mode, support_distance))
        self.records, self.last_time, self.history = records, source_time, history[-8:]
        return UnknownObstacleSnapshot(source_time, cfg, tuple(regions))

"""Opt-in static-location evidence for separated demo props, not object truth.

Only actual current returns receive a policy. Stored bounds are an association
prior, never occupied geometry or a measured complete object extent.
"""

from dataclasses import dataclass, replace
import math

import numpy as np
from scipy.spatial import cKDTree

from robot.perception.physical_attributes import EstimateStatus
from robot.perception.persistent_object_evidence import PERSISTENT_PROPERTY_MARKER


@dataclass(frozen=True)
class StaticPropEvidenceConfig:
    position_allowance_m: float = .06
    original_support_distance_m: float = .06
    minimum_current_points: int = 8
    minimum_distinct_support: int = 8
    minimum_current_support_fraction: float = .6
    minimum_fragment_coverage: float = .8
    maximum_acquisition_range_m: float = 2.5

    def __post_init__(self):
        for name, value in vars(self).items():
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('minimum_current_points', 'minimum_distinct_support'):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 8:
                raise ValueError('at least eight current/original samples required')
        for name in ('minimum_current_support_fraction', 'minimum_fragment_coverage'):
            if getattr(self, name) > 1:
                raise ValueError('fractions must not exceed one')


class StaticPropEvidence:
    """Keep original intrinsic observations through missing visual assessments."""

    def __init__(self, config, property_config):
        self.config = config
        self.property_config = property_config
        self.records = {}

    def enroll(self, key, points, mass, contact, *, frame_id, source_time,
               robot_xy, frozen):
        """Only a currently verified image/LiDAR object can enter this map."""
        if not all(e is not None and e.is_available and e.status in
                   (EstimateStatus.MEASURED, EstimateStatus.INFERRED)
                   and PERSISTENT_PROPERTY_MARKER not in e.evidence
                   and 0 <= source_time - e.timestamp_seconds <= 1.
                   for e in (mass, contact)):
            return False
        points = np.asarray(points, dtype=float)
        if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
                or len(points) < self.config.minimum_current_points
                or np.linalg.norm(points[:, :2].mean(axis=0)-robot_xy)
                > self.config.maximum_acquisition_range_m):
            return False
        if key in self.records:
            if not frozen:
                # Chronologically latest valid acquisition, never score selection.
                self.records[key].update(mass=mass, contact=contact, frame_id=frame_id)
            return True
        lo, hi = points.min(axis=0), points.max(axis=0)
        if any(np.all(lo <= r['hi'] + 2*self.config.position_allowance_m)
               and np.all(hi >= r['lo'] - 2*self.config.position_allowance_m)
               for r in self.records.values()):
            return False
        self.records[key] = dict(points=points.copy(), lo=lo, hi=hi,
            mass=mass, contact=contact, frame_id=frame_id,
            location_source_time=source_time, location_source_frame=frame_id)
        return True

    def expire(self, timestamp):
        maximum = self.property_config.maximum_property_age_seconds
        self.records = {key:r for key,r in self.records.items()
                        if all(0 <= timestamp-r[n].timestamp_seconds <= maximum
                               for n in ('mass','contact'))}

    def match(self, world, point_ids, timestamp, *, excluded=None):
        """Unique original location + live support; no fresh SAM mask required.

        Current points may sample between original rays inside the observed
        bounds. The explicit static-location assumption supplies that identity;
        this is not a new full-surface correspondence proof.
        """
        self.expire(timestamp)
        cfg = self.config
        excluded = np.zeros(len(world), bool) if excluded is None else excluded
        regions = {key:np.all((world >= r['lo']-cfg.position_allowance_m)
                              & (world <= r['hi']+cfg.position_allowance_m), axis=1)
                   for key,r in self.records.items()}
        matches, audits = {}, []
        for key, region in regions.items():
            r = self.records[key]
            reasons = []
            if np.any(region & excluded):
                reasons.append('current excluded/person support intersects static region')
            if any(np.any(region & other) for k,other in regions.items() if k != key):
                reasons.append('current points compete between static locations')
            selected = region & (point_ids >= 0)
            # A fragment extending into another surface cannot be absorbed.
            members = []
            for raw in np.unique(point_ids[selected]):
                own = point_ids == raw
                coverage = np.count_nonzero(selected & own)/np.count_nonzero(own)
                if coverage < cfg.minimum_fragment_coverage:
                    selected[own] = False
                else:
                    members.append(dict(raw_track_id=int(raw), hits=int(np.sum(selected & own)),
                                        visible_coverage=float(coverage)))
            if selected.sum() < cfg.minimum_current_points:
                reasons.append('insufficient current static-region support')
            distinct, fraction = 0, 0.
            if selected.any():
                distances, nearest = cKDTree(r['points']).query(world[selected])
                supported = distances <= cfg.original_support_distance_m
                distinct = len(np.unique(nearest[supported]))
                fraction = float(supported.mean())
            if distinct < cfg.minimum_distinct_support or fraction < cfg.minimum_current_support_fraction:
                reasons.append('insufficient original-location anchor support')
            audit = dict(object_id=key, current_source_time=timestamp,
                location_source_time=r['location_source_time'],
                location_source_frame=r['location_source_frame'],
                observed_bounds_minimum=r['lo'].tolist(), observed_bounds_maximum=r['hi'].tolist(),
                position_allowance_m=cfg.position_allowance_m, current_points=int(selected.sum()),
                original_points=len(r['points']), distinct_original_support=distinct,
                current_support_fraction=fraction, fragments=members, rejections=reasons,
                basis='static separated-prop location PRIOR plus current LiDAR; no fresh visual mask claim')
            audits.append(audit)
            if not reasons:
                matches[key] = (selected, audit)
        return matches, audits

    def retained(self, key, timestamp):
        record = self.records[key]
        values, details = {}, {}
        for name in ('mass','contact'):
            original = record[name]
            age = timestamp-original.timestamp_seconds
            if not 0 <= age <= self.property_config.maximum_property_age_seconds:
                raise ValueError('static property expired or is future')
            values[name] = replace(original, status=EstimateStatus.PRIOR,
                confidence=original.confidence*2**(-age/self.property_config.confidence_half_life_seconds),
                evidence=(*original.evidence, PERSISTENT_PROPERTY_MARKER,
                    'static_demo_prop_location_prior; original property retained, not remeasured',
                    f'original_frame_id={record["frame_id"]}; age_seconds={age}'))
            details[name] = dict(mode='persistence_prior', source_timestamp=original.timestamp_seconds,
                source_frame_id=record['frame_id'], age_seconds=age,
                association_basis='static_prop_location_prior')
        return values, details

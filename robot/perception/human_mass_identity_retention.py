"""Explicit adult-mass PRIOR retention under verified current body identity.

This separate option never creates a mass observation or extends its source
clock. Generic object-property competitor gates remain unchanged.
"""

from dataclasses import dataclass, replace
import math

import numpy as np

from robot.perception.human_body_envelope import HumanBodyEnvelope
from robot.perception.human_mass_prior import HUMAN_MASS_PRIOR_SOURCE
from robot.perception.persistent_object_evidence import PERSISTENT_PROPERTY_MARKER
from robot.perception.physical_attributes import EstimateStatus, ScalarEstimate


HUMAN_MASS_RETENTION_MARKER = 'bounded_human_mass_body_identity_retention_v1'


@dataclass(frozen=True)
class HumanMassIdentityRetentionConfig:
    maximum_age_seconds: float = 3.
    confidence_half_life_seconds: float = 1.5
    bound_growth_fraction_per_second: float = .05

    def __post_init__(self):
        if any(isinstance(v, bool) or not math.isfinite(v) or v <= 0 for v in vars(self).values()):
            raise ValueError('positive finite mass retention limits required')
        if self.maximum_age_seconds > 3.:
            raise ValueError('adult mass source expiry cannot exceed three seconds')


class HumanMassIdentityRetention:
    def __init__(self, config=None):
        self.config = config or HumanMassIdentityRetentionConfig()
        if not isinstance(self.config, HumanMassIdentityRetentionConfig):
            raise TypeError('typed adult-mass retention configuration required')
        self.sources = {}
        self.last_time = None

    def update(self, tracking, body_memory, fresh_mass, *, timestamp_seconds, camera_frame_id):
        """Resolve only original adult priors after the current body update.

        ``fresh_mass`` must contain only same-frame accepted visual/SAM adult
        priors. The caller stages this memory with the complete physical field.
        """
        now = timestamp_seconds
        if (not isinstance(body_memory, HumanBodyEnvelope)
            or isinstance(now, bool) or not math.isfinite(now)
            or self.last_time is not None and now <= self.last_time
            or not 0 <= now-tracking.timestamp_seconds <= .25+1e-9
            or isinstance(camera_frame_id, bool) or not isinstance(camera_frame_id, int) or camera_frame_id < 0):
            raise ValueError('current verified body identity and ordered aligned camera/LiDAR epochs required')
        tracks = {t.track_id: t for t in tracking.tracked_obstacles}
        identities = body_memory.identities
        excluded = {r['track_id'] for r in body_memory.recovery_audit
            if body_memory.recovery_config is not None
            and r['decision'] == 'temporarily_outside_checked_reach'
            and r['evaluation_timestamp_seconds'] == tracking.timestamp_seconds
            and r['clearance_m'] > 0}
        active_identities = {key: value for key, value in identities.items() if key not in excluded}
        for key, identity in active_identities.items():
            track = tracks.get(key)
            count = int(np.count_nonzero(tracking.point_track_ids == key))
            if (track is None or count < 20 or identity['lidar_timestamp'] != tracking.timestamp_seconds
                or identity['observation_count'] != track.observation_count
                or not np.allclose(identity['center'], track.centroid_world_xy, atol=1e-9, rtol=0)):
                raise ValueError('adult mass retention requires current positively supported body identity')
        if not set(fresh_mass) <= active_identities.keys():
            raise ValueError('mass prior cannot transfer to an unverified body identity')
        sources = {key: value for key, value in self.sources.items()
            if key in identities and 0 <= now-value[0].timestamp_seconds <= self.config.maximum_age_seconds+1e-9}
        resolved, rows = {}, []
        for key, identity in active_identities.items():
            fresh = fresh_mass.get(key)
            if fresh is not None:
                if (not isinstance(fresh, ScalarEstimate) or not fresh.is_available
                    or fresh.status is not EstimateStatus.PRIOR or fresh.source != HUMAN_MASS_PRIOR_SOURCE
                    or fresh.unit != 'kg' or not 0 < fresh.lower_bound <= fresh.value <= fresh.upper_bound <= 300.
                    or not 0 < fresh.confidence <= .5
                    or fresh.timestamp_seconds != now
                    or identity['camera_timestamp'] != now or identity['camera_frame'] != camera_frame_id
                    or PERSISTENT_PROPERTY_MARKER in fresh.evidence or HUMAN_MASS_RETENTION_MARKER in fresh.evidence):
                    raise ValueError('seed requires original same-frame accepted adult mass prior and visual identity')
                sources[key] = (fresh, camera_frame_id)
                rows.append(dict(track_id=key, mode='fresh', source_timestamp=now, source_frame_id=camera_frame_id))
                continue
            source = sources.get(key)
            if source is None:
                rows.append(dict(track_id=key, mode='unavailable', reason='no unexpired original adult-mass prior'))
                continue
            original, frame = source
            if identity['camera_timestamp'] < original.timestamp_seconds or identity['camera_frame'] < frame:
                raise ValueError('current body identity predates the original mass association')
            age = now-original.timestamp_seconds
            growth = self.config.bound_growth_fraction_per_second*age
            resolved[key] = replace(original,
                lower_bound=max(0., original.lower_bound*(1-growth)),
                upper_bound=original.upper_bound*(1+growth),
                confidence=original.confidence*2**(-age/self.config.confidence_half_life_seconds),
                evidence=(*original.evidence, PERSISTENT_PROPERTY_MARKER, HUMAN_MASS_RETENTION_MARKER,
                    f'original_frame_id={frame}; original_timestamp={original.timestamp_seconds}; age_seconds={age}',
                    'separate adult-mass PRIOR retention under current verified body identity; no new mass observation or source renewal'))
            rows.append(dict(track_id=key, mode='persistence_prior', source_timestamp=original.timestamp_seconds,
                source_frame_id=frame, age_seconds=age, current_lidar_timestamp=tracking.timestamp_seconds,
                current_body_identity_camera_timestamp=identity['camera_timestamp'],
                current_support_count=int(np.count_nonzero(tracking.point_track_ids == key))))
        self.sources, self.last_time = sources, now
        return resolved, dict(rows=rows, camera_timestamp_seconds=now, camera_frame_id=camera_frame_id,
            missing_geometry_mass_not_resolved=sorted(excluded),
            maximum_source_age_seconds=self.config.maximum_age_seconds,
            configuration=dict(vars(self.config)),
            scope='separate body-identity adult-mass retention PRIOR only; original three-second source expiry; generic object, contact, support, plume and motion evidence unchanged')

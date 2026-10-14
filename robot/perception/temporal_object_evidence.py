"""Short-lived mass/contact memory for continuously observed geometric tracks.

Borrowed evidence is a temporal persistence PRIOR, never a fresh measurement.
Its source epoch is unchanged; confidence decays and intervals widen. Motion,
plume, support and fall height are deliberately outside this store.
"""

from dataclasses import dataclass, replace
import math

import numpy as np

from robot.perception.physical_attributes import EstimateStatus, ScalarEstimate
from robot.safety.obstacle_tracking import TrackedObstacle


TEMPORAL_EVIDENCE_MARKER = "bounded_temporal_object_evidence_v1"
CONTACT_SOURCES = {
    "visual_silhouette_contact_proxy",
    "visual_exposed_edge_geometry_proxy",
}


@dataclass(frozen=True)
class TemporalObjectEvidenceConfig:
    maximum_mass_age_seconds: float = 3.0
    maximum_contact_age_seconds: float = 1.0
    confidence_half_life_seconds: float = 1.0
    maximum_centroid_error_m: float = 0.15
    centroid_error_growth_mps: float = 0.05
    minimum_point_count_ratio: float = 0.4
    maximum_point_count_ratio: float = 2.5
    minimum_competitor_separation_m: float = 0.3
    maximum_contact_view_change_radians: float = math.radians(15)
    mass_bound_growth_fraction_per_second: float = 0.05
    contact_bound_growth_per_second: float = 0.1

    def __post_init__(self):
        for name, value in vars(self).items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if not self.minimum_point_count_ratio <= 1 <= self.maximum_point_count_ratio:
            raise ValueError("point-count ratio bounds must contain one")
        if self.maximum_contact_view_change_radians > math.pi:
            raise ValueError("contact view change must not exceed pi")


@dataclass(frozen=True)
class _Source:
    estimate: ScalarEstimate
    track: object
    sensor_time: float
    frame_id: int
    ego_xy: tuple
    last_observation_count: int


class TemporalObjectEvidence:
    def __init__(self, config=None):
        self.config = config or TemporalObjectEvidenceConfig()
        if not isinstance(self.config, TemporalObjectEvidenceConfig):
            raise TypeError("config must be TemporalObjectEvidenceConfig")
        self.sources = {}

    def has_unexpired(self, track_id, timestamp):
        return any(
            identity == track_id
            and 0
            <= timestamp - source.estimate.timestamp_seconds
            <= self._limit(kind) + 1e-9
            for (identity, kind), source in self.sources.items()
        )

    def _limit(self, kind):
        return getattr(self.config, f"maximum_{kind}_age_seconds")

    def _reject_identity(self, source, track, tracks, sensor_time, ego_xy, kind):
        if (
            not track.matched_previous_track
            or track.observation_count != source.last_observation_count + 1
        ):
            return "geometric observation continuity was broken"
        dt = sensor_time - source.sensor_time
        if dt <= 0:
            return "geometric source epoch is future or repeated"
        expected = np.asarray(source.track.centroid_world_xy) + dt * np.asarray(
            source.track.velocity_world_xy
        )
        if (
            np.linalg.norm(np.asarray(track.centroid_world_xy) - expected)
            > self.config.maximum_centroid_error_m
            + dt * self.config.centroid_error_growth_mps
        ):
            return "world centroid disagrees with the source track prediction"
        ratio = track.point_count / source.track.point_count
        if (
            not self.config.minimum_point_count_ratio
            <= ratio
            <= self.config.maximum_point_count_ratio
        ):
            return "geometric point count changed beyond the identity gate"
        if any(
            other.track_id != track.track_id
            and np.linalg.norm(
                np.asarray(other.centroid_world_xy) - track.centroid_world_xy
            )
            < self.config.minimum_competitor_separation_m
            for other in tracks
        ):
            return "nearby geometric competitor makes identity ambiguous"
        if kind == "contact":
            old_view = np.asarray(source.ego_xy) - source.track.centroid_world_xy
            new_view = np.asarray(ego_xy) - track.centroid_world_xy
            lengths = np.linalg.norm(old_view) * np.linalg.norm(new_view)
            if lengths < 1e-6:
                return "contact viewpoint is unresolved"
            angle = math.acos(
                float(np.clip(np.dot(old_view, new_view) / lengths, -1, 1))
            )
            if angle > self.config.maximum_contact_view_change_radians:
                return "contact viewpoint changed beyond the visible-proxy domain"
        return None

    def _borrow(self, source, kind, age):
        item = source.estimate
        growth = (
            age * self.config.mass_bound_growth_fraction_per_second * item.upper_bound
            if kind == "mass"
            else age * self.config.contact_bound_growth_per_second
        )
        return replace(
            item,
            lower_bound=max(0.0, item.lower_bound - growth),
            upper_bound=min(1.0, item.upper_bound + growth)
            if kind == "contact"
            else item.upper_bound + growth,
            confidence=item.confidence
            * 2 ** (-age / self.config.confidence_half_life_seconds),
            status=EstimateStatus.PRIOR,
            evidence=(
                *item.evidence,
                TEMPORAL_EVIDENCE_MARKER,
                f"original_status={item.status.value}; original_camera_frame_id={source.frame_id}; original_timestamp={item.timestamp_seconds}",
                f"age_seconds={age}; identity_track_id={source.track.track_id}; confidence_half_life_seconds={self.config.confidence_half_life_seconds}",
                "bounded property-persistence assumption; original source time retained; not a fresh camera estimate",
            ),
        )

    def resolve(
        self,
        *,
        track,
        tracked_obstacles,
        sensor_timestamp_seconds,
        timestamp_seconds,
        camera_frame_id,
        ego_position_world_xy,
        fresh_mass,
        fresh_contact,
    ):
        if (
            not all(
                math.isfinite(t) and t >= 0
                for t in (sensor_timestamp_seconds, timestamp_seconds)
            )
            or sensor_timestamp_seconds > timestamp_seconds
        ):
            raise ValueError("source times must be finite, nonnegative and ordered")
        if (
            isinstance(camera_frame_id, bool)
            or not isinstance(camera_frame_id, int)
            or camera_frame_id < 0
        ):
            raise ValueError("camera frame ID must be a nonnegative integer")
        ego = np.asarray(ego_position_world_xy, dtype=float)
        if ego.shape != (2,) or not np.isfinite(ego).all():
            raise ValueError("ego position must be a finite planar vector")
        tracks = tuple(tracked_obstacles)
        for item in (track, *tracks):
            if (
                not isinstance(item, TrackedObstacle)
                or item.point_count <= 0
                or item.observation_count <= 0
            ):
                raise ValueError("identity requires an observed geometric track")
            if not np.isfinite(
                (*item.centroid_world_xy, *item.velocity_world_xy)
            ).all():
                raise ValueError("identity references must be finite")
        if len({item.track_id for item in tracks}) != len(tracks):
            raise ValueError("geometric track IDs must be unique")
        if track.track_id not in {other.track_id for other in tracks}:
            raise ValueError("target must be observed in the current geometric tracks")
        present = {other.track_id for other in tracks}
        self.sources = {
            key: source
            for key, source in self.sources.items()
            if key[0] in present
            and 0
            <= timestamp_seconds - source.estimate.timestamp_seconds
            <= self._limit(key[1]) + 1e-9
        }
        output, diagnostics = {}, {}
        for kind, fresh, unit in (
            ("mass", fresh_mass, "kg"),
            ("contact", fresh_contact, "1"),
        ):
            if fresh is not None and (
                not isinstance(fresh, ScalarEstimate) or fresh.unit != unit
            ):
                raise ValueError(f"{kind} input must be a {unit} scalar estimate")
            key = (track.track_id, kind)
            source = self.sources.get(key)
            borrowed = None
            reason = "no unexpired source evidence for this track"
            if source is not None:
                reason = self._reject_identity(
                    source, track, tracks, sensor_timestamp_seconds, tuple(ego), kind
                )
                if reason is None:
                    borrowed = self._borrow(
                        source,
                        kind,
                        timestamp_seconds - source.estimate.timestamp_seconds,
                    )
                    self.sources[key] = replace(
                        source, last_observation_count=track.observation_count
                    )
                else:
                    del self.sources[key]
            valid_fresh = (
                fresh is not None
                and fresh.is_available
                and fresh.confidence > 0
                and fresh.status in {EstimateStatus.MEASURED, EstimateStatus.INFERRED}
                and abs(timestamp_seconds - fresh.timestamp_seconds) <= 1e-9
                and (kind != "contact" or fresh.source in CONTACT_SOURCES)
            )
            held_stronger = (
                valid_fresh
                and kind == "contact"
                and borrowed is not None
                and borrowed.value > fresh.value + 1e-12
            )
            if valid_fresh and not held_stronger:
                output[kind] = fresh
                self.sources[key] = _Source(
                    fresh,
                    track,
                    sensor_timestamp_seconds,
                    camera_frame_id,
                    tuple(ego),
                    track.observation_count,
                )
                mode = "fresh"
                reason = None
            elif borrowed is not None and (
                fresh is None or not fresh.is_available or held_stronger
            ):
                output[kind] = borrowed
                mode = "retained_stronger_contact" if held_stronger else "retained"
            else:
                output[kind] = fresh
                mode = (
                    "unavailable"
                    if fresh is None or not fresh.is_available
                    else "fresh_not_retention_eligible"
                )
            item = output[kind]
            origin = self.sources.get(key)
            diagnostics[kind] = {
                "mode": mode,
                "rejection_reason": reason,
                "source_timestamp": None if item is None else item.timestamp_seconds,
                "source_camera_frame_id": None if origin is None else origin.frame_id,
                "age_seconds": None
                if item is None
                else timestamp_seconds - item.timestamp_seconds,
                "confidence": 0.0 if item is None else item.confidence,
                "status": "unavailable" if item is None else item.status.value,
            }
        return output["mass"], output["contact"], diagnostics

"""Session-scoped property memory; retained observations become explicit priors.

This is not a free-space map or a general object-identity solution. Geometry,
motion and route interaction remain live. No oracle/fallback input is promoted
to an observed property, and borrowing never changes its acquisition epoch.
"""

from dataclasses import dataclass, replace
import math

import numpy as np

from robot.perception.physical_attributes import EstimateStatus


PERSISTENT_PROPERTY_MARKER = "session_object_property_persistence_v1"


@dataclass(frozen=True)
class PersistentObjectEvidenceConfig:
    scope: str
    maximum_property_age_seconds: float = 120.0
    confidence_half_life_seconds: float = 60.0
    maximum_prediction_error_m: float = 0.25
    minimum_competitor_separation_m: float = 0.3
    maximum_support_displacement_m: float = 0.15
    maximum_visible_height_change_m: float = 0.10
    minimum_visual_reassessment_interval_seconds: float = 1.0
    maximum_visual_tracks_per_update: int = 2
    translate_instance_bounds_prior: bool = False

    def __post_init__(self):
        if not isinstance(self.translate_instance_bounds_prior, bool):
            raise TypeError("translated bounds opt-in must be boolean")
        if not isinstance(self.scope, str) or not self.scope.strip():
            raise ValueError("property persistence requires an explicit scene scope")
        if isinstance(self.maximum_visual_tracks_per_update, bool) or not isinstance(
            self.maximum_visual_tracks_per_update, int
        ):
            raise ValueError(
                "maximum_visual_tracks_per_update must be a positive integer"
            )
        for name, value in vars(self).items():
            if name in {"scope", "translate_instance_bounds_prior"}:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class _Property:
    estimate: object
    frame_id: int
    position_world_xy: tuple


class PersistentObjectEvidence:
    """Retain intrinsic inputs for continuously associated separated tracks."""

    def __init__(self, config):
        if not isinstance(config, PersistentObjectEvidenceConfig):
            raise TypeError("config must be PersistentObjectEvidenceConfig")
        self.config = config
        self.properties = {}
        self.last_tracks = {}
        self.last_timestamp = None
        self.frozen = False
        self.support_height_observations = {}
        self.last_visual_attempt = {}

    def freeze_properties(self):
        """End explicit stationary acquisition; retain the last valid plume cue.

        A clean-reference frame alone must never freeze a low plume cue before
        the subsequent source observation window has completed.
        """
        if self.last_timestamp is None:
            raise RuntimeError("observe the scene before freezing property acquisition")
        self.frozen = True

    def has_thermal(self, track_id, timestamp):
        source = self.properties.get((track_id, "heat"))
        return (
            self.frozen
            and source is not None
            and 0
            <= timestamp - source.estimate.timestamp_seconds
            <= self.config.maximum_property_age_seconds
        )

    def resolve(
        self,
        *,
        track,
        tracks,
        attributes,
        potential,
        contact,
        timestamp,
        camera_frame_id,
    ):
        """Merge valid current intrinsic inputs with guarded session priors.

        Only the fall interaction's support assumption is retained; route
        interaction is recomputed by the caller. Object/center height is never
        turned into fall height. Motion attributes pass through unchanged.
        """
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("invalid property-map timestamp")
        previous = self.last_tracks.get(track.track_id)
        competing = any(
            other.track_id != track.track_id
            and np.linalg.norm(
                np.asarray(other.centroid_world_xy) - track.centroid_world_xy
            )
            < self.config.minimum_competitor_separation_m
            for other in tracks
        )
        reason = None
        if previous is not None:
            old, old_time = previous
            dt = timestamp - old_time
            predicted = np.asarray(old.centroid_world_xy) + dt * np.asarray(
                old.velocity_world_xy
            )
            if (
                dt <= 0
                or not track.matched_previous_track
                or track.observation_count != old.observation_count + 1
            ):
                reason = "track observation continuity was broken"
            elif (
                np.linalg.norm(np.asarray(track.centroid_world_xy) - predicted)
                > self.config.maximum_prediction_error_m
            ):
                reason = "track disagrees with identity prediction"
        if competing:
            reason = "nearby competitor makes persistence ambiguous"
        if reason:
            self.properties = {
                key: value
                for key, value in self.properties.items()
                if key[0] != track.track_id
            }
            self.support_height_observations.pop(track.track_id, None)
        center = attributes.geometry.center_height_m
        old_height = self.support_height_observations.get(track.track_id)
        support_changed = (
            center.is_available
            and 0 <= timestamp - center.timestamp_seconds <= 1
            and old_height is not None
            and abs(center.value - old_height)
            > self.config.maximum_visible_height_change_m
        )
        if support_changed:
            for name in ("fall_height", "fall_interaction"):
                self.properties.pop((track.track_id, name), None)
        if (
            center.is_available
            and 0 <= timestamp - center.timestamp_seconds <= 1
            and (old_height is None or attributes.geometry.fall_height_m.is_available)
        ):
            self.support_height_observations[track.track_id] = center.value

        fresh = {
            "mass": attributes.mass.mass_kg,
            "fall_height": attributes.geometry.fall_height_m,
            "fall_interaction": potential,
            "contact": contact,
            "heat": attributes.thermal.heat_source_likelihood,
        }
        resolved, details = {}, {}
        for name, item in fresh.items():
            key = (track.track_id, name)
            usable = (
                item is not None
                and item.is_available
                and item.status
                in {
                    EstimateStatus.MEASURED,
                    EstimateStatus.INFERRED,
                    EstimateStatus.PRIOR,
                }
                and PERSISTENT_PROPERTY_MARKER not in item.evidence
                and 0 <= timestamp - item.timestamp_seconds <= 1.0
            )
            # The user-selected mode freezes accepted intrinsic observations
            # for this declared static session. Current view-dependent network
            # outputs do not renew or replace them. Support invalidation above
            # and original expiry still take precedence.
            if (
                self.frozen
                and key in self.properties
                and 0
                <= timestamp - self.properties[key].estimate.timestamp_seconds
                <= self.config.maximum_property_age_seconds
            ):
                usable = False
            if usable:
                resolved[name] = item
                details[name] = {
                    "mode": "fresh",
                    "source_timestamp": item.timestamp_seconds,
                    "source_frame_id": camera_frame_id,
                }
                if not competing:
                    self.properties[key] = _Property(
                        item, camera_frame_id, track.centroid_world_xy
                    )
                continue
            source = self.properties.get(key)
            rejection = reason
            if support_changed and name in {"fall_height", "fall_interaction"}:
                rejection = "visible height changed; observed support must be reassessed (height is not fall height)"
            if source is not None:
                age = timestamp - source.estimate.timestamp_seconds
                displacement = np.linalg.norm(
                    np.asarray(track.centroid_world_xy) - source.position_world_xy
                )
                if not 0 <= age <= self.config.maximum_property_age_seconds:
                    rejection = "original property source expired or is future"
                elif (
                    name in {"fall_height", "fall_interaction"}
                    and displacement > self.config.maximum_support_displacement_m
                ):
                    rejection = "object left its observed support location"
                if rejection is None:
                    original = source.estimate
                    resolved[name] = replace(
                        original,
                        status=EstimateStatus.PRIOR,
                        confidence=original.confidence
                        * 2 ** (-age / self.config.confidence_half_life_seconds),
                        evidence=(
                            *original.evidence,
                            PERSISTENT_PROPERTY_MARKER,
                            f"scope={self.config.scope}",
                            f"original_status={original.status.value}; original_frame_id={source.frame_id}; original_timestamp={original.timestamp_seconds}; age_seconds={age}",
                            "property assumed unchanged; retained visible proxy is not hidden-surface safety or temperature",
                        ),
                    )
                    details[name] = {
                        "mode": "persistence_prior",
                        "source_timestamp": original.timestamp_seconds,
                        "source_frame_id": source.frame_id,
                        "age_seconds": age,
                    }
                    continue
                del self.properties[key]
            resolved[name] = item
            details[name] = {
                "mode": "unavailable",
                "reason": rejection or "no accepted original property",
            }
        self.last_tracks[track.track_id] = (track, timestamp)
        attributes = replace(
            attributes,
            mass=replace(attributes.mass, mass_kg=resolved["mass"]),
            geometry=replace(
                attributes.geometry, fall_height_m=resolved["fall_height"]
            ),
            thermal=replace(
                attributes.thermal, heat_source_likelihood=resolved["heat"]
            ),
        )
        return attributes, resolved["fall_interaction"], resolved["contact"], details

    def finish_epoch(self, tracks, timestamp):
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            raise ValueError("property map epochs must increase")
        active = {track.track_id for track in tracks}
        self.properties = {
            key: value
            for key, value in self.properties.items()
            if key[0] in active
            and 0
            <= timestamp - value.estimate.timestamp_seconds
            <= self.config.maximum_property_age_seconds
        }
        self.last_tracks = {
            key: value for key, value in self.last_tracks.items() if key in active
        }
        self.support_height_observations = {
            key: value
            for key, value in self.support_height_observations.items()
            if key in active
        }
        self.last_visual_attempt = {
            key: value
            for key, value in self.last_visual_attempt.items()
            if key in active
        }
        self.last_timestamp = timestamp

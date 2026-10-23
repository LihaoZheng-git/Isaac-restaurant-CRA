"""Opt-in source-window position fit for encounter policy only.

Visible articulated surfaces can move the observed centroid between scans.
This is an inferred average velocity over original observations, not a renewed
measurement, a retained crossing label, or an obstacle-motion safety bound.
"""

from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class EncounterMotionEstimateConfig:
    window_seconds: float = 0.4
    minimum_span_seconds: float = 0.25
    maximum_gap_seconds: float = 0.25
    minimum_samples: int = 4
    minimum_points: int = 20
    maximum_fit_residual_m: float = 0.06
    maximum_speed_mps: float = 2.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not self.minimum_span_seconds < self.window_seconds <= 1.0:
            raise ValueError("motion fit needs a positive span below a window <=1 second")
        for name in ("minimum_samples", "minimum_points"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 3:
                raise ValueError(f"{name} must be an integer >=3")


class EncounterMotionEstimate:
    """Fit only current positively supported IDs; gaps/identity loss clear history.

    The caller stages this state with perception. A rejected fit leaves the
    original instantaneous tracker velocity available to the legacy classifier.
    Neither this fit nor that fallback changes CRA, occupancy or kinetic inputs.
    """

    def __init__(self, config=None):
        self.config = config or EncounterMotionEstimateConfig()
        if not isinstance(self.config, EncounterMotionEstimateConfig):
            raise TypeError("encounter motion estimate configuration required")
        self.history = {}
        self.last_timestamp = None

    def update(self, tracking, human_track_ids):
        cfg = self.config
        now = float(tracking.timestamp_seconds)
        if not math.isfinite(now) or now < 0 or self.last_timestamp is not None and now <= self.last_timestamp:
            raise ValueError("motion fit source epochs must strictly increase")
        tracks = {t.track_id: t for t in tracking.tracked_obstacles}
        ids = tuple(human_track_ids)
        if len(tracks) != len(tracking.tracked_obstacles) or len(set(ids)) != len(ids) or any(k not in tracks for k in ids):
            raise ValueError("unique current supported human IDs required")
        history, estimates, rows = {}, {}, {}
        for key in ids:
            track = tracks[key]
            position, raw = (np.asarray(v, dtype=float) for v in (track.centroid_world_xy, track.velocity_world_xy))
            if any(v.shape != (2,) or not np.isfinite(v).all() for v in (position, raw)):
                raise ValueError("finite current centroid and raw velocity required")
            row = dict(track_id=key, source_timestamp_seconds=now,
                raw_tracker_velocity_world_xy=raw.tolist(), accepted=False,
                current_point_count=track.point_count,
                current_observation_count=track.observation_count,
                current_matched_previous_track=track.matched_previous_track,
                basis="source_window_centroid_linear_fit_for_encounter_policy_only",
                status="unavailable", reason="insufficient_source_window",
                scope="inferred window-average motion, no extrapolation; physical tracking/CRA/occupancy unchanged")
            rows[key] = row
            if not track.matched_previous_track or track.observation_count < 3 or track.point_count < cfg.minimum_points:
                row["reason"] = "current_identity_or_positive_support_unqualified"
                continue
            samples = self.history.get(key, ())
            if samples and now - samples[-1][0] > cfg.maximum_gap_seconds + 1e-9:
                samples = ()
            samples = [s for s in samples if now - s[0] <= cfg.window_seconds + 1e-9]
            samples.append((now, tuple(position)))
            history[key] = samples
            row["original_source_timestamps_seconds"] = [s[0] for s in samples]
            row["source_span_seconds"] = now - samples[0][0]
            if len(samples) < cfg.minimum_samples or row["source_span_seconds"] < cfg.minimum_span_seconds - 1e-9:
                continue
            times = np.array([s[0] - now for s in samples])
            points = np.array([s[1] for s in samples])
            centered = times - times.mean()
            velocity = centered @ (points - points.mean(axis=0)) / (centered @ centered)
            predicted = points.mean(axis=0) + centered[:, None] * velocity
            residual = float(np.max(np.linalg.norm(points - predicted, axis=1)))
            row.update(fitted_velocity_world_xy=velocity.tolist(), maximum_residual_m=residual)
            if residual > cfg.maximum_fit_residual_m:
                row["reason"] = "inconsistent_position_fit"
            elif np.linalg.norm(velocity) > cfg.maximum_speed_mps or np.linalg.norm(raw) > cfg.maximum_speed_mps:
                row["reason"] = "speed_outside_validated_scope"
            elif float(velocity @ raw) < -1e-9:
                # A new opposing current observation cannot inherit old motion.
                row["reason"] = "current_velocity_opposes_window_fit"
            else:
                row.update(accepted=True, status="inferred", reason="bounded_current_source_window_fit")
                estimates[key] = tuple(velocity)
        self.history, self.last_timestamp = history, now
        return estimates, dict(source_timestamp_seconds=now, tracks=rows, configuration=asdict(cfg))

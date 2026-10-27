"""Separate planning PRIOR: observed-source human motion, never future evidence.

The original current occupancy/field/CBF checks remain mandatory. This model
adds future envelope constraints to MPC; it does not change physical kinetic
inputs or establish a general human-motion bound.
"""

from dataclasses import asdict, dataclass
import math

import numpy as np

from robot.perception.human_body_envelope import HUMAN_ENVELOPE_SOURCE


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f'{name} requires a nonnegative integer')
    return int(value)


def _real(value, name, *, minimum=0.):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)) or not math.isfinite(value) or value < minimum:
        raise ValueError(f'{name} requires a finite value >= {minimum}')
    return float(value)


def _xy(value, name):
    a = np.asarray(value, dtype=float)
    if a.shape != (2,) or not np.isfinite(a).all():
        raise ValueError(f'{name} requires a finite two-vector')
    return tuple(map(float, a))


@dataclass(frozen=True)
class HumanGeometryForecastConfig:
    horizon_seconds: float = 1.2
    maximum_source_age_seconds: float = .5
    velocity_uncertainty_prior_mps: float = .25
    history_window_seconds: float = .8
    minimum_span_seconds: float = .5
    minimum_samples: int = 6
    maximum_gap_seconds: float = .25
    maximum_fit_residual_m: float = .1
    maximum_speed_mps: float = 2.
    minimum_points: int = 20

    def __post_init__(self):
        for name, value in asdict(self).items():
            _real(value, name, minimum=1e-12)
        for name in ('minimum_samples', 'minimum_points'):
            _integer(getattr(self, name), name)
            if getattr(self, name) < 3:
                raise ValueError('source fit requires at least three samples/points')
        if (not self.minimum_span_seconds < self.history_window_seconds <= 1.
            or not .2 <= self.horizon_seconds <= 1.2 or self.maximum_gap_seconds > .25
            or self.maximum_source_age_seconds > .5):
            raise ValueError('forecast requires bounded source window <=1s and horizon in[.2,1.2]s')


@dataclass(frozen=True)
class HumanEnvelopePrediction:
    track_id: int
    source_timestamps_seconds: tuple[float, ...]
    centroids_world_xy: tuple[tuple[float, float], ...]
    raw_velocity_world_xy: tuple[float, float]
    radius_prior_m: float
    original_person_camera_timestamp: float
    original_person_camera_frame_id: int
    current_support_count: int
    config: HumanGeometryForecastConfig

    def __post_init__(self):
        if not isinstance(self.config, HumanGeometryForecastConfig):
            raise TypeError('explicit forecast configuration required')
        cfg = self.config
        for name in ('track_id', 'original_person_camera_frame_id', 'current_support_count'):
            _integer(getattr(self, name), name)
        _real(self.radius_prior_m, 'body radius', minimum=1e-12)
        _real(self.original_person_camera_timestamp, 'original image epoch')
        times = np.asarray(self.source_timestamps_seconds, float)
        points = np.asarray(self.centroids_world_xy, float)
        raw = np.asarray(_xy(self.raw_velocity_world_xy, 'raw velocity'))
        if (times.ndim != 1 or len(times) < cfg.minimum_samples or points.shape != (len(times), 2)
            or not np.isfinite(times).all() or not np.isfinite(points).all()
            or np.any(times < 0) or np.any(np.diff(times) <= 0)
            or np.max(np.diff(times)) > cfg.maximum_gap_seconds + 1e-9
            or not cfg.minimum_span_seconds-1e-9 <= times[-1]-times[0] <= cfg.history_window_seconds+1e-9
            or self.current_support_count < cfg.minimum_points
            or not -.25-1e-9 <= times[-1]-self.original_person_camera_timestamp <= 30.):
            raise ValueError('forecast requires original ordered current source window/support/image epochs')
        centered = times - times.mean()
        velocity = centered @ (points-points.mean(axis=0)) / (centered @ centered)
        residual = float(np.max(np.linalg.norm(points-(points.mean(axis=0)+centered[:, None]*velocity), axis=1)))
        if (residual > cfg.maximum_fit_residual_m
            or max(np.linalg.norm(velocity), np.linalg.norm(raw)) > cfg.maximum_speed_mps
            or min(np.linalg.norm(raw), np.linalg.norm(velocity)) > .2 and raw @ velocity < 0):
            raise ValueError('current motion fit exceeds residual/speed/direction scope')
        object.__setattr__(self, 'source_timestamps_seconds', tuple(map(float, times)))
        object.__setattr__(self, 'centroids_world_xy', tuple(map(tuple, points)))
        object.__setattr__(self, 'raw_velocity_world_xy', tuple(raw))

    @property
    def source_timestamp_seconds(self):
        return self.source_timestamps_seconds[-1]

    @property
    def velocity_world_xy(self):
        t = np.asarray(self.source_timestamps_seconds)
        p = np.asarray(self.centroids_world_xy)
        a = t - t.mean()
        return a @ (p-p.mean(axis=0)) / (a @ a)

    def diagnostic(self):
        return dict(**asdict(self), source_timestamp_seconds=self.source_timestamp_seconds,
            velocity_world_xy=self.velocity_world_xy.tolist(), status='prior',
            velocity_basis='inferred_current_LiDAR_centroid_window_fit_for_planning_only',
            forecast_basis='constant_velocity_with_configured_uncertainty_growth_PRIOR',
            scope='additional MPC geometry hypothesis; original current field still required; not CRA kinetic evidence or a general motion bound')


@dataclass(frozen=True)
class PredictiveHumanGeometry:
    sensor_frame_id: int
    sensor_timestamp_seconds: float
    anchor_position_world_xy: tuple[float, float]
    anchor_yaw: float
    required_track_ids: tuple[int, ...]
    forecasts: tuple[HumanEnvelopePrediction, ...]
    config: HumanGeometryForecastConfig

    def __post_init__(self):
        _integer(self.sensor_frame_id, 'source frame')
        _real(self.sensor_timestamp_seconds, 'source epoch')
        _xy(self.anchor_position_world_xy, 'anchor')
        if (not isinstance(self.config, HumanGeometryForecastConfig)
            or isinstance(self.anchor_yaw, (bool, np.bool_)) or not math.isfinite(self.anchor_yaw)):
            raise ValueError('finite anchor and explicit configuration required')
        ids = tuple(_integer(k, 'required track') for k in self.required_track_ids)
        if any(not isinstance(f, HumanEnvelopePrediction) for f in self.forecasts):
            raise TypeError('typed human envelope predictions required')
        actual = [f.track_id for f in self.forecasts]
        if (len(set(ids)) != len(ids) or len(set(actual)) != len(actual) or not set(actual) <= set(ids)
            or any(f.config != self.config or f.source_timestamp_seconds != self.sensor_timestamp_seconds for f in self.forecasts)):
            raise ValueError('unique current-source forecasts must belong to required current human IDs')
        object.__setattr__(self, 'anchor_position_world_xy', _xy(self.anchor_position_world_xy, 'anchor'))
        object.__setattr__(self, 'required_track_ids', ids)
        object.__setattr__(self, 'forecasts', tuple(self.forecasts))

    @property
    def ready(self):
        return len(self.forecasts) == len(self.required_track_ids)

    def validate_source(self, frame, timestamp, anchor, yaw):
        if (self.sensor_frame_id != frame or self.sensor_timestamp_seconds != timestamp
            or not np.allclose(anchor, self.anchor_position_world_xy, atol=1e-12, rtol=0)
            or abs(yaw-self.anchor_yaw) > 1e-12):
            raise ValueError('forecast cannot transfer or renew its field/frame/anchor source')

    def constraints(self, chassis_states, times, footprint, lookahead_distance):
        """Exact oriented-rectangle/circle margins and chassis-state gradients.

        This is additional geometry. Original current C-space/DTCBF checks are
        untouched. The footprint is stored relative to the forward lookahead;
        recover its original chassis extents before computing the rectangle SDF.
        """
        states = np.asarray(chassis_states, float)
        times = np.asarray(times, float)
        from robot.safety.safe_sage_configuration_space import RectangularRobotFootprint
        if not isinstance(footprint, RectangularRobotFootprint):
            raise TypeError('prediction requires the unchanged rectangular robot footprint')
        if (states.shape != (len(times), 3) or not np.isfinite(states).all()
            or not np.isfinite(times).all() or np.any(times < 0)
            or np.any(times > self.config.horizon_seconds+self.config.maximum_source_age_seconds+1e-9) or not self.ready):
            raise ValueError('ready current forecasts and bounded finite query epochs required')
        n, b = len(times), len(self.forecasts)
        margins, gradients = np.empty((n, b)), np.empty((n, b, 3))
        front = footprint.front_extent + lookahead_distance
        rear = footprint.rear_extent - lookahead_distance
        if min(front, rear, footprint.left_extent, footprint.right_extent) < 0:
            raise ValueError('lookahead footprint cannot reconstruct chassis geometry')
        box_center = np.array([(front-rear)/2, (footprint.left_extent-footprint.right_extent)/2])
        half = np.array([(front+rear)/2, (footprint.left_extent+footprint.right_extent)/2])+footprint.safety_margin
        ca, sa = math.cos(self.anchor_yaw), math.sin(self.anchor_yaw)
        rotation = np.array([[ca, -sa], [sa, ca]])
        cs, sn = np.cos(states[:, 2]), np.sin(states[:, 2])
        for j, forecast in enumerate(self.forecasts):
            center = (np.array(forecast.centroids_world_xy[-1])-self.anchor_position_world_xy) @ rotation
            velocity = forecast.velocity_world_xy @ rotation
            delta = center + times[:, None]*velocity - states[:, :2]
            local = np.stack((cs*delta[:, 0]+sn*delta[:, 1], -sn*delta[:, 0]+cs*delta[:, 1]), axis=1)
            centered = local-box_center
            q = np.abs(centered)-half
            outside = np.maximum(q, 0.)
            distance = np.linalg.norm(outside, axis=1)
            margins[:, j] = distance + np.minimum(np.max(q, axis=1), 0.) - (
                forecast.radius_prior_m + self.config.velocity_uncertainty_prior_mps*times)
            grad_local = np.zeros((n, 2))
            outer = distance > 1e-12
            grad_local[outer] = np.sign(centered[outer])*outside[outer]/distance[outer, None]
            inner = np.flatnonzero(~outer)
            axes = np.argmax(q[inner], axis=1)
            grad_local[inner, axes] = np.sign(centered[inner, axes])
            gradients[:, j, 0] = -(cs*grad_local[:, 0]-sn*grad_local[:, 1])
            gradients[:, j, 1] = -(sn*grad_local[:, 0]+cs*grad_local[:, 1])
            gradients[:, j, 2] = grad_local[:, 0]*local[:, 1]-grad_local[:, 1]*local[:, 0]
        return margins, gradients

    def diagnostic(self):
        return dict(sensor_frame_id=self.sensor_frame_id, sensor_timestamp_seconds=self.sensor_timestamp_seconds,
            anchor_position_world_xy=self.anchor_position_world_xy, anchor_yaw=self.anchor_yaw,
            required_track_ids=self.required_track_ids, ready=self.ready,
            forecasts=[f.diagnostic() for f in self.forecasts], configuration=asdict(self.config),
            status='prior', scope='additional predictive MPC geometry only; original occupancy/field/CRA and source clocks unchanged')


class PredictiveHumanGeometryMemory:
    """Transactional source history; missing fits remain explicitly unavailable."""

    def __init__(self, config=None):
        self.config = config or HumanGeometryForecastConfig()
        if not isinstance(self.config, HumanGeometryForecastConfig):
            raise TypeError('forecast configuration required')
        self.history, self.last_timestamp = {}, None

    def update(self, tracking, envelopes, *, sensor_frame_id, anchor, anchor_yaw):
        now = _real(tracking.timestamp_seconds, 'current LiDAR epoch')
        if self.last_timestamp is not None and now <= self.last_timestamp:
            raise ValueError('prediction source epochs must strictly increase')
        tracks = {t.track_id: t for t in tracking.tracked_obstacles}
        ids = [e['track_id'] for e in envelopes]
        if len(set(ids)) != len(ids) or len(tracks) != len(tracking.tracked_obstacles):
            raise ValueError('unique current body identities required')
        history, forecasts, audit = {}, [], []
        for e in envelopes:
            key = e['track_id']
            track = tracks.get(key)
            if (track is None or e.get('source') != HUMAN_ENVELOPE_SOURCE or e.get('status') != 'prior'
                or e['lidar_timestamp_seconds'] != now or e.get('applied_to_occupancy') is not True
                or not np.allclose(e['center_world_xy'], track.centroid_world_xy, rtol=0, atol=1e-9)):
                raise ValueError('prediction requires the original verified active current body envelope')
            row = dict(track_id=key, source_timestamp_seconds=now, status='unavailable', reason='source_window_warmup')
            audit.append(row)
            if (not track.matched_previous_track or track.observation_count < 3
                or min(track.point_count, e['current_support_count']) < self.config.minimum_points):
                row['reason'] = 'current_identity_or_positive_support'
                continue
            old = self.history.get(key, ())
            if old and now-old[-1][0] > self.config.maximum_gap_seconds+1e-9:
                old = ()
            samples = [s for s in old if now-s[0] <= self.config.history_window_seconds+1e-9]
            samples.append((now, _xy(track.centroid_world_xy, 'current centroid')))
            history[key] = samples
            if len(samples) < self.config.minimum_samples or now-samples[0][0] < self.config.minimum_span_seconds-1e-9:
                continue
            try:
                forecast = HumanEnvelopePrediction(key, tuple(s[0] for s in samples), tuple(s[1] for s in samples),
                    _xy(track.velocity_world_xy, 'current velocity'), e['radius_m'],
                    e['original_person_camera_timestamp'], e['original_person_camera_frame_id'],
                    e['current_support_count'], self.config)
            except ValueError as error:
                row['reason'] = str(error)
                continue
            forecasts.append(forecast)
            row.update(status='prior', reason='current_source_window_fit', velocity_world_xy=forecast.velocity_world_xy.tolist())
        result = PredictiveHumanGeometry(sensor_frame_id, now, tuple(anchor), anchor_yaw,
            tuple(ids), tuple(forecasts), self.config)
        self.history, self.last_timestamp = history, now
        return result, dict(**result.diagnostic(), fits=audit)

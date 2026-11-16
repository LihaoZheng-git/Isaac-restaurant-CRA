"""Separate early away-side preference from current observed human geometry.

Quiet motion is not evidence of future walking intent. This preparation only
changes the safety planner's reference; occupancy, CRA and exact checks stay
mandatory. The primary motion-aware selector is kept separately.
"""

import copy
from dataclasses import asdict, dataclass
import math

import numpy as np

from robot.safety.predictive_human_geometry import PredictiveHumanGeometry
from robot.safety.safe_sage_guidance_policy import PassingSide


@dataclass(frozen=True)
class ParallelPreparationConfig:
    maximum_distance_m: float = 3.8
    maximum_lateral_separation_m: float = 1.6
    minimum_lateral_separation_m: float = .15
    quiet_speed_mps: float = .2
    minimum_oncoming_alignment: float = math.cos(math.pi/6)
    maximum_selection_seconds: float = 8.
    maximum_source_gap_seconds: float = .35
    exit_lateral_separation_m: float | None = None

    def __post_init__(self):
        values = {k: v for k, v in asdict(self).items() if k != 'exit_lateral_separation_m' or v is not None}
        if any(isinstance(v, bool) or not math.isfinite(v) or v <= 0 for v in values.values()):
            raise ValueError('positive finite preparation limits required')
        if (self.maximum_selection_seconds > 8 or self.maximum_source_gap_seconds > .35
            or self.maximum_distance_m > 3.8 or self.maximum_lateral_separation_m > 1.6
            or self.minimum_lateral_separation_m < .15 or self.minimum_oncoming_alignment > 1):
            raise ValueError('preparation must retain its bounded distance/side/source/selection scope')
        if (self.maximum_lateral_separation_m < self.minimum_lateral_separation_m
            or self.exit_lateral_separation_m is not None
            and not self.maximum_lateral_separation_m <= self.exit_lateral_separation_m <= 2.2):
            raise ValueError('exit hysteresis must retain the initial gate and be at most2.2m')


class ParallelPreparation:
    def __init__(self, config=None):
        self.config = config or ParallelPreparationConfig()
        if not isinstance(self.config, ParallelPreparationConfig):
            raise TypeError('typed preparation configuration required')
        self.encounter = None
        self.last_time = None

    def update(self, geometry, *, robot_xy, route_direction_world_xy, visual_evidence,
               camera_timestamp_seconds, camera_frame_id):
        if not isinstance(geometry, PredictiveHumanGeometry):
            raise TypeError('preparation requires current verified human forecasts')
        now, cfg = geometry.sensor_timestamp_seconds, self.config
        robot, direction = np.asarray(robot_xy, float), np.asarray(route_direction_world_xy, float)
        if (robot.shape != (2,) or direction.shape != (2,) or not np.isfinite(robot).all()
            or not np.isfinite(direction).all() or abs(np.linalg.norm(direction)-1) > 1e-9
            or self.last_time is not None and now <= self.last_time
            or not math.isfinite(camera_timestamp_seconds)
            or not 0 <= camera_timestamp_seconds-now <= .25+1e-9):
            raise ValueError('current aligned source and obstacle-unaware nominal route direction required')
        fresh = {}
        for key, cue in visual_evidence.items():
            if (isinstance(key, bool) or not isinstance(key, (int, np.integer))
                or cue.get('camera_frame_id') != camera_frame_id
                or cue.get('camera_timestamp_seconds') != camera_timestamp_seconds
                or not isinstance(cue.get('basis'), str) or not cue['basis']):
                raise ValueError('initial preparation selection needs the original same-frame visual person cue')
            fresh[int(key)] = copy.deepcopy(cue)
        previous = self.encounter
        lateral_limit = (cfg.exit_lateral_separation_m or cfg.maximum_lateral_separation_m
            if previous is not None else cfg.maximum_lateral_separation_m)
        qualified, rejected = {}, {}
        for f in geometry.forecasts if geometry.ready else ():
            relative = np.asarray(f.centroids_world_xy[-1])-robot
            speed = float(np.linalg.norm(f.velocity_world_xy))
            lateral = float(direction[0]*relative[1]-direction[1]*relative[0])
            alignment = -float(direction@f.velocity_world_xy)/max(speed, 1e-9)
            reasons = []
            if relative@direction <= 0 or np.linalg.norm(relative) > cfg.maximum_distance_m:
                reasons.append('outside_forward_preparation')
            if not cfg.minimum_lateral_separation_m <= abs(lateral) <= lateral_limit:
                reasons.append('ambiguous_or_distant_away_side')
            if speed > cfg.quiet_speed_mps and alignment < cfg.minimum_oncoming_alignment:
                reasons.append('current_fit_not_quiet_or_oncoming')
            if reasons:
                rejected[f.track_id] = reasons
            else:
                qualified[f.track_id] = dict(track_id=f.track_id,
                    side=(PassingSide.LEFT if lateral < 0 else PassingSide.RIGHT).value,
                    current_center_world_xy=list(f.centroids_world_xy[-1]),
                    fitted_velocity_world_xy=f.velocity_world_xy.tolist(), current_support_count=f.current_support_count,
                    current_fit_source_times=f.source_timestamps_seconds,
                    basis='quiet_current_person_geometry' if speed <= cfg.quiet_speed_mps else 'current_oncoming_fit')
        decision, sides = 'no_qualified_preparation', {}
        if previous is not None:
            key = previous['track_id']
            if (len(qualified) != 1 or key not in qualified
                or qualified[key]['side'] != previous['side']
                or now-previous['selected_timestamp_seconds'] > cfg.maximum_selection_seconds
                or now-self.last_time > cfg.maximum_source_gap_seconds+1e-9):
                self.encounter = None
                decision = 'released_current_geometry_identity_motion_or_expiry'
            else:
                sides[key] = PassingSide(previous['side'])
                decision = 'held_current_preparation_policy'
        elif len(qualified) == 1:
            key, row = next(iter(qualified.items()))
            if key in fresh:
                self.encounter = dict(track_id=key, side=row['side'], selected_timestamp_seconds=now,
                    original_visual_evidence=fresh[key])
                sides[key] = PassingSide(row['side'])
                decision = 'selected_current_person_preparation'
            else:
                decision = 'no_fresh_visual_person_for_new_selection'
        elif len(qualified) > 1:
            decision = 'ambiguous_multiple_preparation_candidates'
        self.last_time = now
        return sides, dict(decision=decision, status='prior', source_timestamp_seconds=now,
            source_frame_id=geometry.sensor_frame_id, current_robot_world_xy=robot.tolist(),
            camera_timestamp_seconds=camera_timestamp_seconds, camera_frame_id=camera_frame_id,
            nominal_route_direction_world_xy=direction.tolist(), current_fit_ready=geometry.ready,
            qualification_mode='holding' if previous is not None else 'initial',
            current_lateral_limit_m=lateral_limit,
            current_visual_selection_evidence=fresh, original_selection=copy.deepcopy(self.encounter),
            qualified_candidates=list(qualified.values()), rejections=rejected,
            preferences={k: v.value for k, v in sides.items()}, configuration=asdict(cfg),
            scope='separate current-person geometric preparation POLICY PRIOR; no walking-intent inference, future measurement, factor/occupancy change or renewed source clock')

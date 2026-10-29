"""Explicit short-gap geometry/identity prior, separate from visual evidence.

No actor truth, category inference or factor values enter this policy. The
reachable-disk test is deliberately independent of the desired passing side.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class HumanTrackRecoveryConfig:
    maximum_gap_seconds: float = .6
    maximum_human_speed_mps: float = 1.
    checked_horizon_seconds: float = 1.55  # 1.2 s MPC plus .35 s field age
    maximum_robot_speed_mps: float = .25
    robot_radius_m: float = math.hypot(.587351 + .05, .2466 + .05)

    def __post_init__(self):
        for name, value in vars(self).items():
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.maximum_gap_seconds > .6:
            raise ValueError("recovery is limited to a .6 second original-source gap")
        if self.checked_horizon_seconds < 1.55 or self.maximum_robot_speed_mps < .25:
            raise ValueError("recovery must cover the full horizon, field age and robot speed")

    def missing_path_clearance(self, previous, *, now, robot_xy, body_radius_m):
        robot = np.asarray(robot_xy, dtype=float)
        if robot.shape != (2,) or not np.isfinite(robot).all():
            raise ValueError("bounded recovery requires current finite robot position")
        age = now - previous['lidar_timestamp']
        if not 0 < age <= self.maximum_gap_seconds + 1e-9:
            return None
        if np.linalg.norm(previous['velocity']) > self.maximum_human_speed_mps:
            return None
        # All possible directions, including an immediate reversal. This is a
        # configured speed bound, NOT constant-velocity prediction/free space.
        human_reach = body_radius_m + self.maximum_human_speed_mps * (
            age + self.checked_horizon_seconds)
        robot_reach = self.robot_radius_m + self.maximum_robot_speed_mps * self.checked_horizon_seconds
        clearance = float(np.linalg.norm(previous['center'] - robot) - human_reach - robot_reach)
        return dict(
            basis='bounded_missing_human_outside_full_robot_reach', status='prior',
            original_lidar_timestamp_seconds=previous['lidar_timestamp'],
            evaluation_timestamp_seconds=now, original_source_age_seconds=age,
            last_center_world_xy=previous['center'].tolist(), robot_world_xy=robot.tolist(),
            human_reachable_radius_m=human_reach, robot_reachable_radius_m=robot_reach,
            clearance_m=clearance, checked_horizon_seconds=self.checked_horizon_seconds,
            maximum_human_speed_prior_mps=self.maximum_human_speed_mps,
            original_person_camera_timestamp=previous['camera_timestamp'],
            scope='short separated walking-person gap; rechecked every field, no identity deletion or clock renewal')

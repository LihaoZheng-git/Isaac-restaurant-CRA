"""Current raw LiDAR crossing request for the explicit right-side trials."""

import math

import numpy as np

from robot.safety.encounter_motion import validated_route_direction


_MAX_SOURCE_GAP_S = .35
_MAX_VELOCITY_CHANGE_MPS = .35
_MAX_MOTION_RESIDUAL_M = .20
_MIN_MOVING_SPEED_MPS = .20
_MAX_MOVING_SPEED_MPS = 2.00
_MIN_TRANSVERSE_FRACTION = .60
_MIN_CONSECUTIVE_MOTION_SOURCES = 3
_MIN_INITIAL_RAW_POINTS = 20
_MIN_CONTINUING_RAW_POINTS = 8
_MIN_LANE_SUPPORT_POINTS = 3
_MAX_FORWARD_RANGE_M = 2.60
_FORECAST_SECONDS = .90
_LANE_HALF_WIDTH_M = .2966
_SEGMENT_END_TOLERANCE_M = .30


class RightCrossingLiDARYield:
    """Request an already-checked zero plan from live transverse occupancy.

    Stored state only verifies motion continuity and distinguishes initiation
    from continuation. Every request needs current raw returns. No obstacle
    cells, person class, factor estimates, or clocks are changed here.
    """

    def __init__(self, case=None):
        if case not in (None, "boy", "trolley", "delivery"):
            raise ValueError("Unknown right crossing case")
        self.case = case
        self.previous_source = None
        self.previous_lane = None
        self.motion = {}
        self.active = ()

    def update(self, tracking, world_xy, *, robot_xy, route_direction=None, goal_xy=None,
               lane_world_xy=None, current_visual_yield_ids=()):
        source = float(tracking.timestamp_seconds)
        if (not math.isfinite(source) or source < 0.
            or self.previous_source is not None and source <= self.previous_source):
            raise ValueError("Crossing LiDAR source times must strictly increase")
        robot = np.asarray(robot_xy, dtype=float)
        if robot.shape != (2,) or not np.isfinite(robot).all():
            raise ValueError("Finite planar robot position required")
        if route_direction is None and goal_xy is None and lane_world_xy is None and self.case is not None:
            from robot.evaluation.restaurant_right_route import RIGHT_GUIDES

            trolley = self.case == "trolley" or (self.case == "delivery" and robot[1] >= 10.5)
            begin, goal_xy = RIGHT_GUIDES[3:5] if trolley else RIGHT_GUIDES[:2]
            step = np.asarray(goal_xy) - begin
            route_direction = step / np.linalg.norm(step)
        direction = validated_route_direction(route_direction)
        normal = np.array((-direction[1], direction[0]))
        goal, lane = (np.asarray(value, dtype=float) for value in (
            goal_xy, goal_xy if lane_world_xy is None else lane_world_xy,
        ))
        if any(value.shape != (2,) or not np.isfinite(value).all() for value in (goal, lane)):
            raise ValueError("Finite crossing segment positions required")
        points, ids = np.asarray(world_xy, dtype=float), np.asarray(tracking.point_track_ids)
        if (points.ndim != 2 or points.shape[1] != 2 or ids.shape != (len(points),)
            or not np.isfinite(points).all() or not np.issubdtype(ids.dtype, np.integer)):
            raise ValueError("Raw source point ownership must align exactly with current LiDAR XY")

        lane_changed = self.previous_lane is None or not np.allclose(
            np.r_[lane, direction], self.previous_lane, rtol=0., atol=1e-6,
        )
        source_gap = self.previous_source is None or source - self.previous_source > _MAX_SOURCE_GAP_S + 1e-9
        previous_motion = {} if lane_changed or source_gap else self.motion
        previous_active = set() if lane_changed or source_gap else set(self.active)
        visual = set(current_visual_yield_ids)
        occupied_forward = (points - robot) @ direction
        lateral = (points - lane) @ normal
        forward_end = min(_MAX_FORWARD_RANGE_M, float((goal - robot) @ direction) + _SEGMENT_END_TOLERANCE_M)
        forward_window = (occupied_forward > 0.) & (occupied_forward <= forward_end)

        new_motion, rows, active = {}, [], []
        seen = set()
        for track in tracking.tracked_obstacles:
            key = int(track.track_id)
            if key in seen:
                raise ValueError("Duplicate crossing track identity")
            seen.add(key)
            raw_mask = ids == key
            count = int(raw_mask.sum())
            if count < _MIN_CONTINUING_RAW_POINTS:
                continue
            center = np.asarray(track.centroid_world_xy, dtype=float)
            velocity = np.asarray(track.velocity_world_xy, dtype=float)
            if any(value.shape != (2,) or not np.isfinite(value).all() for value in (center, velocity)):
                raise ValueError("Finite tracked crossing motion required")
            speed = float(np.linalg.norm(velocity))
            transverse = float(velocity @ normal)
            bounded_motion = bool(
                track.matched_previous_track and track.observation_count >= 3
                and count >= _MIN_INITIAL_RAW_POINTS and _MIN_MOVING_SPEED_MPS <= speed <= _MAX_MOVING_SPEED_MPS
                and abs(transverse) >= _MIN_TRANSVERSE_FRACTION * speed
            )
            consecutive = 0
            if bounded_motion:
                consecutive = 1
                old = previous_motion.get(key)
                if old is not None:
                    dt = source - self.previous_source
                    if (np.linalg.norm(velocity - old[1]) <= _MAX_VELOCITY_CHANGE_MPS
                        and np.linalg.norm(center - old[0] - .5 * dt * (velocity + old[1])) <= _MAX_MOTION_RESIDUAL_M):
                        consecutive = old[2] + 1
                new_motion[key] = (center, velocity, consecutive)

            zone = raw_mask & forward_window
            if not np.any(zone):
                continue
            side = lateral[zone]
            current_support = int(np.count_nonzero(np.abs(side) <= _LANE_HALF_WIDTH_M))
            entry_seconds = None
            if bounded_motion:
                # Compute each actual return's first transverse entry. A kth
                # entry suppresses single-ray edge outliers and a moving track
                # already leaving cannot restart from its away-side returns.
                toward = -np.sign(transverse) * side
                approaching = toward[toward > _LANE_HALF_WIDTH_M]
                predicted = (approaching - _LANE_HALF_WIDTH_M) / abs(transverse)
                # Require several rays and a small fraction of this source's
                # approaching surface, not the minimum of a dense scan.
                required = max(_MIN_LANE_SUPPORT_POINTS, math.ceil(.01 * len(approaching)))
                if len(predicted) >= required:
                    order = required - 1
                    entry_seconds = float(np.partition(predicted, order)[order])
            forecast = bool(entry_seconds is not None and entry_seconds <= _FORECAST_SECONDS)
            straddles = current_support >= _MIN_LANE_SUPPORT_POINTS
            previous = key in previous_active
            motion_established = bounded_motion and (consecutive >= _MIN_CONSECUTIVE_MOTION_SOURCES or key in visual)
            if previous and straddles:
                decision = "continue_current_lane_support"
            elif previous and forecast:
                decision = "continue_current_approaching_support"
            elif motion_established and (forecast or straddles):
                decision = "initiate_current_transverse_motion"
            else:
                decision = "no_current_crossing_request"
            if decision != "no_current_crossing_request":
                active.append(key)
            if bounded_motion or previous or straddles:
                rows.append(dict(track_id=key, decision=decision, current_raw_points=count,
                                 current_forward_zone_points=int(zone.sum()), current_lane_points=current_support,
                                 forward_bounds_m=[float(occupied_forward[zone].min()), float(occupied_forward[zone].max())],
                                 lateral_bounds_m=[float(side.min()), float(side.max())],
                                 world_velocity_xy=velocity.tolist(), consistent_motion_sources=consecutive,
                                 predicted_entry_seconds=entry_seconds,
                                 current_visual_yield=key in visual))

        candidates = tuple(sorted(active))
        continuing = tuple(key for key in candidates if key in previous_active)
        selected = candidates if len(candidates) <= 1 else continuing if len(continuing) == 1 else ()
        for row in rows:
            row["selected"] = row["track_id"] in selected
        self.previous_source, self.previous_lane = source, np.r_[lane, direction]
        self.motion, self.active = new_motion, selected
        return dict(source_timestamp_seconds=source, yield_track_ids=self.active, rows=rows,
                    candidate_track_ids=candidates, ambiguous_current_candidates=len(candidates) > 1,
                    dropped_previous_track_ids=tuple(sorted(previous_active - set(selected))),
                    lane_world_xy=lane.tolist(), route_direction_world_xy=direction.tolist(),
                    lane_half_width_m=_LANE_HALF_WIDTH_M, forward_zone_m=[0., max(0., forward_end)],
                    forecast_seconds=_FORECAST_SECONDS,
                    scope="Current raw LiDAR only; a request still requires exact and analytical zero checks.")

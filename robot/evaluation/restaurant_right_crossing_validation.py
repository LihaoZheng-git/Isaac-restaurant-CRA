"""Independent, same-epoch scoring for one right-side crossing actor."""

from collections import Counter
import math

import numpy as np

from robot.evaluation.crate_clearance_observation import world_pose_from_local
from robot.evaluation.simple_object_test import disk_footprint_clearance


_FLOOR = "/World/Environment/BuildingShell/Default_Floor_758"
_STOP_LINEAR_MPS = 0.005
_STOP_DRIFT_M = 0.02
_MINIMUM_RESUMED_PROGRESS_M = 0.05
_SOURCE_TOLERANCE_SECONDS = 1e-6


def right_crossing_clearance(robot_world_pose, actor_record, *, footprint, actor_radius_m):
    """Measure padded Carter clearance to the body and optional trolley bounds."""
    body = disk_footprint_clearance(
        robot_world_pose, center_xy=actor_record["actor_xyz"][:2],
        radius_m=actor_radius_m, footprint=footprint,
    )
    bounds = actor_record.get("trolley_bounds_world_xy")
    if bounds is None:
        return body
    from robot.scene.controlled_semantic_obstacle import (
        PlanarFootprint, WorldBounds3D, oriented_footprint_clearance_to_bounds,
    )

    lower, upper = np.asarray(bounds, dtype=float)
    margin = footprint.safety_margin
    padded = PlanarFootprint(
        footprint.front_extent + margin, footprint.rear_extent + margin,
        footprint.left_extent + margin, footprint.right_extent + margin,
    )
    trolley = oriented_footprint_clearance_to_bounds(
        robot_world_pose, footprint=padded,
        obstacle_bounds=WorldBounds3D((*lower, 0.), (*upper, 0.)),
    )
    return min(body, trolley)


def _phase_contact_clear(report, phase):
    row = report.get("phases", {}).get(phase, {})
    targets = [value for key, value in row.items() if key.endswith("_contact_reported")]
    return bool(
        report.get("installed") and report.get("error") is None
        and set(row.get("robot_environment_headers", {})) == {_FLOOR}
        and targets and all(value is False for value in targets)
    )


def _crossing_window(actor, rows, start, direction, footprint, radius):
    """Project the union of the actual actor and cart extents onto the route."""
    transverse = np.array([-direction[1], direction[0]])
    side = (actor - start) @ transverse
    forward = (actor - start) @ direction
    low, high = side - radius, side + radius
    forward_low, forward_high = forward - radius, forward + radius
    for i, row in enumerate(rows):
        bounds = row.get("trolley_bounds_world_xy")
        if bounds is not None:
            lower, upper = np.asarray(bounds, dtype=float)
            corners = np.array([lower, upper, (lower[0], upper[1]), (upper[0], lower[1])]) - start
            cart_side, cart_forward = corners @ transverse, corners @ direction
            low[i], high[i] = min(low[i], cart_side.min()), max(high[i], cart_side.max())
            forward_low[i] = min(forward_low[i], cart_forward.min())
            forward_high[i] = max(forward_high[i], cart_forward.max())
    left = footprint.left_extent + footprint.safety_margin
    right = footprint.right_extent + footprint.safety_margin
    entry_side = 1 if low[0] > left else -1 if high[0] < -right else 0
    if not entry_side:
        return None, None, forward_low, forward_high
    lane = (low <= left) & (high >= -right)
    moving = np.array([row["phase"] == "move" for row in rows])
    entries = np.flatnonzero(lane & moving)
    if not len(entries):
        return None, None, forward_low, forward_high
    entry = int(entries[0])
    opposite = high < -right if entry_side == 1 else low > left
    clears = np.flatnonzero(opposite & (np.arange(len(rows)) > entry))
    clear = int(clears[0]) if len(clears) else None
    return entry, clear, forward_low, forward_high


def _observed_policy(live_fields, actor_times, actor_xy, begin, end):
    sources = {}
    for field in live_fields:
        source = field["sensor_timestamp"]
        if not begin <= source <= end:
            continue
        index = min(int(np.searchsorted(actor_times, source - _SOURCE_TOLERANCE_SECONDS)),
                    len(actor_times) - 1)
        if abs(actor_times[index] - source) > _SOURCE_TOLERANCE_SECONDS:
            continue
        diagnostics = field["diagnostics"]
        selected = set(diagnostics.get("crossing_yield_track_ids") or ())
        for item in field.get("objects", ()):
            key = item["track_id"]
            evidence = item["diagnostics"]
            if key not in selected or evidence.get("policy_return_count", 0) <= 0:
                continue
            if math.dist(evidence["centroid_world_xy"], actor_xy[index]) > 0.45:
                continue
            owner = next((row for row in diagnostics.get("applied_field_policies", ())
                          if row["track_or_group_id"] == key), {})
            cue = owner.get("encounter_guidance_ablation", {})
            if (owner.get("lambda") == 0. and cue.get("lambda_suppressed") is True
                and cue.get("source_timestamp_seconds") == source):
                sources[(field["sensor_frame_id"], source)] = key
    return sources


def _lidar_policy(live_fields, times, rows, begin, end, actor_radius):
    sources = {}
    for field in live_fields:
        source = field["sensor_timestamp"]
        if not begin <= source <= end:
            continue
        index = min(int(np.searchsorted(times, source - _SOURCE_TOLERANCE_SECONDS)), len(times) - 1)
        if abs(times[index] - source) > _SOURCE_TOLERANCE_SECONDS:
            continue
        diagnostics = field["diagnostics"]
        ids = diagnostics.get("admitted_lidar_yield_track_ids", ())
        selection = diagnostics.get("current_lidar_crossing_yield") or {}
        admission = diagnostics.get("checked_wait_current_pose_admission") or {}
        if len(ids) != 1 or not admission.get("accepted") or selection.get("source_timestamp_seconds") != source:
            continue
        row = rows[index]
        for item in field.get("objects", ()):
            if item["track_id"] != ids[0]:
                continue
            center = np.asarray(item["diagnostics"]["centroid_world_xy"])
            body_match = np.linalg.norm(center - row["actor_xyz"][:2]) <= actor_radius + .30
            bounds = row.get("trolley_bounds_world_xy")
            if bounds is not None:
                low, high = np.asarray(bounds)
                body_match |= bool(np.all(center >= low - .30) and np.all(center <= high + .30))
            if body_match:
                sources[(field["sensor_frame_id"], source)] = ids[0]
    return sources


def _yield_requests(events, sources, lidar_sources, navigation_start):
    requests = {}
    for event in events:
        audit = event.get("crossing_maneuver_ablation") or {}
        source = (audit.get("sensor_frame_id"), audit.get("sensor_timestamp_seconds"))
        ledger = lidar_sources if audit.get("selection_basis") == "current_lidar_crossing" else sources
        if (event.get("accepted") is True and audit.get("action") == "checked_yield"
            and audit.get("wait_exact_checked") and audit.get("wait_analytical_checked")
            and source in ledger and ledger[source] in audit.get("selected_track_ids", ())):
            request = navigation_start + event["time"] - event["source_age_seconds"]
            requests[request] = source
    return requests


def _hold_near_entry(commands, times, robot, entry, clear):
    velocity = np.abs(np.asarray([row["v"] for row in commands]))
    commanded_stop = velocity <= _STOP_LINEAR_MPS
    # A one- or two-tick plan change can occur during an otherwise measured hold.
    first = max(0, int(np.searchsorted(times, times[entry] - 2.0)))
    start = entry
    while start > first and np.linalg.norm(robot[start - 1, :2] - robot[entry, :2]) <= _STOP_DRIFT_M:
        start -= 1
    while start < entry and not commanded_stop[start]:
        start += 1
    end = clear if clear is not None else entry
    stop_window = slice(start, end)
    drift = float(np.max(np.linalg.norm(robot[start:end + 1, :2] - robot[start, :2], axis=1)))
    stopped_count = int(np.count_nonzero(commanded_stop[stop_window]))
    stopped_fraction = stopped_count / max(1, end - start)
    entry_stop = any(commanded_stop[max(start, entry - 2):entry]) and any(
        commanded_stop[entry:min(len(times), entry + 3)]
    )
    return start, drift, stopped_fraction, entry_stop


def _robot_front_on_route(pose, progress, direction, footprint):
    c, s = math.cos(pose[2]), math.sin(pose[2])
    forward = c * direction[0] + s * direction[1]
    left = -s * direction[0] + c * direction[1]
    margin = footprint.safety_margin
    return progress + max((footprint.front_extent + margin) * forward,
                          -(footprint.rear_extent + margin) * forward) + max(
        (footprint.left_extent + margin) * left, -(footprint.right_extent + margin) * left,
    )


def summarize_right_crossing(
    commands, trajectory, actor_records, live_fields, *, plan_events,
    anchor_world_xy_yaw, navigation_start_seconds, route_start_world_xy,
    route_goal_world_xy, footprint, actor_radius_m, minimum_clearance_m,
    contact_report, arrived, next_guide_index=None, additional_contact_reports=(),
):
    """Score one actor independently; call once per actor for combined trials.

    Command rows have navigation-relative ``time``, ``v`` and ``omega``.
    Trajectory rows hold local ``(x, y, yaw)`` at the same ticks. Actor rows
    contain absolute ``time``, ``phase``, ``actor_xyz`` and optional trolley
    XY bounds. Route endpoints define the straight crossing corridor. With a
    trolley, pass its contact report and the driver's additional report.
    """
    times = np.asarray([row["time"] for row in commands], dtype=float)
    actor_times = np.asarray([row["time"] for row in actor_records], dtype=float)
    epochs = times + navigation_start_seconds
    indices = np.searchsorted(actor_times, epochs - _SOURCE_TOLERANCE_SECONDS)
    paired = bool(
        len(times) >= 2 and len(times) == len(trajectory)
        and np.isfinite(times).all() and np.isfinite(actor_times).all()
        and np.all(np.diff(times) > 0.) and np.all(np.diff(actor_times) > 0.)
        and np.all(indices < len(actor_times))
        and np.all(np.abs(actor_times[indices] - epochs) <= _SOURCE_TOLERANCE_SECONDS)
    )
    if not paired:
        return dict(passed=False, controller_behavior_demonstrated=False,
                    checks={"same_epoch_robot_and_actor_poses": False})

    rows = [actor_records[index] for index in indices]
    robot = np.array([world_pose_from_local(pose, anchor_world_xy_yaw) for pose in trajectory])
    actor = np.asarray([row["actor_xyz"][:2] for row in rows], dtype=float)
    start, goal = (np.asarray(x, dtype=float) for x in (route_start_world_xy, route_goal_world_xy))
    if start.shape != (2,) or goal.shape != (2,):
        raise ValueError("finite distinct right-crossing route endpoints required")
    length = float(np.linalg.norm(goal - start))
    if not math.isfinite(length) or length <= 0.:
        raise ValueError("finite distinct right-crossing route endpoints required")
    direction = (goal - start) / length
    entry, clear, along_low, along_high = _crossing_window(actor, rows, start, direction, footprint, actor_radius_m)
    progress = (robot[:, :2] - start) @ direction
    clearance = min(right_crossing_clearance(pose, row, footprint=footprint, actor_radius_m=actor_radius_m)
                    for pose, row in zip(robot, rows, strict=True))
    trolley_expected = any(row.get("trolley_world_xy") is not None
                           or row.get("trolley_bounds_world_xy") is not None for row in rows)
    checks = dict(
        same_epoch_robot_and_actor_poses=True,
        crossed_the_configured_route=bool(entry is not None and clear is not None
            and np.any((along_low[entry:clear] <= length) & (along_high[entry:clear] >= 0.))),
        stopped_before_crossing=False,
        remained_stopped_through_crossing=False,
        resumed_after_crossing=False,
        reached_next_guide_after_resumption=False,
        complete_trolley_bounds=not trolley_expected or all(
            row.get("trolley_bounds_world_xy") is not None for row in rows
        ),
        padded_actor_cart_and_layout_clearance=clearance > .02 and minimum_clearance_m > .02,
        navigation_and_braking_floor_only_contacts=all(
            _phase_contact_clear(report, phase)
            for report in (contact_report, *additional_contact_reports)
            for phase in ("navigation", "braking")
        ),
        current_lidar_only=bool(live_fields) and all(
            row["diagnostics"].get("current_lidar_occupancy", {}).get("passed") is True
            for row in live_fields
        ),
    )
    details = dict()
    sources, lidar_sources, checked_ticks = {}, {}, 0
    if entry is not None:
        begin, drift, stopped_fraction, entry_stop = _hold_near_entry(commands, times, robot, entry, clear)
        entry_front = _robot_front_on_route(robot[entry], progress[entry], direction, footprint)
        checks["stopped_before_crossing"] = bool(
            begin < entry and entry_stop and entry_front < along_low[entry]
        )
        checks["remained_stopped_through_crossing"] = bool(
            clear is not None and drift <= _STOP_DRIFT_M and stopped_fraction >= .9
        )
        stop_end = clear if clear is not None else entry
        sources = _observed_policy(live_fields, epochs, actor, epochs[begin] - .35, epochs[stop_end])
        lidar_sources = _lidar_policy(live_fields, epochs, rows, epochs[begin] - .35,
                                     epochs[stop_end], actor_radius_m)
        requests = _yield_requests(plan_events, sources, lidar_sources, navigation_start_seconds)
        stop_reasons = Counter()
        for i in range(begin, stop_end):
            row = commands[i]
            if abs(row["v"]) > _STOP_LINEAR_MPS:
                continue
            age = row.get("optimizer_candidate_source_age")
            if (row.get("stop_reason") is None and abs(row["omega"]) <= 1e-10 and age is not None
                and row.get("horizon_validation_age") is not None
                and any(
                    abs(epochs[i] - age - request) <= _SOURCE_TOLERANCE_SECONDS
                    and row.get("optimizer_source_frame_id", source[0]) == source[0]
                    and row.get("optimizer_source_time_seconds", source[1]) == source[1]
                    for request, source in requests.items()
                )):
                checked_ticks += 1
                stop_reasons["checked_yield"] += 1
            else:
                stop_reasons[row.get("stop_reason") or "unattributed_zero"] += 1
        details.update(
            hold_start_navigation_seconds=float(times[begin]), actor_entered_navigation_seconds=float(times[entry]),
            actor_cleared_navigation_seconds=None if clear is None else float(times[clear]),
            robot_world_pose_at_entry=robot[entry].tolist(), actor_world_xy_at_entry=actor[entry].tolist(),
            minimum_forward_gap_at_entry_m=float(along_low[entry] - entry_front),
            measured_pre_entry_hold_seconds=float(times[entry] - times[begin]),
            measured_hold_through_clearance_seconds=None if clear is None else float(times[clear] - times[begin]),
            maximum_hold_translation_m=drift, commanded_stop_fraction=stopped_fraction,
            zero_command_reasons=dict(stop_reasons),
        )
        if clear is not None:
            resumed = next((i for i in range(clear, len(commands)) if commands[i]["v"] > .02), None)
            checks["resumed_after_crossing"] = bool(resumed is not None and np.any(
                progress[resumed:] >= progress[resumed] + _MINIMUM_RESUMED_PROGRESS_M
            ))
            if checks["resumed_after_crossing"]:
                checks["reached_next_guide_after_resumption"] = bool(arrived or (
                    next_guide_index is not None and any(
                        row.get("target_waypoint_index") is not None and row["target_waypoint_index"] >= next_guide_index
                        for row in commands[resumed:]
                    )
                ))
                details.update(resumed_navigation_seconds=float(times[resumed]),
                               robot_world_pose_at_resumption=robot[resumed].tolist(),
                               actor_world_xy_at_resumption=actor[resumed].tolist())

    return dict(
        passed=all(checks.values()), checks=checks,
        controller_behavior_demonstrated=bool((sources or lidar_sources) and checked_ticks),
        admitted_current_lidar_yield_fields=len(lidar_sources),
        per_person_lambda_zero_field_count=len(sources), checked_yield_command_count=checked_ticks,
        minimum_padded_actor_or_cart_clearance_m=float(clearance), **details,
        scope="Post-drive crossing/clearance and original optimizer-clock provenance. Actor and trolley truth are evaluation only.",
    )

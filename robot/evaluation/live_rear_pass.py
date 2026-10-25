"""Separate sensor-tracked rear-reference experiment, never the primary controller.

Only already qualified current crossing IDs enter here. World tracks and the
configured robot route are rotated into the fixed field frame; actor truth is
not an input. All original field/command freshness gates remain caller duties.
"""

import copy
from dataclasses import dataclass, fields, replace
import math
from types import SimpleNamespace

import numpy as np

from robot.evaluation.live_checked_wait import check_wait, packet_state_in_field
from robot.safety.checked_rear_pass_reference import (
    CheckedRearPassReferenceConfig, rear_pass_context, search_checked_rear_pass_reference,
)
from robot.safety.crossing_maneuver import apply_crossing_maneuver
from robot.safety.encounter_motion import validated_route_direction
from robot.safety.safe_sage_runtime import SafeSAGEControlRuntime
from robot.safety.safe_sage_unicycle_mpc import SafeSAGEUnicycleMPCResult


def world_to_grid_rotation(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, s], [-s, c]])


def live_track_context(field, track_id, *, use_policy_motion=True):
    """Reject absent/weak/stale geometry rather than promoting a retained label."""
    source = field.sensor_timestamp_seconds
    selection = field.motion_passing_reference_source_timestamp_seconds
    if (selection is None or not 0 <= source - selection <= 8. + 1e-9
        or field.tracking_result.timestamp_seconds != source
        or track_id not in field.motion_passing_reference_track_ids
        or not field.motion_passing_reference_sides):
        raise ValueError("rear reference needs original bounded current selection")
    tracks = [t for t in field.tracking_result.tracked_obstacles if t.track_id == track_id]
    if (len(tracks) != 1 or tracks[0].point_count < 20
        or not tracks[0].matched_previous_track or tracks[0].observation_count < 3):
        raise ValueError("rear reference needs unique established current positive motion")
    rotation = world_to_grid_rotation(field.anchor_yaw)
    track = tracks[0]
    position = rotation @ (np.asarray(track.centroid_world_xy) - field.anchor_position_world_xy)
    velocity = rotation @ np.asarray(track.velocity_world_xy)
    estimates = [m for m in getattr(field, 'rear_pass_policy_motion', ()) if m.track_id == track_id]
    if len(estimates) > 1:
        raise ValueError("ambiguous source-window rear policy motion")
    if estimates and use_policy_motion:
        estimates[0].validate_current_track(track, source)
        velocity = rotation @ np.asarray(estimates[0].velocity_world_xy)
    return position, velocity


def adapt_live_rear_guidance(normal, applied, rows, tracking, packet, *,
                            anchor, anchor_yaw, crossing_ids, config, policy_motion=(),
                            person_relative_memory=None):
    """Choose rear/checked-wait/release using current source pose and motion.

    Only the wait preference retains encounter lambda0. Rear/released/invalid
    contexts restore the original lambda1.25, never alter risk/flux/occupancy.
    The subsequent wait-field admission still checks its full stopped horizon.
    """
    changed, audit = dict(applied), copy.deepcopy(rows)
    rotation = world_to_grid_rotation(anchor_yaw)
    direction = validated_route_direction(packet.nominal_route_direction_world_xy)
    route = tuple(rotation @ direction)
    context_field = SimpleNamespace(
        anchor_position_world_xy=np.asarray(anchor), anchor_yaw=anchor_yaw,
        sensor_timestamp_seconds=packet.sensor_timestamp_seconds,
        tracking_result=tracking,
        rear_pass_policy_motion=policy_motion,
        motion_passing_reference_track_ids=tuple(crossing_ids),
        motion_passing_reference_sides=("selected",) if crossing_ids else (),
        motion_passing_reference_source_timestamp_seconds=next((
            r["encounter_guidance_ablation"]["original_selection_timestamp_seconds"]
            for r in rows if r["encounter_guidance_ablation"]["lambda_suppressed"]), None),
    )
    decisions = {}
    if config.person_relative is not None and person_relative_memory is None:
        raise ValueError("person-relative mode requires staged maneuver memory")
    memory_updated = False
    for key in crossing_ids:
        decision = dict(mode="none", reason="ambiguous_crossing", track_id=key)
        if len(crossing_ids) == 1:
            try:
                position, velocity = live_track_context(context_field, key)
                decision = rear_pass_context(
                    packet_state_in_field(packet, context_field),
                    rotation @ (np.asarray(packet.waypoint_world_xy) - anchor),
                    position, velocity, route, config,
                )
                person_region = (config.person_relative is not None
                    and (config.person_relative.minimum_current_rear_separation_m > 0
                         or config.person_relative.continue_frozen_until_deadline
                         and person_relative_memory.reference is not None)
                    and decision['reason'] == 'current_human_has_cleared_behavior_window')
                if config.person_relative is not None and (decision['mode'] != 'none' or person_region):
                    terminal = getattr(packet, 'nominal_route_terminal_goal_world_xy', None)
                    if terminal is None:
                        raise ValueError("person-relative mode needs explicit nominal terminal-route context")
                    terminal_grid = rotation@(np.asarray(terminal)-anchor)
                    route_delta = terminal_grid-rotation@(np.asarray(packet.waypoint_world_xy)-anchor)
                    if abs(route_delta[0]*route[1]-route_delta[1]*route[0]) > .05:
                        raise ValueError("person-relative reference currently requires a straight nominal route segment")
                    reference = person_relative_memory.update(track_id=key,
                        selection_timestamp=context_field.motion_passing_reference_source_timestamp_seconds,
                        source_timestamp=packet.sensor_timestamp_seconds,
                        initial=packet_state_in_field(packet, context_field),
                        goal=terminal_grid,
                        person_xy=position, velocity_xy=velocity, route_direction=route)
                    memory_updated = True
                    if reference is None:
                        decision.update(mode='wait', reason=person_relative_memory.reason)
                    else:
                        decision.update(reference.context(packet_state_in_field(packet, context_field)))
            except ValueError as error:
                decision = dict(mode="none", reason=str(error))
        decisions[key] = dict(decision, track_id=key,
            source_timestamp_seconds=packet.sensor_timestamp_seconds,
            original_selection_timestamp_seconds=context_field.motion_passing_reference_source_timestamp_seconds)
        decisions[key]['motion_policy_estimate'] = next(
            (m.diagnostic() for m in policy_motion if m.track_id == key), None)
    if person_relative_memory is not None and not memory_updated:
        person_relative_memory.update(track_id=None, selection_timestamp=None,
            source_timestamp=packet.sensor_timestamp_seconds, initial=None, goal=None,
            person_xy=None, velocity_xy=None, route_direction=None)
    wait_ids = tuple(k for k, d in decisions.items() if d["mode"] == "wait")
    rear_ids = tuple(k for k, d in decisions.items() if d["mode"] == "rear")
    for row in audit:
        cue = row["encounter_guidance_ablation"]
        owner = cue["parent_track_id"]
        if owner not in decisions:
            continue
        row["live_rear_pass_behavior"] = decisions[owner]
        if owner not in wait_ids:
            key = row["track_or_group_id"]
            changed[key] = normal[key]
            row["lambda"] = normal[key].tangential_strength
            cue.update(lambda_suppressed=False, basis="live_rear_reference_or_release_normal_guidance")
    return changed, audit, wait_ids, rear_ids, route


@dataclass(frozen=True)
class LiveRearPassMPCResult(SafeSAGEUnicycleMPCResult):
    rear_pass_reference: dict
    crossing_maneuver: dict


class LiveRearPassControlRuntime(SafeSAGEControlRuntime):
    """Current-frame full-horizon reference selection; normal MPC remains fallback."""

    requires_nominal_goal_context = True

    def __init__(self, *, mpc, backup_filter, config=None):
        super().__init__(mpc=mpc, backup_filter=backup_filter)
        self.rear_pass_config = config or CheckedRearPassReferenceConfig()
        if not isinstance(self.rear_pass_config, CheckedRearPassReferenceConfig):
            raise TypeError("explicit checked rear-reference configuration required")

    def plan(self, initial, nominal, field, *, nominal_goal_grid_xy,
             initial_command_guess=None, preferred_lateral_direction_grid_y=None):
        result = super().plan(initial, nominal, field,
            initial_command_guess=initial_command_guess,
            preferred_lateral_direction_grid_y=preferred_lateral_direction_grid_y)
        audit = dict(selected=False, reason="no_current_qualified_rear_context",
            source_timestamp_seconds=field.sensor_timestamp_seconds,
            sensor_frame_id=field.sensor_frame_id,
            original_selection_timestamp_seconds=field.motion_passing_reference_source_timestamp_seconds,
            scope="sensor-tracked complete checked reference selected for plan; execution independently revalidated")
        ids = field.rear_pass_reference_track_ids
        if len(ids) == 1:
            position, velocity = live_track_context(field, ids[0])
            _, raw_velocity = live_track_context(field, ids[0], use_policy_motion=False)
            person_reference = getattr(field, 'person_relative_rear_reference', None)
            if person_reference is not None:
                person_reference.validate_current(track_id=ids[0],
                    selection_timestamp=field.motion_passing_reference_source_timestamp_seconds,
                    source_timestamp=field.sensor_timestamp_seconds, person_xy=position,
                    velocity_xy=velocity, route_direction=field.rear_pass_route_direction_grid_xy)
            search = search_checked_rear_pass_reference(self.mpc, initial,
                nominal_goal_grid_xy, position, velocity,
                field.rear_pass_route_direction_grid_xy,
                dict(configuration_space=field.pipeline.configuration_space,
                     h_full=field.pipeline.h_full, guidance_fields=field.pipeline.guidance_fields,
                     temporal_safety_derivative=field.temporal_safety_derivative),
                lambda command: self.filter_command(command, initial, field),
                config=self.rear_pass_config, person_relative_reference=person_reference)
            audit.update(search.diagnostics, track_id=ids[0],
                         observed_centroid_grid_xy=position.tolist(),
                         observed_velocity_grid_xy=raw_velocity.tolist(),
                         policy_velocity_grid_xy=velocity.tolist(),
                         motion_policy_estimate=next((m.diagnostic() for m in getattr(field, 'rear_pass_policy_motion', ()) if m.track_id == ids[0]), None),
                         route_direction_grid_xy=field.rear_pass_route_direction_grid_xy)
            if search.trajectory is not None:
                value = search.trajectory
                def frozen(array):
                    array = np.array(array, copy=True)
                    array.setflags(write=False)
                    return array
                result = replace(result, command_sequence=frozen(value.commands),
                    predicted_chassis_states=frozen(value.chassis_states),
                    predicted_safety_states=frozen(value.safety_states),
                    predicted_safety_values=frozen(value.safety_values),
                    dtcbf_margins=frozen(value.margins),
                    objective_value=self.mpc._objective_value(value.commands, self.mpc._nominal_sequence(nominal)),
                    success=True, fallback_active=False, intervention_active=True,
                    status="live_experimental_checked_rear_reference")
        # Only the separately admitted lambda0 wait field may request yielding.
        # Recheck forecast analytical acceptance at this actual planning pose.
        if field.crossing_yield_track_ids and check_wait(self, initial, field)["accepted"]:
            result = apply_crossing_maneuver(self.mpc, result, initial, nominal, field,
                "yield-then-go", lambda: self.filter_command((0., 0.), initial, field))
        return LiveRearPassMPCResult(
            **{f.name: getattr(result, f.name) for f in fields(SafeSAGEUnicycleMPCResult)},
            rear_pass_reference=audit, crossing_maneuver=getattr(result, "crossing_maneuver", {}))


def summarize_rear_execution(plans, commands, truth, *, navigation_start_time):
    """Join actual translating commands by optimizer REQUEST clock, never field time.

    Independent actor phase is used only here, after navigation. A selected
    proposal, safety stop or late post-crossing curve cannot pass this gate.
    """
    times = np.array([r["time"] for r in truth], dtype=float)
    positions = np.array([r["actor_xyz"][:2] for r in truth], dtype=float)
    if len(times) < 3 or np.any(np.diff(times) <= 0) or not np.isfinite(positions).all():
        raise ValueError("complete ordered independent actor trace required")
    walking = (np.array([r.get("phase") for r in truth]) == "move") & (
        np.linalg.norm(np.gradient(positions, times, axis=0), axis=1) >= .3)
    requests, sources = [], set()
    for event in plans:
        audit = event.get("rear_pass_reference_ablation") or {}
        source, selection = audit.get("source_timestamp_seconds"), audit.get("original_selection_timestamp_seconds")
        if (event.get("accepted") is not True or audit.get("selected") is not True
            or source is None or selection is None or not 0 <= source-selection <= 8.+1e-9):
            continue
        request = navigation_start_time + event["time"] - event["source_age_seconds"]
        requests.append(request)
        i = min(int(np.searchsorted(times, source)), len(times)-1)
        if times[0] <= source <= times[-1] and walking[i]:
            sources.add((audit.get("sensor_frame_id"), source))
    joined = []
    for i, command in enumerate(commands[:-1]):
        now = navigation_start_time + command["time"]
        j = min(int(np.searchsorted(times, now)), len(times)-1)
        age = command.get("optimizer_candidate_source_age")
        request = None if age is None else now-age
        if (times[0] <= now <= times[-1] and walking[j]
            and command.get("stop_reason") is None and abs(command["v"]) > .02
            and command.get("horizon_validation_age") is not None
            and request is not None and any(abs(request-r) < 1e-6 for r in requests)):
            joined.append(dict(time=now, optimizer_request_timestamp_seconds=request,
                duration_seconds=commands[i+1]["time"]-command["time"]))
    checks = dict(three_accepted_rear_sources_during_walking=len(sources) >= 3,
                  verified_rear_plan_translation_during_walking=bool(joined))
    return dict(passed=all(checks.values()), checks=checks,
        accepted_rear_sources_during_walking=len(sources), verified_command_rows=len(joined),
        verified_translation_seconds=sum(r["duration_seconds"] for r in joined),
        joined_commands=joined,
        scope="post-hoc optimizer-request-clock execution join; actor truth evaluation only; arrival/arc/clearance/stopping separate")

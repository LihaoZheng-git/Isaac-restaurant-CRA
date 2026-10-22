"""Explicit crossing preferences; every yielded horizon remains exact-checked."""

from dataclasses import dataclass, fields, replace

import numpy as np

from robot.safety.laplace_guidance_tracking_mpc import GuidanceTrackingMPCResult
from robot.safety.safe_sage_unicycle_mpc import SafeSAGEUnicycleMPCResult


@dataclass(frozen=True)
class CrossingMPCResult(SafeSAGEUnicycleMPCResult):
    crossing_maneuver: dict


@dataclass(frozen=True)
class CrossingGuidanceMPCResult(GuidanceTrackingMPCResult):
    crossing_maneuver: dict


def apply_crossing_maneuver(mpc, result, initial, nominal, field, strategy, check_stop):
    """Prefer checked yielding over an unavailable/opposed rear-pass reference.

    The original current positive selection and its expiry govern eligibility.
    A zero command is a candidate, never an unconditional emergency response.
    Failed dynamic/geometry/analytical stop checks retain the original planner.
    """
    if strategy not in {"rear-pass-or-yield", "yield-then-go"}:
        raise ValueError("unknown explicit crossing maneuver")
    ids = field.motion_passing_reference_track_ids
    selected = bool(ids and field.motion_passing_reference_sides)
    detail = getattr(result, "guidance_tracking", None)
    reason = "no_current_crossing_selection"
    want_wait = selected and strategy == "yield-then-go"
    if want_wait:
        reason = "yield_during_current_crossing_selection"
    elif selected:
        if detail is None or not detail.active:
            want_wait, reason = True, "no_checked_rear_passing_reference"
        elif detail.reference_commands:
            reference_turn = detail.reference_commands[0][1]
            if reference_turn * result.angular_velocity < -1e-6:
                want_wait, reason = True, "opposed_first_turn_from_tracking_solver"
            else:
                reason = "checked_rear_passing_reference"
    audit = dict(strategy=strategy, action="planner", reason=reason,
        sensor_timestamp_seconds=field.sensor_timestamp_seconds,
        sensor_frame_id=field.sensor_frame_id,
        selected_track_ids=ids,
        original_selection_timestamp_seconds=field.motion_passing_reference_source_timestamp_seconds,
        original_planner_status=result.status, original_command=result.first_command,
        checked_reference_selected_for_plan=bool(detail is not None and detail.active),
        wait_exact_checked=False, wait_analytical_checked=False,
        scope="controller preference only; current positive identity and original expiry; no physical evidence, occupancy or clock renewal")
    if want_wait:
        commands = np.zeros((mpc.config.horizon_steps, 2))
        value = mpc._evaluate_trajectory(np.asarray(initial), commands, require_feasible=True,
            configuration_space=field.pipeline.configuration_space, h_full=field.pipeline.h_full,
            guidance_fields=field.pipeline.guidance_fields,
            temporal_safety_derivative=field.temporal_safety_derivative)
        audit["wait_exact_checked"] = value is not None and mpc._trajectory_is_safe(value)
        if audit["wait_exact_checked"]:
            stopped = check_stop()
            audit["wait_analytical_checked"] = (
                not stopped.emergency_stop_active
                and abs(stopped.safe_linear_velocity) <= 1e-10
                and abs(stopped.safe_angular_velocity) <= 1e-10)
        if audit["wait_exact_checked"] and audit["wait_analytical_checked"]:
            def frozen(a):
                a = np.array(a, copy=True)
                a.setflags(write=False)
                return a
            sequence = mpc._nominal_sequence(nominal)
            result = replace(result, command_sequence=frozen(value.commands),
                predicted_chassis_states=frozen(value.chassis_states),
                predicted_safety_states=frozen(value.safety_states),
                predicted_safety_values=frozen(value.safety_values),
                dtcbf_margins=frozen(value.margins),
                objective_value=mpc._objective_value(value.commands, sequence),
                # This is an intentional fully verified plan. fallback_active
                # denotes an unusable emergency result in the live caller.
                success=True, intervention_active=True, fallback_active=False,
                status="crossing_strategy_exact_checked_yield")
            audit["action"] = "checked_yield"
            audit["checked_reference_selected_for_plan"] = False
            if isinstance(result, GuidanceTrackingMPCResult):
                result = replace(result, guidance_tracking=replace(result.guidance_tracking,
                    active=False, reason="crossing_strategy_checked_yield; reference retained for audit only"))
        else:
            audit["action"] = "original_safety_response_wait_unavailable"
    cls = CrossingGuidanceMPCResult if isinstance(result, GuidanceTrackingMPCResult) else CrossingMPCResult
    return cls(**{f.name: getattr(result, f.name) for f in fields(result)}, crossing_maneuver=audit)

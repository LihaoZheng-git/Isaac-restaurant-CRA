"""Assess a proposed waiting field before committing a separate control ablation.

This does not publish fields, change physical risk, reset temporal derivatives,
or produce an emergency command. A rejected candidate needs a freshly built
alternative against the same previous accepted source pair.
"""

import math

import numpy as np


def assess_checked_wait_transition(mpc, initial, field, check_stop):
    """Require full exact/dense/future-analytical stopped-horizon acceptance."""
    lidar_ids = tuple(getattr(field, "current_lidar_yield_track_ids", ()))
    ids = lidar_ids or tuple(field.crossing_yield_track_ids or ())
    source = field.sensor_timestamp_seconds
    selected = source if lidar_ids else field.motion_passing_reference_source_timestamp_seconds
    audit = dict(
        accepted=False,
        reason="no_unique_current_crossing",
        source_timestamp_seconds=source,
        source_frame_id=field.sensor_frame_id,
        original_selection_timestamp_seconds=selected,
        track_ids=ids,
        exact_dense_accepted=False,
        future_analytical_accepted=False,
        first_analytical_accepted=False,
        scope="candidate field policy only; rejected fields are not committed; original physical evidence and source clocks unchanged",
    )
    if len(ids) != 1:
        return audit
    if (
        selected is None
        or not math.isfinite(source)
        or not math.isfinite(selected)
        or not 0 <= source - selected <= 8.0 + 1e-9
        or not lidar_ids and ids[0] not in field.motion_passing_reference_track_ids
        or field.tracking_result.timestamp_seconds != source
    ):
        raise ValueError("current crossing and original source/selection pair required")
    tracks = [
        t for t in field.tracking_result.tracked_obstacles if t.track_id == ids[0]
    ]
    if len(tracks) != 1 or (
        tracks[0].point_count < 20
        or not tracks[0].matched_previous_track
        or tracks[0].observation_count < 3
    ):
        audit["reason"] = "insufficient_current_track_support"
        return audit
    commands = np.zeros((mpc.config.horizon_steps, 2))
    value = mpc._evaluate_trajectory(
        np.asarray(initial, dtype=float),
        commands,
        require_feasible=True,
        configuration_space=field.pipeline.configuration_space,
        h_full=field.pipeline.h_full,
        guidance_fields=field.pipeline.guidance_fields,
        temporal_safety_derivative=field.temporal_safety_derivative,
    )
    if value is None or not mpc._trajectory_is_safe(value):
        audit["reason"] = "stopped_horizon_failed_exact_or_dense_checks"
        return audit
    audit["exact_dense_accepted"] = True
    audit["minimum_dtcbf_margin"] = float(np.min(value.margins))
    # Zero has zero mapped left-hand side at every forecast interval. Keep the
    # original time-adjusted values/derivatives, including policy transitions.
    rhs = (
        -mpc.paper_filter.config.gamma * np.maximum(0.0, value.safety_values[:-1])
        - value.samples.scaled_temporal_derivatives[:-1]
    )
    audit["minimum_future_analytical_margin"] = float(np.min(-rhs))
    if (
        not np.isfinite(rhs).all()
        or np.max(rhs) > mpc.paper_filter.config.numerical_tolerance
    ):
        audit["reason"] = "stopped_horizon_failed_future_analytical_checks"
        return audit
    audit["future_analytical_accepted"] = True
    stopped = check_stop()
    audit["first_analytical_accepted"] = bool(
        not stopped.emergency_stop_active
        and abs(stopped.safe_linear_velocity) <= 1e-10
        and abs(stopped.safe_angular_velocity) <= 1e-10
    )
    audit["accepted"] = audit["first_analytical_accepted"]
    audit["reason"] = (
        "complete_stopped_horizon_accepted"
        if audit["accepted"]
        else "stopped_first_command_modified_or_infeasible"
    )
    return audit

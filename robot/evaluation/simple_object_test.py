"""Bounded observed-object CRA/controller checks and optional oracle controls.

The optional controlled profile is an experimental input, not a perception result.
In particular, a round shape and absent detections do not certify low contact
or thermal risk. The unchanged adapter/assessor retains unknown sensor inputs.
"""

from dataclasses import dataclass, replace
import math
from typing import Callable

import numpy as np

from robot.control.unicycle_model import lookahead_state
from robot.navigation.waypoint_controller import (
    RobotPose2D,
    Waypoint,
    WaypointController,
    WaypointControllerConfig,
)
from robot.perception.physical_attributes import (
    EstimateStatus,
    MassAttributes,
    PhysicalAttributes,
    ScalarEstimate,
)
from robot.risk.physical_factor_adapter import (
    PhysicalFactorAdapterConfig,
    PhysicalFactorAdapterResult,
    adapt_physical_factors,
)
from robot.safety.safe_sage_configuration_space import RectangularRobotFootprint
from robot.safety.safe_sage_field_pipeline import SafeSAGEFieldPipelineResult
from robot.safety.safe_sage_unicycle_cbf_filter import SafeSAGEUnicycleCBFFilter
from robot.safety.safe_sage_unicycle_mpc import SafeSAGEUnicycleMPC
from robot.safety.semantic_obstacle import SemanticObstacle


@dataclass(frozen=True)
class ControlledObjectInputs:
    """Illustrative factor inputs; never inferred from the sphere class name."""

    mass_kg: float = 0.25
    fall_height_m: float = 0.0
    relative_speed_mps: float = 0.20
    contact_proxy: float = 0.02
    thermal_cue: float = 0.01
    interaction: float = 1.0

    def __post_init__(self):
        for name in ("mass_kg", "fall_height_m", "relative_speed_mps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be nonnegative and finite")
        for name in ("contact_proxy", "thermal_cue", "interaction"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(f"{name} must lie in [0, 1]")


def controlled_object_adapter(
    inputs: ControlledObjectInputs,
    *,
    track_id: int,
    timestamp_seconds: float,
) -> PhysicalFactorAdapterResult:
    """Exercise the real four-factor adapter under known experimental inputs."""

    def scalar(value, unit="1"):
        return ScalarEstimate(
            value,
            value,
            value,
            1.0,
            unit,
            "simple_object_controlled_oracle",
            timestamp_seconds,
            EstimateStatus.ORACLE,
            (
                "illustrative controlled input; not sensor estimation or calibrated safety",
            ),
        )

    base = PhysicalAttributes.unknown(
        track_id=track_id, timestamp_seconds=timestamp_seconds
    )
    attributes = replace(
        base,
        mass=MassAttributes(scalar(inputs.mass_kg, "kg")),
        geometry=replace(
            base.geometry, fall_height_m=scalar(inputs.fall_height_m, "m")
        ),
        motion=replace(
            base.motion,
            relative_speed_mps=scalar(inputs.relative_speed_mps, "m/s"),
            route_crossing_likelihood=scalar(inputs.interaction),
        ),
        thermal=replace(
            base.thermal, heat_source_likelihood=scalar(inputs.thermal_cue)
        ),
    )
    return adapt_physical_factors(
        attributes,
        timestamp_seconds=timestamp_seconds,
        potential_interaction_likelihood=scalar(inputs.interaction),
        thermal_interaction_likelihood=scalar(inputs.interaction),
        visual_contact_proxy=scalar(inputs.contact_proxy),
        contact_interaction_likelihood=scalar(inputs.interaction),
        config=PhysicalFactorAdapterConfig(allow_visual_contact_proxy=True),
    )


def build_assessed_object_field(
    points_xy, *, adapted, semantic, class_policy, guidance_policy,
    grid_config, field_config, unassessed_points_xy=None,
):
    """Send the assessment's unchanged policy into per-track field parameters.

    Unknown evidence is consumable by the conservative controller policy; it
    does not make four-factor perception complete. This function never chooses
    a replacement phi, class prior, or controlled experimental factor value.
    """
    from robot.safety.cra_safe_sage_policy import CRASafeSAGEPolicy
    from robot.safety.local_occupancy_grid import build_local_occupancy_grid
    from robot.safety.safe_sage_field_pipeline import build_safe_sage_field_pipeline

    if adapted.track_id != semantic.track_id:
        raise ValueError("assessment and semantic track IDs must match")
    policy = CRASafeSAGEPolicy().evaluate(adapted.profile)
    points_xy = np.asarray(points_xy)
    point_ids = np.full(len(points_xy), semantic.track_id)
    semantics = (semantic,)
    safety = {semantic.track_id: policy.guidance_parameters.safety_parameters}
    guidance = {semantic.track_id: policy.guidance_parameters}
    if unassessed_points_xy is not None and len(unassessed_points_xy):
        # A separate occupancy group, not an inferred object identity. Its
        # factors remain unavailable and use the unchanged CRA fallback.
        unknown_id = semantic.track_id + 1_000_000
        unknown = SemanticObstacle(
            replace(semantic.tracked_obstacle, track_id=unknown_id,
                    point_count=len(unassessed_points_xy)),
            "unknown", 0.0, "unassessed_lidar_occupancy_group",
        )
        unknown_policy = CRASafeSAGEPolicy().evaluate(adapt_physical_factors(
            PhysicalAttributes.unknown(track_id=unknown_id,
                timestamp_seconds=adapted.timestamp_seconds),
            timestamp_seconds=adapted.timestamp_seconds,
        ).profile)
        points_xy = np.concatenate((points_xy, np.asarray(unassessed_points_xy)))
        point_ids = np.concatenate((point_ids, np.full(len(unassessed_points_xy), unknown_id)))
        semantics += (unknown,)
        safety[unknown_id] = unknown_policy.guidance_parameters.safety_parameters
        guidance[unknown_id] = unknown_policy.guidance_parameters
    grid = build_local_occupancy_grid(
        points_xy,
        point_track_ids=point_ids,
        semantic_obstacles=semantics, class_safety_policy=class_policy,
        config=grid_config,
        track_safety_parameters=safety,
    )
    field = build_safe_sage_field_pipeline(
        grid, guidance_policy=guidance_policy, config=field_config,
        semantic_obstacles=semantics,
        track_guidance_parameters=guidance,
    )
    return policy, field


def disk_footprint_clearance(state, *, center_xy, radius_m, footprint):
    """Independent disk-to-rectangle clearance, including the policy margin.

    This oracle monitor is not passed to MPC/CBF or waypoint generation.
    Negative values indicate overlapping shapes.
    """

    state = np.asarray(state, dtype=float)
    center = np.asarray(center_xy, dtype=float)
    if (
        state.shape != (3,)
        or center.shape != (2,)
        or not np.isfinite(state).all()
        or not np.isfinite(center).all()
    ):
        raise ValueError("state/center must be finite 3D/2D vectors")
    if not math.isfinite(radius_m) or radius_m <= 0:
        raise ValueError("radius must be positive and finite")
    if not isinstance(footprint, RectangularRobotFootprint):
        raise TypeError("footprint must be RectangularRobotFootprint")
    delta = center - state[:2]
    cosine, sine = math.cos(state[2]), math.sin(state[2])
    local = np.array(
        [cosine * delta[0] + sine * delta[1], -sine * delta[0] + cosine * delta[1]]
    )
    margin = footprint.safety_margin
    nearest = np.clip(
        local,
        [-footprint.rear_extent - margin, -footprint.right_extent - margin],
        [footprint.front_extent + margin, footprint.left_extent + margin],
    )
    return float(np.linalg.norm(local - nearest) - radius_m)


def run_bounded_pass_by(
    *,
    field: SafeSAGEFieldPipelineResult,
    mpc: SafeSAGEUnicycleMPC,
    backup: SafeSAGEUnicycleCBFFilter,
    read_state: Callable,
    advance: Callable,
    stop: Callable,
    clearance: Callable,
    goal_xy=(2.5, 0.0),
    physics_dt=1 / 60,
    maximum_seconds=30.0,
    clearance_abort_m=0.02,
    progress: Callable | None = None,
) -> dict:
    """Run the existing waypoint/MPC/CBF classes through supplied physics.

    A frozen stationary field and zero temporal derivative are explicit test
    assumptions. CBF is rechecked before every physics-step wheel command.
    ``advance`` may execute Isaac wheels or an explicitly labeled numerical
    unicycle model. The caller reports which was used.
    """

    if (
        physics_dt <= 0
        or not math.isfinite(physics_dt)
        or maximum_seconds <= 0
        or not math.isfinite(maximum_seconds)
    ):
        raise ValueError("time steps and duration must be positive and finite")
    if clearance_abort_m < 0 or not math.isfinite(clearance_abort_m):
        raise ValueError("clearance guard must be nonnegative and finite")
    steps_per_cycle = round(mpc.config.time_step / physics_dt)
    if steps_per_cycle < 1 or not math.isclose(
        steps_per_cycle * physics_dt, mpc.config.time_step
    ):
        raise ValueError(
            "MPC interval must contain an integral number of physics steps"
        )
    waypoint = WaypointController(
        [Waypoint(*goal_xy)],
        config=WaypointControllerConfig(
            maximum_linear_velocity=0.20,
            maximum_angular_velocity=0.60,
            linear_gain=0.80,
            angular_gain=1.50,
            position_tolerance=0.10,
            minimum_alignment_scale=0.25,
            rotate_in_place_threshold_radians=math.radians(100),
        ),
    )
    trajectory, commands, nominal_commands, solve_times = [], [], [], []
    minimum_clearance = math.inf
    minimum_mpc_margin = math.inf
    minimum_cbf_margin = math.inf
    mpc_interventions = cbf_interventions = 0
    completed = False
    reason = "duration_limit"
    warm_start = None
    failure_diagnostic = None
    import time

    try:
        for cycle in range(math.ceil(maximum_seconds / mpc.config.time_step)):
            state = np.asarray(read_state(), dtype=float)
            if state.shape != (3,) or not np.isfinite(state).all():
                reason = "invalid_state"
                break
            trajectory.append(state.copy())
            current_clearance = float(clearance(state))
            minimum_clearance = min(minimum_clearance, current_clearance)
            if (
                not math.isfinite(current_clearance)
                or current_clearance <= clearance_abort_m
            ):
                reason = "clearance_guard"
                break
            if progress is not None and cycle % 10 == 0:
                progress(
                    {
                        "cycle": cycle,
                        "state": state.tolist(),
                        "simulated_seconds": len(commands) * physics_dt,
                        "minimum_clearance_m": minimum_clearance,
                    }
                )
            pose = RobotPose2D(*state)
            nominal = waypoint.compute_command(pose)
            if nominal.route_complete:
                completed, reason = True, "goal_reached"
                break
            horizon = waypoint.predict_commands(
                pose,
                horizon_steps=mpc.config.horizon_steps,
                time_step=mpc.config.time_step,
            )
            sequence = np.array(
                [[item.linear_velocity, item.angular_velocity] for item in horizon]
            )
            start = time.perf_counter()
            planned = mpc.plan(
                state,
                sequence,
                initial_command_guess=warm_start,
                configuration_space=field.configuration_space,
                h_full=field.h_full,
                guidance_fields=field.guidance_fields,
                temporal_safety_derivative=0.0,
                preferred_lateral_direction_grid_y=-1.0,
            )
            solve_times.append(time.perf_counter() - start)
            finite = all(
                np.isfinite(item).all()
                for item in (
                    planned.command_sequence,
                    planned.predicted_chassis_states,
                    planned.predicted_safety_states,
                    planned.predicted_safety_values,
                    planned.dtcbf_margins,
                )
            )
            if (
                not planned.success
                or planned.fallback_active
                or not finite
                or np.min(planned.predicted_safety_values)
                < -mpc.config.constraint_tolerance
                or np.min(planned.dtcbf_margins) < -mpc.config.constraint_tolerance
            ):
                reason = "mpc_fail_closed"
                failure_diagnostic = {
                    "state": state.tolist(),
                    "status": getattr(planned, "status", "unreported"),
                    "success": bool(planned.success),
                    "fallback_active": bool(planned.fallback_active),
                    "finite_prediction": bool(finite),
                    "nominal_sequence": sequence.tolist(),
                    "warm_start": None if warm_start is None else warm_start.tolist(),
                    "predicted_states": planned.predicted_chassis_states.tolist()
                    if finite else None,
                    "predicted_safety_values": planned.predicted_safety_values.tolist()
                    if finite else None,
                    "dtcbf_margins": planned.dtcbf_margins.tolist() if finite else None,
                }
                break
            minimum_mpc_margin = min(
                minimum_mpc_margin, float(np.min(planned.dtcbf_margins))
            )
            mpc_interventions += int(planned.intervention_active)
            warm_start = np.vstack((planned.command_sequence[1:], np.zeros((1, 2))))
            interrupted = False
            for _ in range(steps_per_cycle):
                state = np.asarray(read_state(), dtype=float)
                if state.shape != (3,) or not np.isfinite(state).all():
                    reason, interrupted = "invalid_state", True
                    break
                query = lookahead_state(
                    state, lookahead_distance=backup.config.lookahead_distance
                )
                filtered = backup.filter_command(
                    planned.linear_velocity,
                    planned.angular_velocity,
                    query_position_grid_xy=query[:2],
                    query_yaw_radians=float(query[2]),
                    configuration_space=field.configuration_space,
                    h_full=field.h_full,
                    guidance_fields=field.guidance_fields,
                    temporal_safety_derivative=0.0,
                )
                if (
                    not filtered.constraint_feasible
                    or not filtered.constraint_satisfied
                    or filtered.emergency_stop_active
                    or not math.isfinite(filtered.cbf_margin)
                    or not math.isfinite(filtered.safe_linear_velocity)
                    or not math.isfinite(filtered.safe_angular_velocity)
                ):
                    reason, interrupted = "cbf_fail_closed", True
                    failure_diagnostic = {
                        "state": state.tolist(),
                        "emergency_reason": filtered.emergency_reason,
                        "unsafe_reason": filtered.constraint.unsafe_reason,
                        "query_on_unsafe_boundary": filtered.query_on_unsafe_boundary,
                        "sampled_safety_value": filtered.sampled_safety_value,
                        "requested_command": [
                            planned.linear_velocity,
                            planned.angular_velocity,
                        ],
                    }
                    break
                minimum_cbf_margin = min(minimum_cbf_margin, filtered.cbf_margin)
                cbf_interventions += int(filtered.intervention_active)
                advance(filtered.safe_linear_velocity, filtered.safe_angular_velocity)
                commands.append(
                    (filtered.safe_linear_velocity, filtered.safe_angular_velocity)
                )
                nominal_commands.append(
                    (nominal.linear_velocity, nominal.angular_velocity)
                )
                state = np.asarray(read_state(), dtype=float)
                if state.shape != (3,) or not np.isfinite(state).all():
                    reason, interrupted = "invalid_state", True
                    break
                trajectory.append(state.copy())
                observed_clearance = float(clearance(state))
                minimum_clearance = min(minimum_clearance, observed_clearance)
                if (
                    not math.isfinite(observed_clearance)
                    or observed_clearance <= clearance_abort_m
                ):
                    reason, interrupted = "clearance_guard", True
                    break
            if interrupted:
                break
            # Include arrival during the last permitted execution interval.
            # State validity and clearance have already been checked above;
            # an aborted interval must never become a successful arrival.
            if waypoint.compute_command(RobotPose2D(*state)).route_complete:
                completed, reason = True, "goal_reached"
                break
    finally:
        stop()

    states = np.asarray(trajectory)
    path_length = (
        float(np.linalg.norm(np.diff(states[:, :2], axis=0), axis=1).sum())
        if len(states) > 1
        else 0.0
    )
    return {
        "passed": completed
        and minimum_clearance > clearance_abort_m
        and mpc_interventions > 0,
        "route_complete": completed,
        "termination": reason,
        "failure_diagnostic": failure_diagnostic,
        "minimum_clearance_m": minimum_clearance
        if math.isfinite(minimum_clearance)
        else None,
        "minimum_mpc_margin": minimum_mpc_margin
        if math.isfinite(minimum_mpc_margin)
        else None,
        "minimum_cbf_margin": minimum_cbf_margin
        if math.isfinite(minimum_cbf_margin)
        else None,
        "mpc_intervention_count": mpc_interventions,
        "cbf_intervention_count": cbf_interventions,
        "simulated_seconds": len(commands) * physics_dt,
        "path_length_m": path_length,
        "mean_mpc_wall_seconds": float(np.mean(solve_times)) if solve_times else None,
        "maximum_mpc_wall_seconds": max(solve_times) if solve_times else None,
        "trajectory": states.tolist(),
        "commands": commands,
        "nominal_commands": nominal_commands,
    }

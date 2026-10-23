"""ORACLE moving-disk control comparisons; never perception or physical wheels."""

from dataclasses import replace
import math
import time

import numpy as np

from robot.control.unicycle_model import integrate_unicycle_euler
from robot.evaluation.encounter_guidance import apply_encounter_guidance
from robot.evaluation.guidance_control_audit import nominal_sequences
from robot.navigation.waypoint_controller import (
    RobotPose2D,
    Waypoint,
    WaypointController,
    WaypointControllerConfig,
)
from robot.safety.class_safety_policy import ClassSafetyParameters, ClassSafetyPolicy
from robot.safety.encounter_motion import EncounterMotionClassifier
from robot.safety.local_occupancy_grid import (
    LocalOccupancyGridConfig,
    build_local_occupancy_grid,
)
from robot.safety.motion_aware_passing import (
    MotionAwarePassing,
    MotionAwarePassingConfig,
)
from robot.safety.obstacle_tracking import TrackedObstacle, ObstacleTrackingResult
from robot.safety.safe_sage_cbf_filter import SafeSAGECBFSafetyFilter, SafeSAGECBFConfig
from robot.safety.safe_sage_field_pipeline import build_safe_sage_field_pipeline
from robot.safety.safe_sage_guidance_policy import (
    SafeSAGEGuidanceParameters,
    SafeSAGEGuidancePolicy,
)
from robot.safety.safe_sage_runtime import SafeSAGEControlRuntime, SafeSAGERuntimeField
from robot.safety.safe_sage_temporal_derivative import (
    SafeSAGETemporalDerivativeEstimator,
)
from robot.safety.safe_sage_unicycle_cbf_filter import SafeSAGEUnicycleCBFFilter
from robot.safety.safe_sage_unicycle_mpc import (
    SafeSAGEUnicycleMPC,
    SafeSAGEUnicycleMPCConfig,
)
from robot.safety.semantic_label_provider import (
    SemanticLabelResult,
    SemanticTrackAssignment,
)
from robot.safety.semantic_obstacle import SemanticObstacle
from robot.safety.safe_sage_configuration_space import (
    RectangularRobotFootprint,
    SafeSAGEConfigurationSpaceConfig,
)
from robot.safety.safe_sage_unicycle_cbf_filter import (
    shift_rectangular_footprint_to_lookahead,
)
from robot.safety.safe_sage_field_pipeline import SafeSAGEFieldPipelineConfig
from robot.safety.safe_sage_h_full import SafeSAGEHFullConfig
from robot.safety.poisson_safety_function import PoissonSafetyFunctionConfig
from robot.safety.tangential_laplace_guidance_field import (
    TangentialLaplaceGuidanceFieldConfig,
)


CASES = (
    "cross-left-right",
    "cross-right-left",
    "parallel-same",
    "parallel-opposite",
    "cross-then-parallel",
)
VARIANTS = ("fixed", "always-zero", "encounter")


def oracle_actor(case, now):
    """Common exogenous11.5 s onset; authored geometry/velocity, ORACLE only."""
    if case not in CASES or not math.isfinite(now) or now < 0:
        raise ValueError("known case and finite nonnegative time required")
    elapsed = max(0.0, now - 11.5)
    active = now >= 11.5
    if case.startswith("cross"):
        sign = -1 if case == "cross-right-left" else 1
        if elapsed < 4.0:
            return np.array([3.3, sign * (-1.0 + 0.5 * elapsed)]), np.array(
                [0.0, sign * 0.5 if active else 0.0]
            )
        # Cross-only continues laterally away; mixed turns parallel to the route.
        if case == "cross-then-parallel":
            return np.array([3.3 + 0.5 * (elapsed - 4.0), 1.0]), np.array([0.5, 0.0])
        return np.array([3.3, sign * (-1.0 + 0.5 * elapsed)]), np.array(
            [0.0, sign * 0.5]
        )
    sign = 1 if case == "parallel-same" else -1
    return np.array([3.3 + sign * 0.5 * elapsed, 1.0]), np.array(
        [sign * 0.5 if active else 0.0, 0.0]
    )


def _disk(center, radius):
    axis = np.arange(-radius, radius + 0.001, 0.08)
    x, y = np.meshgrid(axis, axis)
    keep = x * x + y * y <= radius * radius
    return np.column_stack((x[keep], y[keep])) + center


def _clearance(state, center, radius):
    c, s = np.cos(state[2]), np.sin(state[2])
    x, y = (center - state[:2]) @ np.array([[c, -s], [s, c]])
    return float(
        np.hypot(max(-0.587351 - x, 0.0, x - 0.14), max(abs(y) - 0.2466, 0.0)) - radius
    )


def summarize_oracle_rear_pass(result):
    """Score the existing fixed-clock ORACLE fixture, never native sensor records.

    Recorded states are interval endpoints: row.time is the start, and each
    interval is .2 s. Authored actor phase/position is independent scoring truth
    only. Route arrival alone is not acceptance of the proposed curved strategy.
    """
    case = result["case"]
    if case not in {
        "cross-left-right",
        "cross-right-left",
        "cross-then-parallel",
    } or not result.get("scope", "").startswith("ORACLE moving-disk virtual unicycle;"):
        raise ValueError("rear-pass scoring requires an ORACLE crossing fixture")
    rows = result["rows"]
    times = np.array([r["time"] + 0.2 for r in rows], dtype=float)
    robot = np.array([r["state"][:2] for r in rows], dtype=float)
    if (
        not len(times)
        or robot.shape != (len(times), 2)
        or not np.isfinite(times).all()
        or not np.isfinite(robot).all()
        or not np.isclose(times[0], 0.2, rtol=0, atol=1e-8)
        or not np.allclose(np.diff(times), 0.2, rtol=0, atol=1e-8)
    ):
        raise ValueError("finite ordered fixed-fixture endpoint trace required")
    actor = np.array([oracle_actor(case, t)[0] for t in times])
    motion = oracle_actor(case, 12.5)[1]
    motion /= np.linalg.norm(motion)
    walking = (
        (times >= 11.5)
        & (times < 15.5)
        & (np.linalg.norm(robot - actor, axis=1) <= 2.0)
    )
    excursion = float(np.max(-robot[walking] @ motion)) if walking.any() else 0.0
    # Preserve the initial transverse plane even when the actor later turns.
    plane = oracle_actor(case, 11.5)[0][0]
    earlier = np.vstack((np.zeros(2), robot[:-1]))
    earlier_times = np.concatenate(([0.0], times[:-1]))
    indices = np.flatnonzero((earlier[:, 0] < plane) & (robot[:, 0] >= plane))
    behind, crossing_time = None, None
    if len(indices):
        i = indices[0]
        f = (plane - earlier[i, 0]) / (robot[i, 0] - earlier[i, 0])
        crossing_time = float(earlier_times[i] + f * (times[i] - earlier_times[i]))
        position = earlier[i] + f * (robot[i] - earlier[i])
        behind = float((position - oracle_actor(case, crossing_time)[0]) @ motion)
    checks = dict(
        original_route_acceptance=bool(
            result.get("passed") is True
            and result.get("reason") == "goal_reached"
            and result.get("arrival_seconds") is not None
            and np.linalg.norm(robot[-1] - [3.9, 0.0]) <= 0.1
        ),
        independent_clearance=result["minimum_clearance_m"] > 0.02,
        analytical_checks=all(
            not r["emergency"]
            and math.isfinite(r["cbf_margin"])
            and r["cbf_margin"] >= -1e-6
            for r in rows
        ),
        visible_trailing_arc_during_walking=excursion >= 0.20,
        crossed_walking_plane_behind_actor=behind is not None and behind <= -0.10,
    )
    return dict(
        passed=all(checks.values()),
        checks=checks,
        maximum_trailing_excursion_during_walking_m=excursion,
        behind_projection_at_route_crossing_m=behind,
        route_crossing_time_seconds=crossing_time,
        scope="ORACLE fixed .2s virtual endpoint scoring only; authored actor phase/route used solely for evaluation; not sensors, physical braking or native rear-pass acceptance",
    )


def run_encounter_control_case(case, variant, *, maximum_seconds=40.0, progress=None):
    if case not in CASES or variant not in VARIANTS or not 0 < maximum_seconds <= 90:
        raise ValueError("bounded known comparison required")
    started = time.perf_counter()
    # Reuse the audited yaw/footprint/solver setup, not its static oracle geometry.
    grid_config = LocalOccupancyGridConfig(
        resolution=0.1,
        minimum_x=-2.0,
        maximum_x=7.0,
        minimum_y=-3.0,
        maximum_y=3.0,
        obstacle_inflation_radius=0.0,
    )
    footprint = shift_rectangular_footprint_to_lookahead(
        RectangularRobotFootprint(0.14, 0.587351, 0.2466, 0.2466, 0.05),
        lookahead_distance=0.1,
    )
    field_config = SafeSAGEFieldPipelineConfig(
        configuration_space_config=SafeSAGEConfigurationSpaceConfig(
            footprint, expansion_backend="vectorized"
        ),
        guidance_field_config=TangentialLaplaceGuidanceFieldConfig(
            solver_backend="sparse_direct",
            interface_backend="vectorized",
            allow_dense_scene_arbitration=True,
            merge_equal_policy_regions=True,
        ),
        h_full_config=SafeSAGEHFullConfig(
            poisson_config=PoissonSafetyFunctionConfig(solver_backend="sparse_direct")
        ),
    )
    paper = SafeSAGECBFSafetyFilter(
        SafeSAGECBFConfig(
            gamma=1.0,
            minimum_planar_velocity=-0.25,
            maximum_planar_velocity=0.25,
            minimum_angular_velocity=-0.8,
            maximum_angular_velocity=0.8,
        )
    )
    mpc = SafeSAGEUnicycleMPC(
        paper,
        SafeSAGEUnicycleMPCConfig(
            horizon_steps=6,
            time_step=0.2,
            maximum_solver_iterations=1000,
            maximum_nonlinear_rescue_iterations=20,
            safety_substeps_per_interval=12,
            integration_substeps_per_interval=12,
            enable_progress_seeds=True,
            enable_recovery_cost_pruning=True,
            enable_recovery_feasibility_pruning=True,
            progress_seed_linear_fraction=0.9,
            configuration_position_padding_m=0.025,
            configuration_yaw_padding_radians=0.21,
            configuration_padding_initial_fraction=0.1,
        ),
    )
    control = SafeSAGEControlRuntime(
        mpc=mpc,
        backup_filter=SafeSAGEUnicycleCBFFilter(paper),
        crossing_strategy="legacy" if variant == "fixed" else "yield-then-go",
    )
    passing = MotionAwarePassing(
        MotionAwarePassingConfig(require_visual_person_selection=True)
    )
    classifier = EncounterMotionClassifier()
    estimator = SafeSAGETemporalDerivativeEstimator()
    waypoint = WaypointController(
        [Waypoint(2.2, 0.0), Waypoint(3.1, 0.0), Waypoint(3.9, 0.0)],
        config=WaypointControllerConfig(
            rotate_in_place_threshold_radians=np.deg2rad(100)
        ),
    )
    state = np.zeros(3)
    previous = None
    rows = []
    reason = "maximum_seconds"
    minimum_clearance = math.inf
    route_completed = None
    for step in range(math.ceil((maximum_seconds + 1.0) / 0.2)):
        now = step * 0.2
        if route_completed is None and now >= maximum_seconds:
            break
        if route_completed is not None and now - route_completed >= 1.0 - 1e-9:
            break
        center, velocity = oracle_actor(case, now)
        # Geometry beyond this fixed oracle field is omitted only after it is
        # >3 m from the robot. This is a declared test fixture, not free-space evidence.
        present = np.linalg.norm(center - state[:2]) <= 3.0 or (
            -1.0 < center[0] < 6.0 and abs(center[1]) < 2.0
        )
        centers = [(7, center, 0.65, velocity)] if present else []
        centers.append((9, np.array([4.5, -2.0]), 0.25, np.zeros(2)))
        points = []
        ids = []
        tracks = []
        semantics = []
        guidance = {}
        audit = []
        for key, xy, radius, v in centers:
            pts = _disk(xy, radius)
            inside = (pts[:, 0] > -1.9) & (pts[:, 0] < 6.9) & (abs(pts[:, 1]) < 2.9)
            pts = pts[inside]
            if not len(pts):
                continue
            points.append(pts)
            ids.extend([key] * len(pts))
            track = TrackedObstacle(
                key,
                tuple(xy),
                tuple(xy - state[:2]),
                tuple(v),
                tuple(v),
                len(pts),
                step + 1,
                step > 0,
            )
            tracks.append(track)
            semantics.append(
                SemanticObstacle(track, "unknown", 0.0, "oracle_encounter_fixture")
            )
            parameters = ClassSafetyParameters(
                1.0 if key == 7 else 0.5, 1.7 if key == 7 else 1.1
            )
            guidance[key] = SafeSAGEGuidanceParameters(parameters, 1.25, "right")
            audit.append(
                dict(
                    track_or_group_id=key,
                    cra_phi=parameters.risk,
                    applied_field_risk=parameters.risk,
                    boundary_flux=-parameters.boundary_flux_magnitude,
                    **{"lambda": 1.25},
                )
            )
        points = np.concatenate(points)
        ids = np.array(ids, dtype=np.int64)
        tracking = ObstacleTrackingResult(
            1.0 + now,
            np.zeros((len(points), 2)),
            ids,
            tuple(tracks),
            len(tracks),
            len(tracks),
            0,
        )
        nominal = waypoint.compute_command(RobotPose2D(*state))
        if nominal.route_complete and route_completed is None:
            route_completed = now
            reason = "goal_reached"
        goal = (
            (waypoint.current_waypoint.x, waypoint.current_waypoint.y)
            if waypoint.current_waypoint
            else (3.9, 0.0)
        )
        human_ids = tuple(x.track_id for x in tracks if x.track_id == 7)
        sides, passing_audit = passing.update(
            tracking,
            robot_xy=state[:2],
            robot_velocity_xy=np.zeros(2)
            if not rows
            else rows[-1]["velocity_world_xy"],
            goal_xy=goal,
            visual_person_track_ids=human_ids,
            nominal_route_direction_world_xy=(1.0, 0.0)
            if variant == "encounter"
            else None,
        )
        classification = classifier.update(
            tracking,
            robot_xy=state[:2],
            goal_xy=goal,
            human_track_ids=human_ids,
            nominal_route_direction_world_xy=(1.0, 0.0)
            if variant == "encounter"
            else None,
        )
        guidance = {
            k: replace(v, passing_side=sides.get(k, v.passing_side))
            for k, v in guidance.items()
        }
        yield_ids = None
        if variant == "encounter":
            guidance, audit, yield_ids = apply_encounter_guidance(
                guidance, audit, classification, passing_audit, source_time=1.0 + now
            )
        elif variant == "always-zero":
            guidance = {
                k: replace(v, tangential_strength=0.0) for k, v in guidance.items()
            }
        parameters = ClassSafetyParameters(0.5, 1.1)
        fallback = SafeSAGEGuidancePolicy(
            {
                "unknown": SafeSAGEGuidanceParameters(
                    parameters, 0.0 if variant == "always-zero" else 1.25, "right"
                )
            }
        )
        grid = build_local_occupancy_grid(
            points,
            point_track_ids=ids,
            semantic_obstacles=semantics,
            class_safety_policy=ClassSafetyPolicy({"unknown": parameters}),
            config=grid_config,
            track_safety_parameters={
                k: v.safety_parameters for k, v in guidance.items()
            },
        )
        pipeline = build_safe_sage_field_pipeline(
            grid,
            guidance_policy=fallback,
            config=field_config,
            semantic_obstacles=semantics,
            track_guidance_parameters=guidance,
        )
        derivative = (
            0.0
            if previous is None
            else estimator.estimate(
                previous_h_full=previous.pipeline.h_full,
                current_h_full=pipeline.h_full,
                grid_config=grid_config,
                time_delta_seconds=0.2,
                egomotion_translation_grid_xy=(0.0, 0.0),
                previous_estimate=None
                if isinstance(previous.temporal_safety_derivative, float)
                else previous.temporal_safety_derivative,
            )
        )
        field = SafeSAGERuntimeField(
            step + 1,
            1.0 + now,
            (0.0, 0.0),
            0.0,
            len(points),
            len(points),
            tracking,
            SemanticLabelResult(tuple(SemanticTrackAssignment(s) for s in semantics)),
            pipeline,
            derivative,
            0.0,
            motion_passing_reference_track_ids=tuple(sides),
            motion_passing_reference_sides=tuple(s.value for s in sides.values()),
            motion_passing_reference_source_timestamp_seconds=passing.encounter[2]
            if sides
            else None,
            crossing_yield_track_ids=yield_ids,
        )
        previous = field
        nominal_sequence, _ = nominal_sequences(state, waypoint, 6)
        plan = (
            None
            if route_completed is not None
            else control.plan(state, nominal_sequence, field)
        )
        command = (
            np.zeros(2)
            if plan is None or not plan.success or plan.fallback_active
            else np.array(plan.first_command)
        )
        minimum_margin = math.inf
        emergency = False
        for sub in range(12):
            filtered = control.filter_command(command, state, field)
            executed = np.array(
                [filtered.safe_linear_velocity, filtered.safe_angular_velocity]
            )
            minimum_margin = min(minimum_margin, filtered.cbf_margin)
            emergency |= filtered.emergency_stop_active
            state = integrate_unicycle_euler(state, executed, time_step=1.0 / 60.0)
            actual_center, _ = oracle_actor(case, now + (sub + 1) / 60.0)
            minimum_clearance = min(
                minimum_clearance,
                _clearance(state, actual_center, 0.4),
                _clearance(state, np.array([4.5, -2.0]), 0.25),
            )
        rows.append(
            dict(
                time=now,
                state=state.tolist(),
                actor_xy=center.tolist(),
                actor_velocity=velocity.tolist(),
                classification=classification,
                passing=passing_audit,
                lambdas={k: v.tangential_strength for k, v in guidance.items()},
                command=command.tolist(),
                velocity_world_xy=(
                    executed[0] * np.array([np.cos(state[2]), np.sin(state[2])])
                ).tolist(),
                cbf_margin=minimum_margin,
                emergency=emergency,
                plan_success=None if plan is None else plan.success,
                checked_yield=getattr(plan, "crossing_maneuver", {}).get("action")
                == "checked_yield",
            )
        )
        if progress and step % 10 == 0:
            progress(dict(case=case, variant=variant, time=now, state=state.tolist()))
        if minimum_clearance <= 0.02 or emergency:
            reason = (
                "independent_clearance"
                if minimum_clearance <= 0.02
                else "analytical_emergency"
            )
            break
    passed = (
        reason == "goal_reached"
        and minimum_clearance > 0.02
        and np.linalg.norm(state[:2] - [3.9, 0.0]) <= 0.1
    )
    return dict(
        case=case,
        variant=variant,
        passed=bool(passed),
        reason=reason,
        rows=rows,
        arrival_seconds=route_completed,
        minimum_clearance_m=minimum_clearance,
        endpoint_state=state.tolist(),
        maximum_lateral_excursion_m=max(
            (abs(r["state"][1]) for r in rows), default=0.0
        ),
        wall_seconds=time.perf_counter() - started,
        scope="ORACLE moving-disk virtual unicycle; authored geometry, identity and velocity; no sensors, latency, physical robot stopping or CRA estimation validation; same case/onset across variants",
    )

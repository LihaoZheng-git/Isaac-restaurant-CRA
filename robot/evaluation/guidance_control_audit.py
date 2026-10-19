"""Fixed-field, zero-wheel comparison of planar and unicycle Safe-SAGE.

All occupancy and phi here are authored ORACLE inputs. This isolates control
realization; it neither validates perception nor supplies a navigation field.
"""

from dataclasses import asdict, dataclass
import math
import time

import numpy as np

from robot.control.unicycle_model import (
    integrate_unicycle_euler,
    lookahead_control_matrix,
    lookahead_state,
)
from robot.navigation.waypoint_controller import (
    RobotPose2D,
    Waypoint,
    WaypointController,
    WaypointControllerConfig,
)
from robot.safety.class_safety_policy import ClassSafetyParameters, ClassSafetyPolicy
from robot.safety.local_occupancy_grid import (
    LocalOccupancyGridConfig,
    build_local_occupancy_grid,
)
from robot.safety.obstacle_tracking import TrackedObstacle
from robot.safety.poisson_safety_function import PoissonSafetyFunctionConfig
from robot.safety.safe_sage_cbf_filter import SafeSAGECBFConfig, SafeSAGECBFSafetyFilter
from robot.safety.safe_sage_configuration_space import (
    RectangularRobotFootprint,
    SafeSAGEConfigurationSpaceConfig,
)
from robot.safety.safe_sage_field_pipeline import (
    SafeSAGEFieldPipelineConfig,
    build_safe_sage_field_pipeline,
)
from robot.safety.safe_sage_guidance_policy import (
    PassingSide,
    SafeSAGEGuidanceParameters,
    SafeSAGEGuidancePolicy,
)
from robot.safety.safe_sage_h_full import SafeSAGEHFullConfig
from robot.safety.safe_sage_planar_mpc import SafeSAGEPlanarMPC, SafeSAGEPlanarMPCConfig
from robot.safety.safe_sage_unicycle_cbf_filter import (
    SafeSAGEUnicycleCBFConfig,
    SafeSAGEUnicycleCBFFilter,
    shift_rectangular_footprint_to_lookahead,
)
from robot.safety.safe_sage_unicycle_mpc import (
    SafeSAGEUnicycleMPC,
    SafeSAGEUnicycleMPCConfig,
)
from robot.safety.semantic_obstacle import SemanticObstacle
from robot.safety.tangential_laplace_guidance_field import (
    TangentialLaplaceGuidanceFieldConfig,
)


@dataclass(frozen=True)
class GuidanceAuditConfig:
    resolution: float = 0.1
    oracle_phi: float = 0.65
    maximum_seconds: float = 40.0
    execution_dt: float = 0.05
    cbf_angular_weight: float = 1.0
    horizon_steps: int = 30
    solver_iterations: int = 1000
    progress_seed_fraction: float = 0.5

    def __post_init__(self):
        for key, value in asdict(self).items():
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"positive finite {key} required")
        if self.oracle_phi > 1 or self.resolution > 0.2:
            raise ValueError("phi or resolution outside audit scope")
        if self.progress_seed_fraction >= 1:
            raise ValueError("progress seed fraction must be below one")
        if not math.isclose(0.2 / self.execution_dt, round(0.2 / self.execution_dt)):
            raise ValueError("execution interval must divide MPC interval")
        if any(
            not isinstance(v, int) for v in (self.horizon_steps, self.solver_iterations)
        ):
            raise ValueError("integer horizon and solver budget required")


def build_audit_field(side, config):
    """One disk and the measured rectangle; identical pipeline per controller."""
    side = PassingSide(side)
    parameters = ClassSafetyParameters(config.oracle_phi, 0.5 + 1.2 * config.oracle_phi)
    policy = ClassSafetyPolicy({name: parameters for name in ("unknown", "object")})
    guidance = SafeSAGEGuidancePolicy(
        {
            name: SafeSAGEGuidanceParameters(parameters, 1.25, side)
            for name in ("unknown", "object")
        }
    )
    axis = np.arange(-0.25, 0.251, 0.025)
    xx, yy = np.meshgrid(axis, axis)
    disk = xx**2 + yy**2 <= 0.25**2
    points = np.column_stack((1.6 + xx[disk], yy[disk]))
    track = TrackedObstacle(
        7, (1.6, 0.0), (1.6, 0.0), (0.0, 0.0), (0.0, 0.0), len(points), 1, False
    )
    semantic = SemanticObstacle(track, "object", 1.0, "authored_control_audit_oracle")
    grid = build_local_occupancy_grid(
        points,
        point_track_ids=np.full(len(points), 7),
        semantic_obstacles=(semantic,),
        class_safety_policy=policy,
        config=LocalOccupancyGridConfig(
            resolution=config.resolution,
            minimum_x=-1.5,
            maximum_x=4.5,
            minimum_y=-2.5,
            maximum_y=2.5,
            obstacle_inflation_radius=0.0,
        ),
    )
    footprint = RectangularRobotFootprint(0.14, 0.587351, 0.2466, 0.2466, 0.05)
    shifted = shift_rectangular_footprint_to_lookahead(
        footprint, lookahead_distance=0.1
    )
    pipeline = build_safe_sage_field_pipeline(
        grid,
        guidance_policy=guidance,
        semantic_obstacles=(semantic,),
        config=SafeSAGEFieldPipelineConfig(
            configuration_space_config=SafeSAGEConfigurationSpaceConfig(
                shifted, expansion_backend="vectorized"
            ),
            guidance_field_config=TangentialLaplaceGuidanceFieldConfig(
                solver_backend="sparse_direct", interface_backend="vectorized"
            ),
            h_full_config=SafeSAGEHFullConfig(
                poisson_config=PoissonSafetyFunctionConfig(
                    solver_backend="sparse_direct"
                )
            ),
        ),
    )
    return pipeline


def nominal_sequences(chassis, waypoint, horizon_steps, time_step=0.2):
    """Map the same obstacle-unaware horizon into both command spaces."""
    commands = waypoint.predict_commands(
        RobotPose2D(*chassis), horizon_steps=horizon_steps, time_step=time_step
    )
    wheel = np.array([[c.linear_velocity, c.angular_velocity] for c in commands])
    pose = np.array(chassis, copy=True)
    planar = []
    for u in wheel:
        planar.append(lookahead_control_matrix(pose[2], lookahead_distance=0.1) @ u)
        pose = integrate_unicycle_euler(pose, u, time_step=time_step)
    return wheel, np.array(planar)


def run_control_audit(pipeline, kind, config, *, progress=None):
    if kind not in {"planar", "unicycle"}:
        raise ValueError("planar or unicycle required")
    paper = SafeSAGECBFSafetyFilter(
        SafeSAGECBFConfig(
            gamma=1.0,
            minimum_planar_velocity=-0.25,
            maximum_planar_velocity=0.25,
            minimum_angular_velocity=-0.8,
            maximum_angular_velocity=0.8,
        )
    )
    common = dict(
        horizon_steps=config.horizon_steps,
        time_step=0.2,
        maximum_solver_iterations=config.solver_iterations,
        maximum_sqp_iterations=12,
        angular_velocity_weight=0.02,
        command_smoothness_weight=0.005,
    )
    substeps = round(0.2 / config.execution_dt)
    mpc = (
        SafeSAGEPlanarMPC(paper, SafeSAGEPlanarMPCConfig(**common))
        if kind == "planar"
        else SafeSAGEUnicycleMPC(
            paper,
            SafeSAGEUnicycleMPCConfig(
                **common,
                safety_substeps_per_interval=substeps,
                integration_substeps_per_interval=substeps,
                enable_progress_seeds=True,
                enable_recovery_cost_pruning=True,
                enable_recovery_feasibility_pruning=True,
                maximum_nonlinear_rescue_iterations=20,
                progress_seed_linear_fraction=config.progress_seed_fraction,
            ),
        )
    )
    backup = SafeSAGEUnicycleCBFFilter(
        paper,
        SafeSAGEUnicycleCBFConfig(angular_velocity_weight=config.cbf_angular_weight),
    )
    field = dict(
        configuration_space=pipeline.configuration_space,
        h_full=pipeline.h_full,
        guidance_fields=pipeline.guidance_fields,
    )
    waypoint = WaypointController(
        [Waypoint(3.3, 0.0)],
        config=WaypointControllerConfig(
            rotate_in_place_threshold_radians=math.radians(100)
        ),
    )
    chassis = np.zeros(3)
    safety = lookahead_state(chassis, lookahead_distance=0.1)
    records, trajectory, warm, plans = [], [chassis.tolist()], None, []
    reason, plan = "maximum_seconds", None
    for step in range(round(config.maximum_seconds / config.execution_dt)):
        now = step * config.execution_dt
        nominal = waypoint.compute_command(RobotPose2D(*chassis))
        if nominal.route_complete:
            reason = "goal_reached"
            break
        solve_time = None
        if step % substeps == 0:
            wheel, planar = nominal_sequences(chassis, waypoint, config.horizon_steps)
            tic = time.perf_counter()
            plan = mpc.plan(
                safety if kind == "planar" else chassis,
                planar if kind == "planar" else wheel,
                initial_command_guess=warm,
                **field,
            )
            solve_time = time.perf_counter() - tic
            plans.append(
                dict(
                    time=now,
                    commands=plan.command_sequence.tolist(),
                    predicted_states=(
                        plan.predicted_states
                        if kind == "planar"
                        else plan.predicted_chassis_states
                    ).tolist(),
                    cost=plan.objective_value,
                    status=plan.status,
                    dtcbf_margins=plan.dtcbf_margins.tolist(),
                )
            )
            warm = (
                np.vstack(
                    (
                        plan.command_sequence[1:],
                        np.zeros_like(plan.command_sequence[:1]),
                    )
                )
                if plan.success
                else None
            )
        current = paper.evaluate_constraint(
            query_position_grid_xy=safety[:2], query_yaw_radians=safety[2], **field
        )
        command = np.array(plan.first_command)
        if not plan.success:
            reason = "mpc_failed:" + plan.status
            break
        if kind == "planar":
            filtered = paper.filter_command(
                command[:2],
                command[2],
                query_position_grid_xy=safety[:2],
                query_yaw_radians=safety[2],
                **field,
            )
            executed = np.array(
                [*filtered.safe_planar_velocity_grid_xy, filtered.safe_angular_velocity]
            )
            candidate = safety + config.execution_dt * executed
            next_chassis = candidate - [
                0.1 * math.cos(candidate[2]),
                0.1 * math.sin(candidate[2]),
                0.0,
            ]
        else:
            filtered = backup.filter_command(
                *command,
                query_position_grid_xy=safety[:2],
                query_yaw_radians=safety[2],
                **field,
            )
            executed = np.array(
                [filtered.safe_linear_velocity, filtered.safe_angular_velocity]
            )
            next_chassis = integrate_unicycle_euler(
                chassis, executed, time_step=config.execution_dt
            )
            candidate = lookahead_state(next_chassis, lookahead_distance=0.1)
        record = dict(
            time=now,
            chassis=chassis.tolist(),
            guidance=list(current.sampled_guidance_grid_xy),
            planar_cbf_normal=list(current.constraint_normal),
            unicycle_cbf_normal=(
                lookahead_control_matrix(chassis[2], lookahead_distance=0.1).T
                @ np.asarray(current.constraint_normal)
            ).tolist(),
            nominal=[nominal.linear_velocity, nominal.angular_velocity],
            mpc=command.tolist(),
            executed=executed.tolist(),
            cbf_margin=filtered.cbf_margin,
            mpc_margin=plan.minimum_dtcbf_margin,
            status=plan.status,
            restoration_used=plan.exact_restoration_used
            if kind == "planar"
            else plan.restoration_used,
            solve_wall_seconds=solve_time,
            safety_value=current.sampled_safety_value,
        )
        records.append(record)
        if progress is not None and step % (substeps * 10) == 0:
            progress(
                {
                    key: record[key]
                    for key in ("time", "chassis", "mpc", "executed", "status")
                }
            )
        if filtered.emergency_stop_active:
            reason = "cbf_stop:" + str(filtered.emergency_reason)
            break
        if paper.is_unsafe_configuration(
            query_position_grid_xy=candidate[:2],
            query_yaw_radians=candidate[2],
            configuration_space=pipeline.configuration_space,
            h_full=pipeline.h_full,
        ):
            reason = "executed_geometry_rejected"
            break
        chassis, safety = next_chassis, candidate
        trajectory.append(chassis.tolist())
        if len(trajectory) > round(4.0 / config.execution_dt):
            recent = np.asarray(trajectory[-round(4.0 / config.execution_dt) :])
            if (
                np.max(np.linalg.norm(recent[:, :2] - chassis[:2], axis=1)) < 0.001
                and np.max(np.abs(recent[:, 2] - chassis[2])) < 0.001
            ):
                reason = "four_second_stationary_local_solution"
                break
    arrived = reason == "goal_reached"
    checks = dict(
        arrived=arrived,
        executed_geometry=reason != "executed_geometry_rejected",
        cbf_margins=bool(records) and min(r["cbf_margin"] for r in records) >= -1e-8,
        mpc_margins=bool(records) and min(r["mpc_margin"] for r in records) >= -1e-6,
    )
    return dict(
        kind=kind,
        passed=all(checks.values()),
        checks=checks,
        reason=reason,
        simulated_seconds=(len(trajectory) - 1) * config.execution_dt,
        records=records,
        plans=plans,
        trajectory=trajectory,
        maximum_positive_lateral_m=max(p[1] for p in trajectory),
        maximum_negative_lateral_m=min(p[1] for p in trajectory),
        configuration=asdict(config),
        mpc_configuration=asdict(mpc.config),
        fixed_lambda=1.25,
        nonzero_wheel_commands=0,
        scope="authored fixed ORACLE occupancy/phi; virtual controller comparison; no sensor or wheel acceptance",
    )

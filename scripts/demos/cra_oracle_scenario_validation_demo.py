"""Validate the complete CRA-to-control path before learned risk estimation.

This controlled Isaac Sim demo holds perception, geometry, robot state, and
the nominal waypoint command fixed.  One real PandarXT-32 return cluster from
the session-only blue proxy is reused for four manually specified restaurant
risk hypotheses:

* ordinary chair (potential-dominant);
* fast crossing person (kinetic-dominant);
* exposed knife (contact-dominant); and
* waiter carrying hot soup (thermal-dominant).

For each counterfactual, the live runtime computes CRA ``phi``, maps it to
per-track Safe-SAGE parameters, rebuilds the yaw-dependent ``h_full`` field,
runs stationary unicycle MPC, and applies the analytical CBF backup to the
first MPC command.  No command reaches Nova Carter's wheels.

The oracle values are illustrative software-validation hypotheses.  This
demo does not claim that they are calibrated measurements or camera output;
that learned estimation is the next milestone.

Run from the project root:

    python -u scripts/demos/cra_oracle_scenario_validation_demo.py

For an automated run:

    $env:ISAAC_SIM_HEADLESS = "1"
    python -u scripts/demos/cra_oracle_scenario_validation_demo.py
"""

from dataclasses import dataclass
import math
import os
from pathlib import Path
import sys
import time
import traceback


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# This module owns the one SimulationApp and exposes the already validated
# restaurant-placement, LiDAR, tracking, and footprint helpers.
from scripts.demos import safe_sage_h_full_cbf_demo as stationary


simulation_app = stationary.simulation_app


import numpy as np
import isaacsim.core.experimental.utils.app as app_utils
from isaacsim.core.simulation_manager import SimulationManager
from pxr import Gf, Usd, UsdGeom

from robot.control.differential_drive import NovaCarterController
from robot.navigation.waypoint_controller import RobotPose2D
from robot.risk.cra_oracle_scenarios import (
    CRAOracleScenario,
    controlled_cra_oracle_scenarios,
)
from robot.runtime.simulation_loop import (
    advance_playing_steps,
    wait_until_playing,
)
from robot.safety.cra_safe_sage_policy import (
    CRASafeSAGEPolicy,
    CRASafeSAGEPolicyConfig,
)
from robot.safety.cra_track_policy import TrackIdCRARiskProfileProvider
from robot.safety.local_occupancy_grid import LocalOccupancyGridConfig
from robot.safety.obstacle_detection import ObstacleDetectionConfig
from robot.safety.obstacle_tracking import (
    ObstacleTracker,
    ObstacleTrackerConfig,
)
from robot.safety.safe_sage_cbf_filter import (
    SafeSAGECBFConfig,
    SafeSAGECBFSafetyFilter,
)
from robot.safety.safe_sage_configuration_space import (
    RectangularRobotFootprint,
)
from robot.safety.safe_sage_guidance_policy import SafeSAGEGuidancePolicy
from robot.safety.safe_sage_runtime import (
    SafeSAGEControlRuntime,
    SafeSAGEPerceptionObservation,
    SafeSAGEPerceptionRuntime,
    SafeSAGERuntimeField,
)
from robot.safety.safe_sage_unicycle_cbf_filter import (
    SafeSAGEUnicycleCBFConfig,
    SafeSAGEUnicycleCBFFilter,
    SafeSAGEUnicycleCBFResult,
    shift_rectangular_footprint_to_lookahead,
)
from robot.safety.safe_sage_unicycle_mpc import (
    SafeSAGEUnicycleMPC,
    SafeSAGEUnicycleMPCConfig,
    SafeSAGEUnicycleMPCResult,
)
from robot.safety.semantic_label_provider import (
    SemanticClassLabel,
    TrackIdSemanticLabelProvider,
)
from robot.scene.loader import load_robot_scene
from robot.scene.transforms import compute_relative_prim_transform
from robot.sensor.lidar import NovaCarterLidar


SHARED_SEMANTIC_CLASS = "object"
PREFERRED_LATERAL_DIRECTION_GRID_Y = -1.0

# This final pre-vision CRA experiment intentionally enables the policy's
# monotone lambda mapping.  The endpoint range is bounded by the repository's
# existing chair/person Safe-SAGE settings; it remains an ablation hypothesis,
# not a calibrated perception result.
CRA_TANGENTIAL_STRENGTH_AT_ZERO_RISK = 0.75
CRA_TANGENTIAL_STRENGTH_AT_UNIT_RISK = 1.25

MPC_HORIZON_STEPS = 30
MPC_TIME_STEP = 0.20
MPC_LINEAR_VELOCITY_WEIGHT = 1.0
MPC_ANGULAR_VELOCITY_WEIGHT = 0.02
MPC_COMMAND_SMOOTHNESS_WEIGHT = 0.005
MPC_MAXIMUM_SQP_ITERATIONS = 12
MPC_MAXIMUM_SOLVER_ITERATIONS = 4000

PREDICTION_ROOT_PRIM_PATH = "/World/CRAOraclePredictions"
PREDICTION_MARKER_RADIUS = 0.025
PREDICTION_MARKER_BASE_HEIGHT = 0.08
PREDICTION_MARKER_HEIGHT_STEP = 0.025
PREDICTION_COLORS = (
    (0.25, 0.95, 0.30),
    (0.20, 0.90, 1.00),
    (1.00, 0.55, 0.10),
    (1.00, 0.20, 0.80),
)

AUTOCLOSE_AFTER_REPORT = os.environ.get(
    "ISAAC_SIM_AUTOCLOSE",
    "0",
).strip().casefold() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ScenarioControlEvaluation:
    """Field and stationary controller results for one oracle profile."""

    scenario: CRAOracleScenario
    field: SafeSAGERuntimeField
    mpc_result: SafeSAGEUnicycleMPCResult
    backup_result: SafeSAGEUnicycleCBFResult
    mpc_wall_seconds: float

    @property
    def policy_assignment(self):
        """Return the sole controlled track's CRA assignment."""

        result = self.field.cra_policy_result
        if result is None:
            raise RuntimeError("The scenario field has no CRA diagnostics.")
        return result.assignment_for(self.field.primary_track_id)

    @property
    def maximum_rightward_displacement(self) -> float:
        """Return maximum predicted displacement toward robot-right."""

        return float(-np.min(self.mpc_result.predicted_chassis_states[:, 1]))


@dataclass(frozen=True)
class ForwardProbeComparison:
    """Same nominal command filtered at one shared hypothetical state."""

    chassis_x: float
    results: tuple[SafeSAGEUnicycleCBFResult, ...]
    command_spread: float


def resolve_scene_path() -> Path:
    """Return an optional external scene path for isolated test clones."""

    configured = os.environ.get("ISAAC_RESTAURANT_SCENE_PATH")
    path = stationary.ROBOT_SCENE_PATH if configured is None else Path(configured)
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            "Restaurant scene not found. Set ISAAC_RESTAURANT_SCENE_PATH "
            f"or restore the project scene: {path}"
        )
    return path


def create_perception_runtime(
    *,
    guidance_policy: SafeSAGEGuidancePolicy,
    field_config,
) -> SafeSAGEPerceptionRuntime:
    """Create one fresh runtime so every counterfactual starts identically."""

    return SafeSAGEPerceptionRuntime(
        obstacle_tracker=ObstacleTracker(
            ObstacleTrackerConfig(
                cluster_radius=stationary.TRACKER_CLUSTER_RADIUS,
                minimum_cluster_points=(
                    stationary.TRACKER_MINIMUM_CLUSTER_POINTS
                ),
                maximum_association_distance=(
                    stationary.TRACKER_MAXIMUM_ASSOCIATION_DISTANCE
                ),
            )
        ),
        class_policy=stationary.class_policy_from_guidance_policy(
            guidance_policy
        ),
        guidance_policy=guidance_policy,
        grid_config=LocalOccupancyGridConfig(
            resolution=stationary.GRID_RESOLUTION,
            minimum_x=stationary.GRID_MINIMUM_X,
            maximum_x=stationary.GRID_MAXIMUM_X,
            minimum_y=stationary.GRID_MINIMUM_Y,
            maximum_y=stationary.GRID_MAXIMUM_Y,
            obstacle_inflation_radius=(
                stationary.GRID_OBSTACLE_INFLATION_RADIUS
            ),
        ),
        field_config=field_config,
        enable_temporal_updates=False,
        cra_policy=CRASafeSAGEPolicy(
            config=CRASafeSAGEPolicyConfig(
                tangential_strength_at_zero_risk=(
                    CRA_TANGENTIAL_STRENGTH_AT_ZERO_RISK
                ),
                tangential_strength_at_unit_risk=(
                    CRA_TANGENTIAL_STRENGTH_AT_UNIT_RISK
                ),
            )
        ),
    )


def create_control_runtime(
    controller: NovaCarterController,
) -> tuple[SafeSAGEControlRuntime, SafeSAGEUnicycleMPCConfig]:
    """Create the shared Safe-SAGE MPC and analytical CBF layers."""

    paper_filter = SafeSAGECBFSafetyFilter(
        SafeSAGECBFConfig(
            gamma=stationary.CBF_GAMMA,
            minimum_planar_velocity=(-controller.config.max_linear_speed),
            maximum_planar_velocity=controller.config.max_linear_speed,
            minimum_angular_velocity=(-controller.config.max_angular_speed),
            maximum_angular_velocity=controller.config.max_angular_speed,
            planar_velocity_weight=(
                stationary.CBF_LINEAR_VELOCITY_WEIGHT
            ),
            angular_velocity_weight=(
                stationary.CBF_ANGULAR_VELOCITY_WEIGHT
            ),
        )
    )
    mpc_config = SafeSAGEUnicycleMPCConfig(
        horizon_steps=MPC_HORIZON_STEPS,
        time_step=MPC_TIME_STEP,
        gamma=stationary.CBF_GAMMA,
        lookahead_distance=stationary.UNICYCLE_LOOKAHEAD_DISTANCE,
        minimum_linear_velocity=0.0,
        maximum_linear_velocity=controller.config.max_linear_speed,
        minimum_angular_velocity=(-controller.config.max_angular_speed),
        maximum_angular_velocity=controller.config.max_angular_speed,
        linear_velocity_weight=MPC_LINEAR_VELOCITY_WEIGHT,
        angular_velocity_weight=MPC_ANGULAR_VELOCITY_WEIGHT,
        command_smoothness_weight=MPC_COMMAND_SMOOTHNESS_WEIGHT,
        maximum_sqp_iterations=MPC_MAXIMUM_SQP_ITERATIONS,
        maximum_solver_iterations=MPC_MAXIMUM_SOLVER_ITERATIONS,
    )
    backup_filter = SafeSAGEUnicycleCBFFilter(
        paper_filter,
        SafeSAGEUnicycleCBFConfig(
            lookahead_distance=stationary.UNICYCLE_LOOKAHEAD_DISTANCE,
            minimum_linear_velocity=0.0,
            maximum_linear_velocity=controller.config.max_linear_speed,
            minimum_angular_velocity=(-controller.config.max_angular_speed),
            maximum_angular_velocity=controller.config.max_angular_speed,
            linear_velocity_weight=(
                stationary.CBF_LINEAR_VELOCITY_WEIGHT
            ),
            angular_velocity_weight=(
                stationary.CBF_ANGULAR_VELOCITY_WEIGHT
            ),
        ),
    )
    return (
        SafeSAGEControlRuntime(
            mpc=SafeSAGEUnicycleMPC(paper_filter, mpc_config),
            backup_filter=backup_filter,
        ),
        mpc_config,
    )


def field_state_to_world_xy(
    state_grid: np.ndarray,
    *,
    anchor_pose: RobotPose2D,
) -> tuple[float, float]:
    """Transform one fixed-field state into the restaurant world frame."""

    cosine = math.cos(anchor_pose.yaw)
    sine = math.sin(anchor_pose.yaw)
    return (
        anchor_pose.x
        + cosine * float(state_grid[0])
        - sine * float(state_grid[1]),
        anchor_pose.y
        + sine * float(state_grid[0])
        + cosine * float(state_grid[1]),
    )


def create_prediction_markers(
    stage: Usd.Stage,
    evaluations: tuple[ScenarioControlEvaluation, ...],
    *,
    anchor_pose: RobotPose2D,
) -> None:
    """Draw one non-colliding colored prediction path per scenario."""

    if stage.GetPrimAtPath(PREDICTION_ROOT_PRIM_PATH).IsValid():
        raise RuntimeError(
            "The CRA prediction marker root already exists: "
            f"{PREDICTION_ROOT_PRIM_PATH}"
        )
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        UsdGeom.Xform.Define(stage, PREDICTION_ROOT_PRIM_PATH)
        for scenario_index, evaluation in enumerate(evaluations):
            root_path = (
                f"{PREDICTION_ROOT_PRIM_PATH}/"
                f"{evaluation.scenario.key}"
            )
            UsdGeom.Xform.Define(stage, root_path)
            color = PREDICTION_COLORS[scenario_index]
            height = (
                PREDICTION_MARKER_BASE_HEIGHT
                + scenario_index * PREDICTION_MARKER_HEIGHT_STEP
            )
            for point_index, state in enumerate(
                evaluation.mpc_result.predicted_chassis_states
            ):
                sphere = UsdGeom.Sphere.Define(
                    stage,
                    f"{root_path}/Point_{point_index:02d}",
                )
                sphere.CreateRadiusAttr(PREDICTION_MARKER_RADIUS)
                world_x, world_y = field_state_to_world_xy(
                    state,
                    anchor_pose=anchor_pose,
                )
                sphere.AddTranslateOp().Set(
                    Gf.Vec3d(world_x, world_y, height)
                )
                sphere.CreateDisplayColorAttr([Gf.Vec3f(*color)])


def geometry_is_shared(
    evaluations: tuple[ScenarioControlEvaluation, ...],
) -> bool:
    """Return whether every field used identical configuration geometry."""

    reference = evaluations[0].field.pipeline.configuration_space
    for evaluation in evaluations[1:]:
        candidate = evaluation.field.pipeline.configuration_space
        if not np.array_equal(
            reference.yaw_angles_radians,
            candidate.yaw_angles_radians,
        ):
            return False
        for reference_grid, candidate_grid in zip(
            reference.occupancy_grids,
            candidate.occupancy_grids,
            strict=True,
        ):
            if not np.array_equal(
                reference_grid.occupied_mask,
                candidate_grid.occupied_mask,
            ):
                return False
            if not np.array_equal(
                reference_grid.cell_track_ids,
                candidate_grid.cell_track_ids,
            ):
                return False
    return True


def fields_are_distinct(
    evaluations: tuple[ScenarioControlEvaluation, ...],
) -> bool:
    """Return whether CRA changed the numerical ``h_full`` solutions."""

    values = tuple(
        evaluation.field.pipeline.h_full.values
        for evaluation in evaluations
    )
    return all(
        not np.allclose(left, right, atol=1e-10, rtol=0.0)
        for left, right in zip(values, values[1:])
    )


def predicted_trajectories_are_distinct(
    evaluations: tuple[ScenarioControlEvaluation, ...],
) -> bool:
    """Return whether CRA changed at least one predictive state plan.

    A safety filter is allowed to map multiple risk levels to the same active
    constraint or actuator-saturated command.  Requiring all four plans to be
    unique would incorrectly treat that valid many-to-one behavior as a
    failure.
    """

    trajectories = tuple(
        evaluation.mpc_result.predicted_chassis_states
        for evaluation in evaluations
    )
    return any(
        not np.allclose(trajectories[0], candidate, atol=1e-5, rtol=0.0)
        for candidate in trajectories[1:]
    )


def find_forward_probe_comparison(
    *,
    evaluations: tuple[ScenarioControlEvaluation, ...],
    control_runtime: SafeSAGEControlRuntime,
    nominal_input: np.ndarray,
) -> ForwardProbeComparison:
    """Find one common safe probe where CRA changes analytical filtering.

    The physical robot never visits these states.  The sweep is a controlled
    diagnostic because every profile receives the same chassis state and
    nominal command.  The maximum command spread is reported rather than
    assuming the safety constraint must already be active at the start pose.
    """

    best: ForwardProbeComparison | None = None
    for chassis_x in np.arange(
        0.0,
        stationary.FORWARD_PROBE_MAXIMUM_X
        + 0.5 * stationary.FORWARD_PROBE_SPACING,
        stationary.FORWARD_PROBE_SPACING,
    ):
        state = np.array([chassis_x, 0.0, 0.0], dtype=np.float64)
        results = tuple(
            control_runtime.filter_command(
                nominal_input,
                state,
                evaluation.field,
            )
            for evaluation in evaluations
        )
        if not all(
            result.constraint_feasible
            and result.constraint_satisfied
            and not result.emergency_stop_active
            and not result.query_on_unsafe_boundary
            for result in results
        ):
            continue
        linear = np.asarray(
            [result.safe_linear_velocity for result in results],
            dtype=np.float64,
        )
        angular = np.asarray(
            [result.safe_angular_velocity for result in results],
            dtype=np.float64,
        )
        spread = float(np.ptp(linear) + 0.1 * np.ptp(angular))
        candidate = ForwardProbeComparison(
            chassis_x=float(chassis_x),
            results=results,
            command_spread=spread,
        )
        if best is None or candidate.command_spread > best.command_spread:
            best = candidate
    if best is None:
        raise RuntimeError(
            "No common feasible forward-probe state was available."
        )
    return best


def print_report(
    *,
    scene_path: Path,
    frame_info,
    scan_wait_steps: int,
    all_obstacle_point_count: int,
    proxy_point_count: int,
    proxy_track_id: int,
    proxy_association_error: float,
    nominal_command,
    evaluations: tuple[ScenarioControlEvaluation, ...],
    forward_probe: ForwardProbeComparison,
    mpc_config: SafeSAGEUnicycleMPCConfig,
    initial_pose: RobotPose2D,
    final_pose: RobotPose2D,
) -> bool:
    """Print the controlled four-scenario acceptance report."""

    print("\n" + "=" * 96)
    print("CRA ORACLE SCENARIO VALIDATION - FIXED PERCEPTION AND CONTROL")
    print("=" * 96)
    print(f"  Scene:                       {scene_path}")
    print(f"  LiDAR frame ID:              {frame_info.sensor_frame_id}")
    print(f"  Waited scan updates:         {scan_wait_steps}")
    print(f"  All obstacle points:         {all_obstacle_point_count}")
    print(f"  Controlled proxy points:     {proxy_point_count}")
    print(f"  Source proxy track ID:       {proxy_track_id}")
    print(f"  Proxy association error:     {proxy_association_error:.4f} m")
    print(f"  Shared semantic placeholder: {SHARED_SEMANTIC_CLASS}")
    print(
        "  Shared nominal (v, omega):  "
        f"({nominal_command.linear_velocity:.6f}, "
        f"{nominal_command.angular_velocity:.6f})"
    )
    print(
        "  MPC horizon:                 "
        f"{mpc_config.horizon_steps} x {mpc_config.time_step:.3f} s"
    )
    print("\n  Oracle profile                  phi       b   lambda  dominant  "
          "MPC (v,omega)        right     h_min      margin    backup")
    print("  " + "-" * 92)
    for evaluation in evaluations:
        assignment = evaluation.policy_assignment.policy_result
        dominant = assignment.assessment.dominant_factor
        dominant_name = "none" if dominant is None else dominant.value
        mpc = evaluation.mpc_result
        backup = evaluation.backup_result
        print(
            f"  {evaluation.scenario.display_name:<29} "
            f"{assignment.risk:>6.4f} "
            f"{assignment.boundary_flux:>7.4f} "
            f"{assignment.tangential_strength:>7.3f}  "
            f"{dominant_name:<9} "
            f"({mpc.linear_velocity:>6.3f},{mpc.angular_velocity:>6.3f}) "
            f"{evaluation.maximum_rightward_displacement:>8.4f} "
            f"{mpc.minimum_predicted_safety_value:>10.5f} "
            f"{mpc.minimum_dtcbf_margin:>9.2e} "
            f"{'ok' if backup.constraint_satisfied else 'FAIL'}"
        )
        print(
            "      solver="
            f"{mpc.status}; success={mpc.success}; "
            f"fallback={mpc.fallback_active}; "
            f"wall={evaluation.mpc_wall_seconds:.3f} s"
        )

    print(
        "\nShared forward-probe counterfactual at chassis x="
        f"{forward_probe.chassis_x:.3f} m:"
    )
    for evaluation, result in zip(
        evaluations,
        forward_probe.results,
        strict=True,
    ):
        print(
            f"  {evaluation.scenario.display_name:<29} "
            f"safe=({result.safe_linear_velocity:.6f}, "
            f"{result.safe_angular_velocity:.6f}) "
            f"intervention={result.intervention_active} "
            f"margin={result.cbf_margin:.3e}"
        )

    same_geometry = geometry_is_shared(evaluations)
    distinct_fields = fields_are_distinct(evaluations)
    same_semantics = len(
        {
            evaluation.policy_assignment.obstacle.semantic_class
            for evaluation in evaluations
        }
    ) == 1
    risks = tuple(
        evaluation.policy_assignment.policy_result.risk
        for evaluation in evaluations
    )
    fluxes = tuple(
        abs(evaluation.policy_assignment.policy_result.boundary_flux)
        for evaluation in evaluations
    )
    lambdas = tuple(
        evaluation.policy_assignment.policy_result.tangential_strength
        for evaluation in evaluations
    )
    expected_dominants = all(
        evaluation.policy_assignment.policy_result.assessment.dominant_factor
        is evaluation.scenario.expected_dominant_factor
        for evaluation in evaluations
    )
    ordered_cra_response = (
        risks == tuple(sorted(risks))
        and fluxes == tuple(sorted(fluxes))
        and len(set(risks)) == len(risks)
    )
    lambda_ordered = (
        lambdas == tuple(sorted(lambdas))
        and len(set(lambdas)) == len(lambdas)
    )
    no_cra_fallback = all(
        evaluation.field.cra_fallback_track_ids == ()
        for evaluation in evaluations
    )
    predictive_valid = all(
        evaluation.mpc_result.success
        and not evaluation.mpc_result.fallback_active
        and evaluation.mpc_result.minimum_dtcbf_margin
        >= -mpc_config.constraint_tolerance
        for evaluation in evaluations
    )
    backup_valid = all(
        evaluation.backup_result.constraint_feasible
        and evaluation.backup_result.constraint_satisfied
        and not evaluation.backup_result.emergency_stop_active
        for evaluation in evaluations
    )
    initial_command_signatures = {
        (
            round(evaluation.mpc_result.linear_velocity, 6),
            round(evaluation.mpc_result.angular_velocity, 6),
        )
        for evaluation in evaluations
    }
    initial_mpc_response_differs = len(initial_command_signatures) > 1
    predictive_trajectories_differ = predicted_trajectories_are_distinct(
        evaluations
    )
    probe_signatures = {
        (
            round(result.safe_linear_velocity, 6),
            round(result.safe_angular_velocity, 6),
        )
        for result in forward_probe.results
    }
    probe_response_differs = (
        len(probe_signatures) > 1
        and forward_probe.command_spread > 1e-6
        and any(result.intervention_active for result in forward_probe.results)
    )
    position_drift = math.hypot(
        final_pose.x - initial_pose.x,
        final_pose.y - initial_pose.y,
    )
    yaw_drift = stationary.absolute_wrapped_angle_difference(
        final_pose.yaw,
        initial_pose.yaw,
    )
    robot_stationary = (
        position_drift <= stationary.MAXIMUM_STATIONARY_POSITION_DRIFT
        and yaw_drift <= stationary.MAXIMUM_STATIONARY_YAW_DRIFT_RADIANS
    )

    print("\nControlled checks:")
    print(f"  Same sensed/configuration geometry: {same_geometry}")
    print(f"  CRA changed numerical h_full:       {distinct_fields}")
    print(f"  Same placeholder semantic class:    {same_semantics}")
    print(f"  Expected dominant factors:          {expected_dominants}")
    print(f"  CRA phi and |b| ordered:            {ordered_cra_response}")
    print(f"  CRA lambda mapping is ordered:       {lambda_ordered}")
    print(f"  No missing-profile fallback:        {no_cra_fallback}")
    print(f"  All predictive constraints valid:   {predictive_valid}")
    print(f"  All analytical constraints valid:   {backup_valid}")
    print(
        "  Initial MPC responses are distinct:"
        f" {initial_mpc_response_differs} (diagnostic only)"
    )
    print(
        "  CRA changes a predicted trajectory:  "
        f" {predictive_trajectories_differ}"
    )
    print(
        "  Forward-probe CBF responses differ:"
        f" {probe_response_differs} (diagnostic only)"
    )
    print(f"  Robot remained stationary:          {robot_stationary}")
    print(f"  Measured position drift:             {position_drift:.6f} m")
    print(f"  Measured yaw drift:                  {math.degrees(yaw_drift):.4f} deg")

    passed = all(
        (
            same_geometry,
            distinct_fields,
            same_semantics,
            expected_dominants,
            ordered_cra_response,
            lambda_ordered,
            no_cra_fallback,
            predictive_valid,
            backup_valid,
            predictive_trajectories_differ,
            robot_stationary,
        )
    )
    print("\nInterpretation:")
    if passed:
        print(
            "  [PASS] One real LiDAR geometry produced distinct safe-control "
            "responses only after the controlled CRA hypothesis changed."
        )
        print(
            "  The mathematical/runtime/control path is ready for a learned "
            "vision-based CRA estimator."
        )
    else:
        print(
            "  [WARNING] At least one pre-vision acceptance check failed. "
            "Inspect the per-scenario solver status before using vision."
        )
    print(
        "  The four physical-factor values remain illustrative and "
        "uncalibrated."
    )
    print("  No MPC or CBF command reached the wheels.")
    print("=" * 96)
    return passed


def main() -> None:
    """Run the controlled four-profile CRA validation in Isaac Sim."""

    scene_path = resolve_scene_path()
    guidance_policy = SafeSAGEGuidancePolicy.from_json(
        stationary.GUIDANCE_POLICY_PATH
    )
    measured_footprint = RectangularRobotFootprint(
        front_extent=stationary.ROBOT_FRONT_EXTENT,
        rear_extent=stationary.ROBOT_REAR_EXTENT,
        left_extent=stationary.ROBOT_LEFT_EXTENT,
        right_extent=stationary.ROBOT_RIGHT_EXTENT,
        safety_margin=stationary.ROBOT_SAFETY_MARGIN,
    )
    footprint = shift_rectangular_footprint_to_lookahead(
        measured_footprint,
        lookahead_distance=stationary.UNICYCLE_LOOKAHEAD_DISTANCE,
    )
    field_config = stationary.create_field_pipeline_config(footprint)

    print("\n" + "=" * 96)
    print("CRA ORACLE SCENARIO VALIDATION DEMO")
    print("=" * 96)
    print("Mode: fixed real perception + four oracle CRA counterfactuals")
    print(f"Scene: {scene_path}")

    loaded_scene = load_robot_scene(
        scene_path=scene_path,
        simulation_app=simulation_app,
        robot_prim_path=stationary.ROBOT_PRIM_PATH,
        expected_meters_per_unit=stationary.EXPECTED_METERS_PER_UNIT,
    )
    stationary.create_static_proxy(loaded_scene.stage)
    robot_controller = NovaCarterController(
        robot_prim_path=stationary.ROBOT_PRIM_PATH
    )
    SimulationManager.setup_simulation(
        dt=stationary.PHYSICS_DT,
        device=stationary.PHYSICS_DEVICE,
    )
    print("\nStarting Isaac Sim playback automatically...")
    app_utils.play()
    if not wait_until_playing(simulation_app):
        return
    app_utils.update_app(steps=10)
    robot_controller.verify_articulation()
    stationary.position_robot_for_experiment(robot_controller)
    stationary.verify_robot_start_pose(loaded_scene.stage)
    robot_controller.print_configuration()

    lidar = NovaCarterLidar(
        lidar_prim_path=stationary.PANDAR_XT32_LIDAR_PRIM_PATH
    )
    lidar.initialize()
    robot_from_lidar = compute_relative_prim_transform(
        loaded_scene.stage,
        source_prim_path=stationary.PANDAR_XT32_LIDAR_PRIM_PATH,
        target_prim_path=stationary.ROBOT_BASE_PRIM_PATH,
    )
    floor_height = stationary.estimate_floor_height_in_robot_frame(
        loaded_scene.stage
    )
    detection_config = ObstacleDetectionConfig(
        minimum_height=(
            floor_height + stationary.MINIMUM_HEIGHT_ABOVE_FLOOR
        ),
        maximum_height=(
            floor_height + stationary.MAXIMUM_HEIGHT_ABOVE_FLOOR
        ),
        minimum_range=stationary.MINIMUM_DETECTION_RANGE,
        maximum_range=stationary.MAXIMUM_DETECTION_RANGE,
        forward_only=True,
        angular_bin_size_degrees=stationary.ANGULAR_BIN_SIZE_DEGREES,
    )

    print(
        f"\nHolding the robot and proxy still for "
        f"{stationary.SETTLE_STEPS} physics steps..."
    )
    if not advance_playing_steps(
        simulation_app,
        physics_steps=stationary.SETTLE_STEPS,
        on_step=robot_controller.stop,
    ):
        return
    robot_pose = stationary.read_robot_pose(loaded_scene.stage)
    points_sensor_frame, frame_info, scan_wait_steps = (
        stationary.wait_for_first_scan(lidar, robot_controller)
    )
    obstacle_points = stationary.extract_obstacle_points(
        points_sensor_frame,
        robot_from_lidar=robot_from_lidar,
        detection_config=detection_config,
    )
    source_tracker = ObstacleTracker(
        ObstacleTrackerConfig(
            cluster_radius=stationary.TRACKER_CLUSTER_RADIUS,
            minimum_cluster_points=stationary.TRACKER_MINIMUM_CLUSTER_POINTS,
            maximum_association_distance=(
                stationary.TRACKER_MAXIMUM_ASSOCIATION_DISTANCE
            ),
        )
    )
    source_tracking = source_tracker.update(
        obstacle_points,
        robot_position_world_xy=np.array(
            [robot_pose.x, robot_pose.y],
            dtype=np.float64,
        ),
        robot_yaw=robot_pose.yaw,
        timestamp_seconds=frame_info.sensor_timestamp_nanoseconds * 1e-9,
    )
    proxy_track, proxy_association_error = stationary.find_proxy_track(
        source_tracking,
        robot_pose=robot_pose,
    )
    if proxy_track is None:
        raise RuntimeError(
            "No LiDAR track matched the controlled proxy; nearest "
            f"association error was {proxy_association_error:.3f} m."
        )
    proxy_mask = source_tracking.point_track_ids == proxy_track.track_id
    proxy_points = np.ascontiguousarray(
        obstacle_points[proxy_mask],
        dtype=np.float64,
    )
    if proxy_points.shape[0] < stationary.TRACKER_MINIMUM_CLUSTER_POINTS:
        raise RuntimeError("The controlled proxy track has too few points.")

    observation = SafeSAGEPerceptionObservation(
        sensor_frame_id=frame_info.sensor_frame_id,
        sensor_timestamp_seconds=(
            frame_info.sensor_timestamp_nanoseconds * 1e-9
        ),
        simulation_time_seconds=(
            frame_info.sensor_timestamp_nanoseconds * 1e-9
        ),
        obstacle_points_robot_xy=proxy_points,
        robot_position_world_xy=np.array(
            [robot_pose.x, robot_pose.y],
            dtype=np.float64,
        ),
        robot_yaw=robot_pose.yaw,
    )
    nominal_command = stationary.create_waypoint_controller().compute_command(
        robot_pose
    )
    control_runtime, mpc_config = create_control_runtime(robot_controller)
    initial_state = np.zeros(3, dtype=np.float64)
    nominal_input = np.array(
        [
            nominal_command.linear_velocity,
            nominal_command.angular_velocity,
        ],
        dtype=np.float64,
    )
    shared_label = SemanticClassLabel(
        SHARED_SEMANTIC_CLASS,
        confidence=1.0,
        source="controlled_oracle",
    )

    evaluations: list[ScenarioControlEvaluation] = []
    print("\nBuilding four CRA fields and stationary controller predictions...")
    for scenario in controlled_cra_oracle_scenarios():
        runtime = create_perception_runtime(
            guidance_policy=guidance_policy,
            field_config=field_config,
        )
        # A fresh one-cluster tracker deterministically assigns track ID zero.
        status = runtime.initialize_with_cra(
            observation,
            TrackIdSemanticLabelProvider({0: shared_label}),
            TrackIdCRARiskProfileProvider({0: scenario.profile}),
            anchor_position_world_xy=np.array(
                [robot_pose.x, robot_pose.y],
                dtype=np.float64,
            ),
            anchor_yaw=robot_pose.yaw,
        )
        field = status.snapshot.payload
        if field.selected_track_ids != (0,):
            raise RuntimeError(
                "The controlled one-cluster runtime did not select track 0."
            )
        start = time.perf_counter()
        mpc_result = control_runtime.plan(
            initial_state,
            nominal_input,
            field,
            preferred_lateral_direction_grid_y=(
                PREFERRED_LATERAL_DIRECTION_GRID_Y
            ),
        )
        mpc_wall_seconds = time.perf_counter() - start
        backup_result = control_runtime.filter_command(
            np.array(mpc_result.first_command, dtype=np.float64),
            initial_state,
            field,
        )
        evaluations.append(
            ScenarioControlEvaluation(
                scenario=scenario,
                field=field,
                mpc_result=mpc_result,
                backup_result=backup_result,
                mpc_wall_seconds=mpc_wall_seconds,
            )
        )
        print(
            f"  completed {scenario.display_name:<28} "
            f"field={field.build_wall_seconds:.3f} s, "
            f"mpc={mpc_wall_seconds:.3f} s"
        )

    evaluation_tuple = tuple(evaluations)
    forward_probe = find_forward_probe_comparison(
        evaluations=evaluation_tuple,
        control_runtime=control_runtime,
        nominal_input=nominal_input,
    )
    create_prediction_markers(
        loaded_scene.stage,
        evaluation_tuple,
        anchor_pose=robot_pose,
    )
    robot_controller.stop()
    app_utils.update_app(steps=5)
    final_pose = stationary.read_robot_pose(loaded_scene.stage)
    passed = print_report(
        scene_path=scene_path,
        frame_info=frame_info,
        scan_wait_steps=scan_wait_steps,
        all_obstacle_point_count=obstacle_points.shape[0],
        proxy_point_count=proxy_points.shape[0],
        proxy_track_id=proxy_track.track_id,
        proxy_association_error=proxy_association_error,
        nominal_command=nominal_command,
        evaluations=evaluation_tuple,
        forward_probe=forward_probe,
        mpc_config=mpc_config,
        initial_pose=robot_pose,
        final_pose=final_pose,
    )

    app_utils.stop()
    print("\nExpected scene behavior:")
    print("  - Nova Carter and the blue proxy remain stationary.")
    print("  - Four colored MPC prediction paths begin from the robot.")
    print("  - Green: chair; cyan: person; orange: knife; magenta: hot soup.")
    print("  - The nominal waypoint itself remains straight through the proxy.")
    print("  - No predicted or filtered command reaches the wheel controller.")
    if stationary.SIMULATION_HEADLESS or AUTOCLOSE_AFTER_REPORT:
        print("\nAutomated run complete; closing Isaac Sim.")
    else:
        print("\nThe GUI will remain open for inspection.")
        print("Close the Isaac Sim window when finished.")
        while simulation_app.is_running():
            simulation_app.update()
    if not passed:
        raise RuntimeError("CRA oracle scenario validation did not pass.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001 - close SimulationApp on failure
        print("\n" + "=" * 96)
        print("ERROR DURING CRA ORACLE SCENARIO VALIDATION")
        print("=" * 96)
        print(error)
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()

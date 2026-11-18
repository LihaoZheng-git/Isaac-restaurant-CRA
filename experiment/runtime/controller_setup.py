"""Controller and field configuration used by the recorded service mission."""

from experiment.runtime import isaac_session as stationary
from robot.control.differential_drive import NovaCarterController
from robot.safety.class_safety_policy import ClassSafetyPolicy
from robot.safety.local_occupancy_grid import LocalOccupancyGridConfig
from robot.safety.safe_sage_cbf_filter import SafeSAGECBFConfig, SafeSAGECBFSafetyFilter
from robot.safety.safe_sage_configuration_space import RectangularRobotFootprint
from robot.safety.safe_sage_field_pipeline import SafeSAGEFieldPipelineConfig
from robot.safety.safe_sage_guidance_policy import SafeSAGEGuidancePolicy
from robot.safety.safe_sage_unicycle_cbf_filter import (
    SafeSAGEUnicycleCBFConfig,
    SafeSAGEUnicycleCBFFilter,
    shift_rectangular_footprint_to_lookahead,
)
from robot.safety.safe_sage_unicycle_mpc import (
    SafeSAGEUnicycleMPC,
    SafeSAGEUnicycleMPCConfig,
)

MPC_HORIZON_STEPS = 30
MPC_TIME_STEP = 0.20
MPC_LINEAR_VELOCITY_WEIGHT = 1.0
MPC_ANGULAR_VELOCITY_WEIGHT = 0.02
MPC_COMMAND_SMOOTHNESS_WEIGHT = 0.005
MPC_MAXIMUM_SQP_ITERATIONS = 12
MPC_MAXIMUM_SOLVER_ITERATIONS = 4000

MOTION_GRID_MINIMUM_X = -2.0
MOTION_GRID_MAXIMUM_X = 5.0


def create_controller_stack(
    robot_controller: NovaCarterController,
) -> tuple[SafeSAGEUnicycleMPC, SafeSAGEUnicycleCBFFilter]:
    """Create the shared paper filter, MPC, and analytical backup."""

    paper_filter = SafeSAGECBFSafetyFilter(
        SafeSAGECBFConfig(
            gamma=stationary.CBF_GAMMA,
            minimum_planar_velocity=(-robot_controller.config.max_linear_speed),
            maximum_planar_velocity=(robot_controller.config.max_linear_speed),
            minimum_angular_velocity=(-robot_controller.config.max_angular_speed),
            maximum_angular_velocity=(robot_controller.config.max_angular_speed),
            planar_velocity_weight=(stationary.CBF_LINEAR_VELOCITY_WEIGHT),
            angular_velocity_weight=(stationary.CBF_ANGULAR_VELOCITY_WEIGHT),
        )
    )
    mpc = SafeSAGEUnicycleMPC(
        paper_filter,
        SafeSAGEUnicycleMPCConfig(
            horizon_steps=MPC_HORIZON_STEPS,
            time_step=MPC_TIME_STEP,
            gamma=stationary.CBF_GAMMA,
            lookahead_distance=stationary.UNICYCLE_LOOKAHEAD_DISTANCE,
            minimum_linear_velocity=0.0,
            maximum_linear_velocity=(robot_controller.config.max_linear_speed),
            minimum_angular_velocity=(-robot_controller.config.max_angular_speed),
            maximum_angular_velocity=(robot_controller.config.max_angular_speed),
            linear_velocity_weight=(MPC_LINEAR_VELOCITY_WEIGHT),
            angular_velocity_weight=(MPC_ANGULAR_VELOCITY_WEIGHT),
            command_smoothness_weight=(MPC_COMMAND_SMOOTHNESS_WEIGHT),
            maximum_sqp_iterations=(MPC_MAXIMUM_SQP_ITERATIONS),
            maximum_solver_iterations=(MPC_MAXIMUM_SOLVER_ITERATIONS),
        ),
    )
    backup_filter = SafeSAGEUnicycleCBFFilter(
        paper_filter,
        SafeSAGEUnicycleCBFConfig(
            lookahead_distance=stationary.UNICYCLE_LOOKAHEAD_DISTANCE,
            minimum_linear_velocity=0.0,
            maximum_linear_velocity=(robot_controller.config.max_linear_speed),
            minimum_angular_velocity=(-robot_controller.config.max_angular_speed),
            maximum_angular_velocity=(robot_controller.config.max_angular_speed),
            linear_velocity_weight=(stationary.CBF_LINEAR_VELOCITY_WEIGHT),
            angular_velocity_weight=(stationary.CBF_ANGULAR_VELOCITY_WEIGHT),
        ),
    )
    return mpc, backup_filter


def create_fixed_field_configuration() -> tuple[
    ClassSafetyPolicy,
    SafeSAGEGuidancePolicy,
    LocalOccupancyGridConfig,
    SafeSAGEFieldPipelineConfig,
]:
    """Create the policies, measured footprint, grid, and PDE settings."""

    guidance_policy = SafeSAGEGuidancePolicy.from_json(stationary.GUIDANCE_POLICY_PATH)
    class_policy = stationary.class_policy_from_guidance_policy(guidance_policy)
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
    grid_config = LocalOccupancyGridConfig(
        resolution=stationary.GRID_RESOLUTION,
        minimum_x=MOTION_GRID_MINIMUM_X,
        maximum_x=MOTION_GRID_MAXIMUM_X,
        minimum_y=stationary.GRID_MINIMUM_Y,
        maximum_y=stationary.GRID_MAXIMUM_Y,
        obstacle_inflation_radius=0.0,
    )
    return (
        class_policy,
        guidance_policy,
        grid_config,
        stationary.create_field_pipeline_config(footprint),
    )

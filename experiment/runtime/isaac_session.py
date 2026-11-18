"""Isaac session, Carter geometry, and simulation helpers for the service mission."""

import math
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from robot.config import (
    PANDAR_XT32_LIDAR_PRIM_PATH as PANDAR_XT32_LIDAR_PRIM_PATH,
    PHYSICS_DEVICE as PHYSICS_DEVICE,
    PHYSICS_DT as PHYSICS_DT,
    ROBOT_BASE_PRIM_PATH,
    ROBOT_PRIM_PATH as ROBOT_PRIM_PATH,
    ROBOT_SCENE_PATH as ROBOT_SCENE_PATH,
)
from isaacsim import SimulationApp

RTX_HISTORY_TRANSFORM_COUNT = int(
    os.environ.get("ISAAC_RTX_HISTORY_TRANSFORM_COUNT", "20")
)

if not 2 <= RTX_HISTORY_TRANSFORM_COUNT <= 256:
    raise ValueError("RTX history transform count must lie within [2, 256]")

SIMULATION_HEADLESS = os.environ.get(
    "ISAAC_SIM_HEADLESS",
    "0",
).strip().casefold() in {"1", "true", "yes", "on"}
USE_D3D12 = os.environ.get("ISAAC_SIM_USE_D3D12") == "1"


simulation_app = SimulationApp(
    {
        "headless": SIMULATION_HEADLESS,
        "enable_motion_bvh": True,
        "extra_args": [
            *(["--/app/vulkan=false"] if USE_D3D12 else []),
            *(
                ["--enable", "omni.anim.behavior.core"]
                if os.environ.get("CRA_NATIVE_HUMAN") == "1"
                else []
            ),
            (f"--/rtx/scenedb/maxHistoryTransformCount={RTX_HISTORY_TRANSFORM_COUNT}"),
        ],
    }
)

import isaacsim.core.experimental.utils.app as app_utils
from pxr import Usd
from robot.control.differential_drive import NovaCarterController
from robot.navigation.waypoint_controller import RobotPose2D
from robot.safety.class_safety_policy import ClassSafetyPolicy
from robot.safety.poisson_safety_function import PoissonSafetyFunctionConfig
from robot.safety.safe_sage_configuration_space import (
    RectangularRobotFootprint,
    SafeSAGEConfigurationSpaceConfig,
)
from robot.safety.safe_sage_field_pipeline import SafeSAGEFieldPipelineConfig
from robot.safety.safe_sage_guidance_policy import SafeSAGEGuidancePolicy
from robot.safety.safe_sage_h_full import SafeSAGEHFullConfig
from robot.safety.tangential_laplace_guidance_field import (
    TangentialLaplaceGuidanceFieldConfig,
)
from robot.scene.transforms import get_prim_planar_pose

GUIDANCE_POLICY_PATH = (
    PROJECT_ROOT / "experiment" / "configs" / "safe_sage_class_safety.json"
)
ROBOT_START_POSITION_WORLD = (
    26.4955,
    5.00175,
    0.07382,
)
ROBOT_START_YAW_RADIANS = 0.0
SETTLE_STEPS = 60
GRID_RESOLUTION = 0.20
GRID_MINIMUM_Y = -3.0
GRID_MAXIMUM_Y = 3.0
ROBOT_FRONT_EXTENT = 0.140000
ROBOT_REAR_EXTENT = 0.587351
ROBOT_LEFT_EXTENT = 0.246600
ROBOT_RIGHT_EXTENT = 0.246600
ROBOT_SAFETY_MARGIN = 0.050000
YAW_SLICE_COUNT = 10
LAPLACE_CONVERGENCE_TOLERANCE = 1e-6
TANGENTIAL_BUFFER_RADIUS = 0.60
H_FULL_SOFTPLUS_BETA = 1.0
H_FULL_MINIMUM_FORCING_MAGNITUDE = 1e-9
H_FULL_POISSON_CONVERGENCE_TOLERANCE = 1e-6
CBF_GAMMA = 1.0
CBF_LINEAR_VELOCITY_WEIGHT = 1.0
CBF_ANGULAR_VELOCITY_WEIGHT = 1.0
UNICYCLE_LOOKAHEAD_DISTANCE = 0.10


def yaw_to_quaternion_wxyz(yaw_radians: float) -> list[float]:
    """Return a scalar-first quaternion for a world-Z yaw rotation."""

    half_yaw = 0.5 * yaw_radians
    return [
        math.cos(half_yaw),
        0.0,
        0.0,
        math.sin(half_yaw),
    ]


def position_robot_for_experiment(
    robot_controller: NovaCarterController,
) -> None:
    """Teleport the initialized articulation to the controlled start pose."""

    robot_controller.robot.set_world_poses(
        positions=[ROBOT_START_POSITION_WORLD],
        orientations=[yaw_to_quaternion_wxyz(ROBOT_START_YAW_RADIANS)],
    )
    robot_controller.stop()
    app_utils.update_app(steps=5)


def read_robot_pose(stage: Usd.Stage) -> RobotPose2D:
    """Read the chassis planar pose from the stage."""

    x_position, y_position, yaw = get_prim_planar_pose(
        stage,
        ROBOT_BASE_PRIM_PATH,
    )
    return RobotPose2D(
        x=x_position,
        y=y_position,
        yaw=yaw,
    )


def class_policy_from_guidance_policy(
    guidance_policy: SafeSAGEGuidancePolicy,
) -> ClassSafetyPolicy:
    """Expose the guidance policy's shared risk and normal-flux values."""

    return ClassSafetyPolicy(
        {
            semantic_class: parameters.safety_parameters
            for semantic_class, parameters in guidance_policy.class_parameters.items()
        },
        unknown_class=guidance_policy.unknown_class,
        risk_flux_mapping=guidance_policy.risk_flux_mapping,
    )


def create_field_pipeline_config(
    footprint: RectangularRobotFootprint,
) -> SafeSAGEFieldPipelineConfig:
    """Return the shared configuration-space and PDE configuration."""

    return SafeSAGEFieldPipelineConfig(
        configuration_space_config=(
            SafeSAGEConfigurationSpaceConfig(
                footprint=footprint,
                yaw_slice_count=YAW_SLICE_COUNT,
            )
        ),
        guidance_field_config=(
            TangentialLaplaceGuidanceFieldConfig(
                convergence_tolerance=(LAPLACE_CONVERGENCE_TOLERANCE),
                tangential_buffer_radius=TANGENTIAL_BUFFER_RADIUS,
            )
        ),
        h_full_config=SafeSAGEHFullConfig(
            yaw_slice_count=YAW_SLICE_COUNT,
            softplus_beta=H_FULL_SOFTPLUS_BETA,
            minimum_forcing_magnitude=(H_FULL_MINIMUM_FORCING_MAGNITUDE),
            poisson_config=PoissonSafetyFunctionConfig(
                convergence_tolerance=(H_FULL_POISSON_CONVERGENCE_TOLERANCE),
            ),
        ),
    )

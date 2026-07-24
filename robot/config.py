"""Shared project, robot, sensor, and simulation configuration."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GENERATED_SCENE_DIRECTORY = (
    PROJECT_ROOT / "scenes" / "generated"
)

ROBOT_SCENE_PATH = (
    GENERATED_SCENE_DIRECTORY / "restaurant_robot.usda"
)

ROBOT_PRIM_PATH = "/World/Robot"

LEFT_WHEEL_DOF = "joint_wheel_left"
RIGHT_WHEEL_DOF = "joint_wheel_right"

WHEEL_RADIUS_METERS = 0.14
WHEEL_BASE_METERS = 0.4132

ROBOT_BASE_PRIM_PATH = (
    f"{ROBOT_PRIM_PATH}/chassis_link"
)

PANDAR_XT32_LIDAR_PRIM_PATH = (
    f"{ROBOT_BASE_PRIM_PATH}/sensors/"
    "XT_32/PandarXT_32_10hz"
)

FRONT_HAWK_LEFT_CAMERA_PRIM_PATH = (
    f"{ROBOT_BASE_PRIM_PATH}/sensors/"
    "front_hawk/left/camera_left"
)

PHYSICS_DT = 1.0 / 60.0
PHYSICS_DEVICE = "cpu"

EXPECTED_METERS_PER_UNIT = 1.0
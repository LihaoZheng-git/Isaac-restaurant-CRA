"""Reusable differential-drive controller for Nova Carter.

Import this module only after ``SimulationApp`` has been created.
"""

from dataclasses import dataclass
from typing import Sequence

from isaacsim.robot.experimental.wheeled_robots.controllers import (
    DifferentialController,
)
from isaacsim.robot.experimental.wheeled_robots.robots import (
    WheeledRobot,
)

from robot.config import (
    LEFT_WHEEL_DOF,
    RIGHT_WHEEL_DOF,
    ROBOT_PRIM_PATH,
    WHEEL_BASE_METERS,
    WHEEL_RADIUS_METERS,
)



@dataclass(frozen=True)
class NovaCarterControllerConfig:
    """Geometry and command limits for the differential controller."""

    wheel_radius: float = WHEEL_RADIUS_METERS
    wheel_base: float = WHEEL_BASE_METERS
    max_linear_speed: float = 0.25
    max_angular_speed: float = 0.80
    max_wheel_speed: float = 10.0

    def __post_init__(self) -> None:
        positive_values = {
            "wheel_radius": self.wheel_radius,
            "wheel_base": self.wheel_base,
            "max_linear_speed": self.max_linear_speed,
            "max_angular_speed": self.max_angular_speed,
            "max_wheel_speed": self.max_wheel_speed,
        }

        for name, value in positive_values.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be greater than zero.")


class NovaCarterController:
    """Wrap Nova Carter's articulation and differential controller."""

    def __init__(
        self,
        *,
        robot_prim_path: str = ROBOT_PRIM_PATH,
        config: NovaCarterControllerConfig | None = None,
    ) -> None:
        self.robot_prim_path = robot_prim_path
        self.config = config or NovaCarterControllerConfig()

        self.robot = WheeledRobot(
            paths=self.robot_prim_path,
            wheel_dof_names=[
                LEFT_WHEEL_DOF,
                RIGHT_WHEEL_DOF,
            ],
        )

        self._controller = DifferentialController(
            wheel_radius=self.config.wheel_radius,
            wheel_base=self.config.wheel_base,
            max_linear_speed=self.config.max_linear_speed,
            max_angular_speed=self.config.max_angular_speed,
            max_wheel_speed=self.config.max_wheel_speed,
        )

    def verify_articulation(self) -> None:
        """Verify the articulation after physics has initialized."""

        dof_names = list(self.robot.dof_names)

        print("\n" + "=" * 70)
        print("ROBOT ARTICULATION REPORT")
        print("=" * 70)
        print(f"Robot path: {self.robot_prim_path}")
        print(f"Number of DOFs: {self.robot.num_dofs}")
        print("DOF names:")

        for index, dof_name in enumerate(dof_names):
            print(f"  [{index}] {dof_name}")

        required_dofs = [LEFT_WHEEL_DOF, RIGHT_WHEEL_DOF]
        missing_dofs = [
            dof_name
            for dof_name in required_dofs
            if dof_name not in dof_names
        ]

        if missing_dofs:
            raise RuntimeError(
                "Nova Carter is missing the expected wheel DOFs: "
                f"{missing_dofs}"
            )

        print("\n[PASS] Both drive-wheel joints were found.")
        print(
            "[PASS] Wheel command order is "
            "[joint_wheel_left, joint_wheel_right]."
        )

    def wheel_velocities(
        self,
        command: Sequence[float],
    ):
        """Convert ``[linear m/s, angular rad/s]`` to wheel speeds."""

        if len(command) != 2:
            raise ValueError(
                "A drive command must contain exactly "
                "[linear_velocity, angular_velocity]."
            )

        linear_velocity = float(command[0])
        angular_velocity = float(command[1])

        return self._controller.forward(
            [linear_velocity, angular_velocity]
        )

    def apply_command(
        self,
        linear_velocity: float,
        angular_velocity: float,
    ):
        """Calculate and apply one differential-drive command."""

        velocities = self.wheel_velocities(
            [linear_velocity, angular_velocity]
        )
        self.robot.apply_wheel_actions(velocities)
        return velocities

    def apply_wheel_velocities(self, velocities) -> None:
        """Apply a previously calculated pair of wheel velocities."""

        self.robot.apply_wheel_actions(velocities)

    def stop(self):
        """Apply a zero-velocity command to both drive wheels."""

        return self.apply_command(0.0, 0.0)

    def print_configuration(self) -> None:
        """Print the active geometry and command limits."""

        print("\nNova Carter differential controller:")
        print(
            f"  Wheel radius:       {self.config.wheel_radius} m"
        )
        print(
            f"  Wheel base:         {self.config.wheel_base} m"
        )
        print(
            "  Max linear speed:   "
            f"{self.config.max_linear_speed} m/s"
        )
        print(
            "  Max angular speed:  "
            f"{self.config.max_angular_speed} rad/s"
        )
        print(
            "  Max wheel speed:    "
            f"{self.config.max_wheel_speed} rad/s"
        )
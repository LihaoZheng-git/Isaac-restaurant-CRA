"""Pure waypoint controller for nominal restaurant navigation.

This module contains no Isaac Sim or safety-filter dependencies. It converts a
robot pose and a sequence of world-frame waypoints into nominal linear and
angular velocity commands.

The resulting command may later be passed through a CBF safety filter before it
is applied to the robot.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from robot.control.unicycle_model import integrate_unicycle_euler


@dataclass(frozen=True)
class Waypoint:
    """A two-dimensional waypoint expressed in the world frame."""

    x: float
    y: float

    def __post_init__(self) -> None:
        x_position = float(self.x)
        y_position = float(self.y)

        if not math.isfinite(x_position):
            raise ValueError("Waypoint x must be finite.")

        if not math.isfinite(y_position):
            raise ValueError("Waypoint y must be finite.")

        object.__setattr__(self, "x", x_position)
        object.__setattr__(self, "y", y_position)


@dataclass(frozen=True)
class RobotPose2D:
    """Planar robot pose expressed in the world frame."""

    x: float
    y: float
    yaw: float

    def __post_init__(self) -> None:
        values = {
            "x": float(self.x),
            "y": float(self.y),
            "yaw": float(self.yaw),
        }

        for name, value in values.items():
            if not math.isfinite(value):
                raise ValueError(
                    f"Robot pose {name} must be finite."
                )

        object.__setattr__(self, "x", values["x"])
        object.__setattr__(self, "y", values["y"])
        object.__setattr__(self, "yaw", values["yaw"])


@dataclass(frozen=True)
class WaypointControllerConfig:
    """Configuration for nominal waypoint tracking."""

    maximum_linear_velocity: float = 0.20
    maximum_angular_velocity: float = 0.60

    linear_gain: float = 0.80
    angular_gain: float = 1.50

    position_tolerance: float = 0.10
    minimum_alignment_scale: float = 0.0
    rotate_in_place_threshold_radians: float = (
        math.radians(25.0)
    )

    def __post_init__(self) -> None:
        positive_values = {
            "maximum_linear_velocity": (
                self.maximum_linear_velocity
            ),
            "maximum_angular_velocity": (
                self.maximum_angular_velocity
            ),
            "linear_gain": self.linear_gain,
            "angular_gain": self.angular_gain,
            "position_tolerance": self.position_tolerance,
            "rotate_in_place_threshold_radians": (
                self.rotate_in_place_threshold_radians
            ),
        }

        for name, value in positive_values.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")

            if value <= 0.0:
                raise ValueError(
                    f"{name} must be greater than zero."
                )

        if self.rotate_in_place_threshold_radians > math.pi:
            raise ValueError(
                "rotate_in_place_threshold_radians cannot "
                "exceed pi."
            )
        if not math.isfinite(self.minimum_alignment_scale):
            raise ValueError("minimum_alignment_scale must be finite.")
        if not 0.0 <= self.minimum_alignment_scale <= 1.0:
            raise ValueError(
                "minimum_alignment_scale must lie in [0, 1]."
            )


class WaypointMode(str, Enum):
    """Current behavior selected by the waypoint controller."""

    ROTATING = "rotating"
    DRIVING = "driving"
    COMPLETED = "completed"


@dataclass(frozen=True)
class WaypointCommand:
    """Complete result from one waypoint-controller update."""

    linear_velocity: float
    angular_velocity: float

    mode: WaypointMode

    target_waypoint_index: int | None
    target_waypoint: Waypoint | None

    distance_to_waypoint: float
    heading_error: float

    waypoint_advanced: bool
    route_complete: bool


def wrap_to_pi(angle_radians: float) -> float:
    """Wrap an angle to the interval [-pi, pi)."""

    angle = float(angle_radians)

    if not math.isfinite(angle):
        raise ValueError("angle_radians must be finite.")

    return (
        (angle + math.pi)
        % (2.0 * math.pi)
        - math.pi
    )


class WaypointController:
    """Generate nominal commands for a sequence of waypoints."""

    def __init__(
        self,
        waypoints: Sequence[Waypoint],
        *,
        config: WaypointControllerConfig | None = None,
    ) -> None:
        self.config = config or WaypointControllerConfig()

        self._waypoints = tuple(waypoints)

        if not self._waypoints:
            raise ValueError(
                "At least one waypoint is required."
            )

        if not all(
            isinstance(waypoint, Waypoint)
            for waypoint in self._waypoints
        ):
            raise TypeError(
                "Every route entry must be a Waypoint."
            )

        self._current_waypoint_index = 0

    @property
    def waypoints(self) -> tuple[Waypoint, ...]:
        """Return the immutable waypoint route."""

        return self._waypoints

    @property
    def current_waypoint_index(self) -> int | None:
        """Return the active waypoint index, or None when complete."""

        if self.route_complete:
            return None

        return self._current_waypoint_index

    @property
    def current_waypoint(self) -> Waypoint | None:
        """Return the active waypoint, or None when complete."""

        if self.route_complete:
            return None

        return self._waypoints[
            self._current_waypoint_index
        ]

    @property
    def route_complete(self) -> bool:
        """Whether every waypoint has been reached."""

        return (
            self._current_waypoint_index
            >= len(self._waypoints)
        )

    def reset(self) -> None:
        """Restart navigation from the first waypoint."""

        self._current_waypoint_index = 0

    def compute_command(
        self,
        pose: RobotPose2D,
    ) -> WaypointCommand:
        """Return the nominal command for the current robot pose."""

        waypoint_advanced = (
            self._advance_reached_waypoints(pose)
        )

        if self.route_complete:
            return WaypointCommand(
                linear_velocity=0.0,
                angular_velocity=0.0,
                mode=WaypointMode.COMPLETED,
                target_waypoint_index=None,
                target_waypoint=None,
                distance_to_waypoint=0.0,
                heading_error=0.0,
                waypoint_advanced=waypoint_advanced,
                route_complete=True,
            )

        target = self._waypoints[
            self._current_waypoint_index
        ]

        x_error = target.x - pose.x
        y_error = target.y - pose.y

        distance = math.hypot(
            x_error,
            y_error,
        )

        desired_heading = math.atan2(
            y_error,
            x_error,
        )

        heading_error = wrap_to_pi(
            desired_heading - pose.yaw
        )

        if (
            abs(heading_error)
            > self.config.rotate_in_place_threshold_radians
        ):
            linear_velocity = 0.0
            angular_velocity = self._clip_angular_velocity(
                self.config.angular_gain
                * heading_error
            )
            mode = WaypointMode.ROTATING

        else:
            unconstrained_linear_velocity = (
                self.config.linear_gain
                * distance
            )

            alignment_scale = max(
                self.config.minimum_alignment_scale,
                math.cos(heading_error),
            )

            linear_velocity = min(
                self.config.maximum_linear_velocity,
                unconstrained_linear_velocity,
            ) * alignment_scale

            angular_velocity = self._clip_angular_velocity(
                self.config.angular_gain
                * heading_error
            )

            mode = WaypointMode.DRIVING

        return WaypointCommand(
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
            mode=mode,
            target_waypoint_index=(
                self._current_waypoint_index
            ),
            target_waypoint=target,
            distance_to_waypoint=distance,
            heading_error=heading_error,
            waypoint_advanced=waypoint_advanced,
            route_complete=False,
        )

    def predict_commands(
        self,
        pose: RobotPose2D,
        *,
        horizon_steps: int,
        time_step: float,
    ) -> tuple[WaypointCommand, ...]:
        """Roll out closed-loop nominal commands without changing this controller.

        Recomputing the waypoint command at every predicted unicycle state
        captures transitions between rotate-in-place and forward-driving modes.
        This is the nominal command sequence required by predictive filters.
        """

        if isinstance(horizon_steps, bool) or not isinstance(horizon_steps, int):
            raise TypeError("horizon_steps must be an integer.")
        if horizon_steps <= 0:
            raise ValueError("horizon_steps must be greater than zero.")
        prediction_time_step = float(time_step)
        if not math.isfinite(prediction_time_step):
            raise ValueError("time_step must be finite.")
        if prediction_time_step <= 0.0:
            raise ValueError("time_step must be greater than zero.")

        predictor = WaypointController(self._waypoints, config=self.config)
        predictor._current_waypoint_index = self._current_waypoint_index
        predicted_pose = pose
        commands: list[WaypointCommand] = []
        for _ in range(horizon_steps):
            command = predictor.compute_command(predicted_pose)
            commands.append(command)
            next_state = integrate_unicycle_euler(
                (predicted_pose.x, predicted_pose.y, predicted_pose.yaw),
                (command.linear_velocity, command.angular_velocity),
                time_step=prediction_time_step,
            )
            predicted_pose = RobotPose2D(
                x=float(next_state[0]),
                y=float(next_state[1]),
                yaw=float(next_state[2]),
            )
        return tuple(commands)

    def _advance_reached_waypoints(
        self,
        pose: RobotPose2D,
    ) -> bool:
        """Advance past all waypoints already within tolerance."""

        waypoint_advanced = False

        while not self.route_complete:
            target = self._waypoints[
                self._current_waypoint_index
            ]

            distance = math.hypot(
                target.x - pose.x,
                target.y - pose.y,
            )

            if (
                distance
                > self.config.position_tolerance
            ):
                break

            self._current_waypoint_index += 1
            waypoint_advanced = True

        return waypoint_advanced

    def _clip_angular_velocity(
        self,
        angular_velocity: float,
    ) -> float:
        """Clamp an angular command to its configured limits."""

        limit = self.config.maximum_angular_velocity

        return max(
            -limit,
            min(
                limit,
                angular_velocity,
            ),
        )

"""Pure NumPy Control Barrier Function safety filter.

This module modifies linear velocity to maintain a circular safety region
around the robot. Obstacle points must be expressed in the robot base frame:

    +X forward
    +Y left

The first version preserves angular velocity. It does not choose an
obstacle-avoidance direction.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class CBFConfig:
    """Configuration for the distance-based CBF safety filter."""

    safety_radius: float
    alpha: float = 1.0
    minimum_linear_velocity: float = 0.0
    maximum_linear_velocity: float = 0.25
    numerical_tolerance: float = 1e-9
    intervention_tolerance: float = 1e-6

    def __post_init__(self) -> None:
        finite_values = {
            "safety_radius": self.safety_radius,
            "alpha": self.alpha,
            "minimum_linear_velocity": self.minimum_linear_velocity,
            "maximum_linear_velocity": self.maximum_linear_velocity,
            "numerical_tolerance": self.numerical_tolerance,
            "intervention_tolerance": self.intervention_tolerance,
        }

        for name, value in finite_values.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")

        if self.safety_radius <= 0.0:
            raise ValueError("safety_radius must be greater than zero.")

        if self.alpha <= 0.0:
            raise ValueError("alpha must be greater than zero.")

        if (
            self.minimum_linear_velocity
            > self.maximum_linear_velocity
        ):
            raise ValueError(
                "minimum_linear_velocity cannot be greater than "
                "maximum_linear_velocity."
            )

        if not (
            self.minimum_linear_velocity
            <= 0.0
            <= self.maximum_linear_velocity
        ):
            raise ValueError(
                "The linear-velocity limits must contain zero so the "
                "filter can issue an emergency stop."
            )

        if self.numerical_tolerance <= 0.0:
            raise ValueError(
                "numerical_tolerance must be greater than zero."
            )

        if self.intervention_tolerance < 0.0:
            raise ValueError(
                "intervention_tolerance cannot be negative."
            )


@dataclass(frozen=True)
class CBFResult:
    """Complete result returned by the CBF safety filter."""

    nominal_linear_velocity: float
    nominal_angular_velocity: float

    safe_linear_velocity: float
    safe_angular_velocity: float

    minimum_obstacle_distance: float
    minimum_barrier_value: float

    admissible_linear_velocity_minimum: float
    admissible_linear_velocity_maximum: float

    obstacle_count: int
    constraint_feasible: bool
    intervention_active: bool
    emergency_stop_active: bool


class CBFSafetyFilter:
    """Project a nominal command onto the CBF-safe velocity interval."""

    def __init__(self, config: CBFConfig) -> None:
        self.config = config

    def filter_command(
        self,
        nominal_linear_velocity: float,
        nominal_angular_velocity: float,
        obstacle_points_xy: np.ndarray,
    ) -> CBFResult:
        """Return the closest admissible linear-velocity command.

        For every obstacle point ``(x, y)``, the barrier function is:

            h = x**2 + y**2 - safety_radius**2

        Assuming a stationary obstacle and translational robot velocity,
        the CBF condition is:

            -2 * x * v + alpha * h >= 0

        This produces an upper velocity bound for obstacles ahead and a
        lower velocity bound for obstacles behind. Projecting the nominal
        velocity onto the resulting interval is equivalent to solving the
        corresponding one-dimensional quadratic program.
        """

        nominal_linear = self._validate_command_value(
            nominal_linear_velocity,
            name="nominal_linear_velocity",
        )
        nominal_angular = self._validate_command_value(
            nominal_angular_velocity,
            name="nominal_angular_velocity",
        )

        points = self._validate_obstacle_points(
            obstacle_points_xy
        )

        obstacle_count = int(points.shape[0])

        if obstacle_count == 0:
            return self._result_without_obstacles(
                nominal_linear,
                nominal_angular,
            )

        distances = np.linalg.norm(points, axis=1)

        barrier_values = (
            np.square(distances)
            - self.config.safety_radius**2
        )

        minimum_obstacle_distance = float(np.min(distances))
        minimum_barrier_value = float(
            np.min(barrier_values)
        )

        lower_bound = self.config.minimum_linear_velocity
        upper_bound = self.config.maximum_linear_velocity

        lateral_constraint_infeasible = False
        tolerance = self.config.numerical_tolerance

        for point, barrier_value in zip(
            points,
            barrier_values,
            strict=True,
        ):
            x_position = float(point[0])
            barrier = float(barrier_value)

            if x_position > tolerance:
                velocity_limit = (
                    self.config.alpha
                    * barrier
                    / (2.0 * x_position)
                )
                upper_bound = min(
                    upper_bound,
                    velocity_limit,
                )

            elif x_position < -tolerance:
                velocity_limit = (
                    self.config.alpha
                    * barrier
                    / (2.0 * x_position)
                )
                lower_bound = max(
                    lower_bound,
                    velocity_limit,
                )

            elif barrier < -tolerance:
                # Pure forward/reverse motion cannot immediately increase
                # the distance to an obstacle exactly beside the robot.
                lateral_constraint_infeasible = True

        constraint_feasible = (
            not lateral_constraint_infeasible
            and lower_bound <= upper_bound + tolerance
        )

        if constraint_feasible:
            if lower_bound > upper_bound:
                # Collapse a tiny numerical disagreement to one value.
                boundary = 0.5 * (
                    lower_bound + upper_bound
                )
                lower_bound = boundary
                upper_bound = boundary

            safe_linear = float(
                np.clip(
                    nominal_linear,
                    lower_bound,
                    upper_bound,
                )
            )
            emergency_stop_active = False

        else:
            # Zero is guaranteed to be within the configured actuator
            # limits. Stopping is the safest available fallback when the
            # constraints cannot be satisfied by this simple model.
            safe_linear = 0.0
            emergency_stop_active = True

        intervention_active = (
            abs(safe_linear - nominal_linear)
            > self.config.intervention_tolerance
        )

        return CBFResult(
            nominal_linear_velocity=nominal_linear,
            nominal_angular_velocity=nominal_angular,
            safe_linear_velocity=safe_linear,
            safe_angular_velocity=nominal_angular,
            minimum_obstacle_distance=minimum_obstacle_distance,
            minimum_barrier_value=minimum_barrier_value,
            admissible_linear_velocity_minimum=float(
                lower_bound
            ),
            admissible_linear_velocity_maximum=float(
                upper_bound
            ),
            obstacle_count=obstacle_count,
            constraint_feasible=constraint_feasible,
            intervention_active=intervention_active,
            emergency_stop_active=emergency_stop_active,
        )

    def _result_without_obstacles(
        self,
        nominal_linear_velocity: float,
        nominal_angular_velocity: float,
    ) -> CBFResult:
        """Apply actuator limits when no obstacle constraints exist."""

        safe_linear = float(
            np.clip(
                nominal_linear_velocity,
                self.config.minimum_linear_velocity,
                self.config.maximum_linear_velocity,
            )
        )

        intervention_active = (
            abs(safe_linear - nominal_linear_velocity)
            > self.config.intervention_tolerance
        )

        return CBFResult(
            nominal_linear_velocity=nominal_linear_velocity,
            nominal_angular_velocity=nominal_angular_velocity,
            safe_linear_velocity=safe_linear,
            safe_angular_velocity=nominal_angular_velocity,
            minimum_obstacle_distance=math.inf,
            minimum_barrier_value=math.inf,
            admissible_linear_velocity_minimum=(
                self.config.minimum_linear_velocity
            ),
            admissible_linear_velocity_maximum=(
                self.config.maximum_linear_velocity
            ),
            obstacle_count=0,
            constraint_feasible=True,
            intervention_active=intervention_active,
            emergency_stop_active=False,
        )

    @staticmethod
    def _validate_command_value(
        value: float,
        *,
        name: str,
    ) -> float:
        """Convert and validate one scalar command value."""

        converted_value = float(value)

        if not math.isfinite(converted_value):
            raise ValueError(f"{name} must be finite.")

        return converted_value

    @staticmethod
    def _validate_obstacle_points(
        obstacle_points_xy: np.ndarray,
    ) -> np.ndarray:
        """Return obstacle points as a validated float64 array."""

        points = np.asarray(
            obstacle_points_xy,
            dtype=np.float64,
        )

        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(
                "obstacle_points_xy must have shape (N, 2), "
                f"received {points.shape}."
            )

        if not np.isfinite(points).all():
            raise ValueError(
                "obstacle_points_xy must contain only finite values."
            )

        return points
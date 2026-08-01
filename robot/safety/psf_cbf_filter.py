"""Poisson-safety-function CBF filter for differential-drive commands.

The filter samples the scalar Poisson safety function ``h`` and semantic
Laplace guidance vector ``g`` at a look-ahead point fixed to the robot. For a
differential-drive command ``(v, omega)``, that point has robot-frame velocity

    w = [v, L * omega]

where ``L`` is the look-ahead distance. The filter solves

    minimize 0.5 * ||w - w_nominal||**2
    subject to g dot w >= -gamma * h

plus linear and angular actuator limits. The two-variable convex projection is
solved analytically and requires no optimization-library dependency.
"""

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np

from robot.safety.laplace_guidance_field import (
    LaplaceGuidanceFieldResult,
)
from robot.safety.local_occupancy_grid import LocalOccupancyGrid
from robot.safety.poisson_safety_function import (
    PoissonSafetyFunctionResult,
)


def _validated_real(
    value: object,
    *,
    name: str,
) -> float:
    """Return a finite real value while rejecting booleans and strings."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")

    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")

    return result


@dataclass(frozen=True)
class PSFCBFConfig:
    """Configuration for the PSF class-aware CBF projection."""

    gamma: float = 1.0
    lookahead_distance: float = 0.25
    minimum_linear_velocity: float = 0.0
    maximum_linear_velocity: float = 0.25
    minimum_angular_velocity: float = -0.8
    maximum_angular_velocity: float = 0.8
    numerical_tolerance: float = 1e-9
    intervention_tolerance: float = 1e-6

    def __post_init__(self) -> None:
        values = {
            "gamma": _validated_real(self.gamma, name="gamma"),
            "lookahead_distance": _validated_real(
                self.lookahead_distance,
                name="lookahead_distance",
            ),
            "minimum_linear_velocity": _validated_real(
                self.minimum_linear_velocity,
                name="minimum_linear_velocity",
            ),
            "maximum_linear_velocity": _validated_real(
                self.maximum_linear_velocity,
                name="maximum_linear_velocity",
            ),
            "minimum_angular_velocity": _validated_real(
                self.minimum_angular_velocity,
                name="minimum_angular_velocity",
            ),
            "maximum_angular_velocity": _validated_real(
                self.maximum_angular_velocity,
                name="maximum_angular_velocity",
            ),
            "numerical_tolerance": _validated_real(
                self.numerical_tolerance,
                name="numerical_tolerance",
            ),
            "intervention_tolerance": _validated_real(
                self.intervention_tolerance,
                name="intervention_tolerance",
            ),
        }

        if values["gamma"] <= 0.0:
            raise ValueError("gamma must be greater than zero.")
        if values["lookahead_distance"] <= 0.0:
            raise ValueError(
                "lookahead_distance must be greater than zero."
            )
        if (
            values["minimum_linear_velocity"]
            > values["maximum_linear_velocity"]
        ):
            raise ValueError(
                "minimum_linear_velocity cannot exceed "
                "maximum_linear_velocity."
            )
        if (
            values["minimum_angular_velocity"]
            > values["maximum_angular_velocity"]
        ):
            raise ValueError(
                "minimum_angular_velocity cannot exceed "
                "maximum_angular_velocity."
            )
        if not (
            values["minimum_linear_velocity"]
            <= 0.0
            <= values["maximum_linear_velocity"]
        ):
            raise ValueError(
                "Linear-velocity limits must contain zero."
            )
        if not (
            values["minimum_angular_velocity"]
            <= 0.0
            <= values["maximum_angular_velocity"]
        ):
            raise ValueError(
                "Angular-velocity limits must contain zero."
            )
        if values["numerical_tolerance"] <= 0.0:
            raise ValueError(
                "numerical_tolerance must be greater than zero."
            )
        if values["intervention_tolerance"] < 0.0:
            raise ValueError(
                "intervention_tolerance cannot be negative."
            )

        for name, value in values.items():
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class PSFCBFResult:
    """Safe command and complete PSF-CBF diagnostics."""

    nominal_linear_velocity: float
    nominal_angular_velocity: float
    safe_linear_velocity: float
    safe_angular_velocity: float
    lookahead_point_robot_xy: tuple[float, float]
    lookahead_cell: tuple[int, int]
    sampled_safety_value: float
    sampled_guidance_vector_robot_xy: tuple[float, float]
    sampled_guidance_magnitude: float
    constraint_right_hand_side: float
    nominal_constraint_left_hand_side: float
    safe_constraint_left_hand_side: float
    constraint_margin: float
    constraint_feasible: bool
    constraint_satisfied: bool
    intervention_active: bool
    emergency_stop_active: bool
    lookahead_occupied: bool


class PSFCBFSafetyFilter:
    """Project a waypoint command onto the class-aware PSF-CBF set."""

    def __init__(
        self,
        config: PSFCBFConfig | None = None,
    ) -> None:
        self.config = config or PSFCBFConfig()
        if not isinstance(self.config, PSFCBFConfig):
            raise TypeError("config must be a PSFCBFConfig.")

    def filter_command(
        self,
        nominal_linear_velocity: float,
        nominal_angular_velocity: float,
        occupancy_grid: LocalOccupancyGrid,
        safety_function: PoissonSafetyFunctionResult,
        guidance_field: LaplaceGuidanceFieldResult,
    ) -> PSFCBFResult:
        """Return the closest actuator-limited PSF-CBF-safe command."""

        nominal_linear = _validated_real(
            nominal_linear_velocity,
            name="nominal_linear_velocity",
        )
        nominal_angular = _validated_real(
            nominal_angular_velocity,
            name="nominal_angular_velocity",
        )
        _validate_pipeline_alignment(
            occupancy_grid,
            safety_function,
            guidance_field,
        )

        lookahead_point = np.array(
            [self.config.lookahead_distance, 0.0],
            dtype=np.float64,
        )
        lookahead_cell = occupancy_grid.robot_point_to_cell(
            lookahead_point
        )
        if lookahead_cell is None:
            raise ValueError(
                "The PSF-CBF look-ahead point lies outside the local grid."
            )

        sampled_safety = float(
            _bilinear_sample_cell_centers(
                safety_function.values,
                point_robot_xy=lookahead_point,
                occupancy_grid=occupancy_grid,
            )
        )
        sampled_guidance = np.asarray(
            _bilinear_sample_cell_centers(
                guidance_field.vectors_robot_xy,
                point_robot_xy=lookahead_point,
                occupancy_grid=occupancy_grid,
            ),
            dtype=np.float64,
        )

        tolerance = self.config.numerical_tolerance
        if sampled_safety < -tolerance:
            return self._emergency_result(
                nominal_linear,
                nominal_angular,
                lookahead_point=lookahead_point,
                lookahead_cell=lookahead_cell,
                sampled_safety=sampled_safety,
                sampled_guidance=sampled_guidance,
                lookahead_occupied=False,
            )
        sampled_safety = max(0.0, sampled_safety)

        lookahead_occupied = bool(
            occupancy_grid.occupied_mask[lookahead_cell]
        )
        if lookahead_occupied:
            return self._emergency_result(
                nominal_linear,
                nominal_angular,
                lookahead_point=lookahead_point,
                lookahead_cell=lookahead_cell,
                sampled_safety=sampled_safety,
                sampled_guidance=sampled_guidance,
                lookahead_occupied=True,
            )

        lookahead_distance = self.config.lookahead_distance
        nominal_velocity = np.array(
            [
                nominal_linear,
                lookahead_distance * nominal_angular,
            ],
            dtype=np.float64,
        )
        lower_bounds = np.array(
            [
                self.config.minimum_linear_velocity,
                (
                    lookahead_distance
                    * self.config.minimum_angular_velocity
                ),
            ],
            dtype=np.float64,
        )
        upper_bounds = np.array(
            [
                self.config.maximum_linear_velocity,
                (
                    lookahead_distance
                    * self.config.maximum_angular_velocity
                ),
            ],
            dtype=np.float64,
        )
        constraint_rhs = -self.config.gamma * sampled_safety

        safe_velocity, feasible = _project_onto_box_halfspace(
            nominal_velocity,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            constraint_normal=sampled_guidance,
            constraint_rhs=constraint_rhs,
            numerical_tolerance=tolerance,
        )

        if not feasible:
            return self._emergency_result(
                nominal_linear,
                nominal_angular,
                lookahead_point=lookahead_point,
                lookahead_cell=lookahead_cell,
                sampled_safety=sampled_safety,
                sampled_guidance=sampled_guidance,
                lookahead_occupied=False,
            )

        safe_linear = float(safe_velocity[0])
        safe_angular = float(
            safe_velocity[1] / lookahead_distance
        )
        nominal_lhs = float(
            np.dot(sampled_guidance, nominal_velocity)
        )
        safe_lhs = float(
            np.dot(sampled_guidance, safe_velocity)
        )
        constraint_margin = safe_lhs - constraint_rhs
        constraint_satisfied = constraint_margin >= -tolerance
        intervention_active = (
            abs(safe_linear - nominal_linear)
            > self.config.intervention_tolerance
            or abs(safe_angular - nominal_angular)
            > self.config.intervention_tolerance
        )

        return PSFCBFResult(
            nominal_linear_velocity=nominal_linear,
            nominal_angular_velocity=nominal_angular,
            safe_linear_velocity=safe_linear,
            safe_angular_velocity=safe_angular,
            lookahead_point_robot_xy=(
                float(lookahead_point[0]),
                float(lookahead_point[1]),
            ),
            lookahead_cell=lookahead_cell,
            sampled_safety_value=sampled_safety,
            sampled_guidance_vector_robot_xy=(
                float(sampled_guidance[0]),
                float(sampled_guidance[1]),
            ),
            sampled_guidance_magnitude=float(
                np.linalg.norm(sampled_guidance)
            ),
            constraint_right_hand_side=constraint_rhs,
            nominal_constraint_left_hand_side=nominal_lhs,
            safe_constraint_left_hand_side=safe_lhs,
            constraint_margin=constraint_margin,
            constraint_feasible=True,
            constraint_satisfied=constraint_satisfied,
            intervention_active=intervention_active,
            emergency_stop_active=False,
            lookahead_occupied=False,
        )

    def _emergency_result(
        self,
        nominal_linear_velocity: float,
        nominal_angular_velocity: float,
        *,
        lookahead_point: np.ndarray,
        lookahead_cell: tuple[int, int],
        sampled_safety: float,
        sampled_guidance: np.ndarray,
        lookahead_occupied: bool,
    ) -> PSFCBFResult:
        """Return a stationary fallback for an invalid or infeasible state."""

        lookahead_distance = self.config.lookahead_distance
        nominal_velocity = np.array(
            [
                nominal_linear_velocity,
                lookahead_distance * nominal_angular_velocity,
            ],
            dtype=np.float64,
        )
        constraint_rhs = -self.config.gamma * sampled_safety
        nominal_lhs = float(
            np.dot(sampled_guidance, nominal_velocity)
        )
        safe_lhs = 0.0

        intervention_active = (
            abs(nominal_linear_velocity)
            > self.config.intervention_tolerance
            or abs(nominal_angular_velocity)
            > self.config.intervention_tolerance
        )

        return PSFCBFResult(
            nominal_linear_velocity=nominal_linear_velocity,
            nominal_angular_velocity=nominal_angular_velocity,
            safe_linear_velocity=0.0,
            safe_angular_velocity=0.0,
            lookahead_point_robot_xy=(
                float(lookahead_point[0]),
                float(lookahead_point[1]),
            ),
            lookahead_cell=lookahead_cell,
            sampled_safety_value=sampled_safety,
            sampled_guidance_vector_robot_xy=(
                float(sampled_guidance[0]),
                float(sampled_guidance[1]),
            ),
            sampled_guidance_magnitude=float(
                np.linalg.norm(sampled_guidance)
            ),
            constraint_right_hand_side=constraint_rhs,
            nominal_constraint_left_hand_side=nominal_lhs,
            safe_constraint_left_hand_side=safe_lhs,
            constraint_margin=safe_lhs - constraint_rhs,
            constraint_feasible=False,
            constraint_satisfied=False,
            intervention_active=intervention_active,
            emergency_stop_active=True,
            lookahead_occupied=lookahead_occupied,
        )


def _validate_pipeline_alignment(
    occupancy_grid: LocalOccupancyGrid,
    safety_function: PoissonSafetyFunctionResult,
    guidance_field: LaplaceGuidanceFieldResult,
) -> None:
    """Validate types, shapes, resolutions, and finite PDE values."""

    if not isinstance(occupancy_grid, LocalOccupancyGrid):
        raise TypeError("occupancy_grid must be a LocalOccupancyGrid.")
    if not isinstance(
        safety_function,
        PoissonSafetyFunctionResult,
    ):
        raise TypeError(
            "safety_function must be a PoissonSafetyFunctionResult."
        )
    if not isinstance(
        guidance_field,
        LaplaceGuidanceFieldResult,
    ):
        raise TypeError(
            "guidance_field must be a LaplaceGuidanceFieldResult."
        )

    if safety_function.shape != occupancy_grid.shape:
        raise ValueError(
            "safety_function shape must match occupancy_grid."
        )
    if guidance_field.shape != occupancy_grid.shape:
        raise ValueError(
            "guidance_field shape must match occupancy_grid."
        )

    resolution = occupancy_grid.config.resolution
    if not math.isclose(
        safety_function.resolution,
        resolution,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "safety_function resolution must match occupancy_grid."
        )
    if not math.isclose(
        guidance_field.resolution,
        resolution,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "guidance_field resolution must match occupancy_grid."
        )

    if not np.isfinite(safety_function.values).all():
        raise ValueError("safety_function values must be finite.")
    if not np.isfinite(guidance_field.vectors_robot_xy).all():
        raise ValueError("guidance_field vectors must be finite.")


def _bilinear_sample_cell_centers(
    values: np.ndarray,
    *,
    point_robot_xy: np.ndarray,
    occupancy_grid: LocalOccupancyGrid,
) -> np.ndarray | float:
    """Bilinearly sample a scalar or vector array at cell centers."""

    config = occupancy_grid.config
    column_position = (
        (point_robot_xy[0] - config.minimum_x) / config.resolution
        - 0.5
    )
    row_position = (
        (point_robot_xy[1] - config.minimum_y) / config.resolution
        - 0.5
    )
    tolerance = 1e-9

    if not (
        -tolerance <= row_position <= config.row_count - 1 + tolerance
        and -tolerance
        <= column_position
        <= config.column_count - 1 + tolerance
    ):
        raise ValueError(
            "Sampling point must lie within the grid-cell-center domain."
        )

    row_position = float(
        np.clip(row_position, 0.0, config.row_count - 1)
    )
    column_position = float(
        np.clip(column_position, 0.0, config.column_count - 1)
    )

    row_zero = int(math.floor(row_position))
    column_zero = int(math.floor(column_position))
    row_one = min(row_zero + 1, config.row_count - 1)
    column_one = min(column_zero + 1, config.column_count - 1)
    row_fraction = row_position - row_zero
    column_fraction = column_position - column_zero

    lower_left = values[row_zero, column_zero]
    lower_right = values[row_zero, column_one]
    upper_left = values[row_one, column_zero]
    upper_right = values[row_one, column_one]

    lower_value = (
        (1.0 - column_fraction) * lower_left
        + column_fraction * lower_right
    )
    upper_value = (
        (1.0 - column_fraction) * upper_left
        + column_fraction * upper_right
    )

    return (
        (1.0 - row_fraction) * lower_value
        + row_fraction * upper_value
    )


def _project_onto_box_halfspace(
    nominal: np.ndarray,
    *,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    constraint_normal: np.ndarray,
    constraint_rhs: float,
    numerical_tolerance: float,
) -> tuple[np.ndarray, bool]:
    """Project a two-dimensional point onto a box and one half-space."""

    clipped_nominal = np.clip(
        nominal,
        lower_bounds,
        upper_bounds,
    )
    normal_squared = float(
        np.dot(constraint_normal, constraint_normal)
    )

    if normal_squared <= numerical_tolerance**2:
        feasible = constraint_rhs <= numerical_tolerance
        return clipped_nominal, feasible

    maximum_point = np.where(
        constraint_normal >= 0.0,
        upper_bounds,
        lower_bounds,
    )
    maximum_lhs = float(
        np.dot(constraint_normal, maximum_point)
    )
    if maximum_lhs < constraint_rhs - numerical_tolerance:
        return np.zeros(2, dtype=np.float64), False

    clipped_lhs = float(
        np.dot(constraint_normal, clipped_nominal)
    )
    if clipped_lhs >= constraint_rhs - numerical_tolerance:
        return clipped_nominal, True

    candidates: list[np.ndarray] = []

    def add_candidate(candidate: np.ndarray) -> None:
        if np.any(candidate < lower_bounds - numerical_tolerance):
            return
        if np.any(candidate > upper_bounds + numerical_tolerance):
            return

        bounded_candidate = np.clip(
            candidate,
            lower_bounds,
            upper_bounds,
        )
        candidate_lhs = float(
            np.dot(constraint_normal, bounded_candidate)
        )
        if candidate_lhs >= constraint_rhs - numerical_tolerance:
            candidates.append(bounded_candidate)

    projection_scale = (
        constraint_rhs
        - float(np.dot(constraint_normal, nominal))
    ) / normal_squared
    add_candidate(
        nominal + projection_scale * constraint_normal
    )

    for fixed_index in range(2):
        free_index = 1 - fixed_index
        for fixed_value in (
            lower_bounds[fixed_index],
            upper_bounds[fixed_index],
        ):
            candidate = np.array(nominal, copy=True)
            candidate[fixed_index] = fixed_value
            free_normal = float(
                constraint_normal[free_index]
            )
            required_free_contribution = (
                constraint_rhs
                - constraint_normal[fixed_index] * fixed_value
            )

            feasible_lower = float(lower_bounds[free_index])
            feasible_upper = float(upper_bounds[free_index])

            if abs(free_normal) <= numerical_tolerance:
                if required_free_contribution > numerical_tolerance:
                    continue
            elif free_normal > 0.0:
                feasible_lower = max(
                    feasible_lower,
                    required_free_contribution / free_normal,
                )
            else:
                feasible_upper = min(
                    feasible_upper,
                    required_free_contribution / free_normal,
                )

            if (
                feasible_lower
                > feasible_upper + numerical_tolerance
            ):
                continue

            candidate[free_index] = np.clip(
                nominal[free_index],
                feasible_lower,
                feasible_upper,
            )
            add_candidate(candidate)

    add_candidate(maximum_point)

    if not candidates:
        return np.zeros(2, dtype=np.float64), False

    objective_values = [
        float(np.sum(np.square(candidate - nominal)))
        for candidate in candidates
    ]
    best_index = int(np.argmin(objective_values))

    return candidates[best_index], True

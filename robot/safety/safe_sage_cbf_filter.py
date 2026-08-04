"""Full-state Safe-SAGE CBF filter for differential-drive commands.

For the configuration state ``z = (x, y, psi)`` and unicycle controls
``u = (v, omega)``, the dynamics are

    z_dot = (v cos(psi), v sin(psi), omega).

Given the yaw-indexed safety field ``h_full(x, y, psi)``, the CBF condition is

    (h_x cos(psi) + h_y sin(psi)) v
        + h_psi omega + h_t + gamma h >= 0.

``h_t`` is accepted explicitly and defaults to zero. It is the extension
point for dynamic-obstacle field derivatives; this module does not estimate
it from tracking data.

The closest actuator-limited command is found by an analytical projection
onto one half-space and a two-dimensional command box. Optional positive
weights control the relative cost of changing linear and angular velocity.
"""

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np

from robot.safety.safe_sage_configuration_space import (
    SafeSAGEConfigurationSpaceResult,
)
from robot.safety.safe_sage_h_full import SafeSAGEHFullResult


def _validated_real(value: object, *, name: str) -> float:
    """Return one finite real number while rejecting booleans."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


@dataclass(frozen=True)
class SafeSAGECBFConfig:
    """Barrier, actuator, numerical, and command-projection settings."""

    gamma: float = 1.0
    minimum_linear_velocity: float = 0.0
    maximum_linear_velocity: float = 0.25
    minimum_angular_velocity: float = -0.8
    maximum_angular_velocity: float = 0.8
    linear_velocity_weight: float = 1.0
    angular_velocity_weight: float = 1.0
    numerical_tolerance: float = 1e-9
    intervention_tolerance: float = 1e-6

    def __post_init__(self) -> None:
        values = {
            name: _validated_real(value, name=name)
            for name, value in (
                ("gamma", self.gamma),
                (
                    "minimum_linear_velocity",
                    self.minimum_linear_velocity,
                ),
                (
                    "maximum_linear_velocity",
                    self.maximum_linear_velocity,
                ),
                (
                    "minimum_angular_velocity",
                    self.minimum_angular_velocity,
                ),
                (
                    "maximum_angular_velocity",
                    self.maximum_angular_velocity,
                ),
                (
                    "linear_velocity_weight",
                    self.linear_velocity_weight,
                ),
                (
                    "angular_velocity_weight",
                    self.angular_velocity_weight,
                ),
                ("numerical_tolerance", self.numerical_tolerance),
                (
                    "intervention_tolerance",
                    self.intervention_tolerance,
                ),
            )
        }

        if values["gamma"] <= 0.0:
            raise ValueError("gamma must be greater than zero.")
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
        if values["linear_velocity_weight"] <= 0.0:
            raise ValueError(
                "linear_velocity_weight must be greater than zero."
            )
        if values["angular_velocity_weight"] <= 0.0:
            raise ValueError(
                "angular_velocity_weight must be greater than zero."
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
class SafeSAGECBFResult:
    """Safe command and complete full-state CBF diagnostics."""

    nominal_linear_velocity: float
    nominal_angular_velocity: float
    safe_linear_velocity: float
    safe_angular_velocity: float
    query_position_grid_xy: tuple[float, float]
    query_yaw_radians: float
    query_cell: tuple[int, int]
    lower_yaw_slice: int
    upper_yaw_slice: int
    yaw_interpolation_weight: float
    sampled_safety_value: float
    sampled_gradient: tuple[float, float, float]
    temporal_safety_derivative: float
    linear_velocity_coefficient: float
    angular_velocity_coefficient: float
    constraint_right_hand_side: float
    nominal_control_left_hand_side: float
    safe_control_left_hand_side: float
    cbf_margin: float
    constraint_feasible: bool
    constraint_satisfied: bool
    intervention_active: bool
    emergency_stop_active: bool
    query_on_unsafe_boundary: bool
    emergency_reason: str | None


class SafeSAGECBFSafetyFilter:
    """Project differential-drive commands onto the full-state safe set."""

    def __init__(self, config: SafeSAGECBFConfig | None = None) -> None:
        self.config = config or SafeSAGECBFConfig()
        if not isinstance(self.config, SafeSAGECBFConfig):
            raise TypeError("config must be a SafeSAGECBFConfig.")

    def filter_command(
        self,
        nominal_linear_velocity: float,
        nominal_angular_velocity: float,
        *,
        query_position_grid_xy: np.ndarray,
        query_yaw_radians: float,
        configuration_space: SafeSAGEConfigurationSpaceResult,
        h_full: SafeSAGEHFullResult,
        temporal_safety_derivative: float = 0.0,
    ) -> SafeSAGECBFResult:
        """Return the nearest actuator-limited full-state-safe command."""

        nominal_linear = _validated_real(
            nominal_linear_velocity,
            name="nominal_linear_velocity",
        )
        nominal_angular = _validated_real(
            nominal_angular_velocity,
            name="nominal_angular_velocity",
        )
        query_yaw = _validated_real(
            query_yaw_radians,
            name="query_yaw_radians",
        )
        temporal_derivative = _validated_real(
            temporal_safety_derivative,
            name="temporal_safety_derivative",
        )
        query_position = _validated_point(query_position_grid_xy)
        _validate_field_alignment(configuration_space, h_full)

        reference_grid = configuration_space.occupancy_grids[0]
        query_cell = reference_grid.robot_point_to_cell(query_position)
        if query_cell is None:
            raise ValueError(
                "query_position_grid_xy lies outside the local grid."
            )

        lower_yaw, upper_yaw, yaw_weight = _yaw_interpolation(
            query_yaw,
            yaw_slice_count=h_full.yaw_slice_count,
        )
        sampled_safety = float(
            _sample_periodic_field(
                h_full.values,
                query_position=query_position,
                grid=reference_grid,
                lower_yaw=lower_yaw,
                upper_yaw=upper_yaw,
                yaw_weight=yaw_weight,
            )
        )
        sampled_spatial_gradient = np.asarray(
            _sample_periodic_field(
                h_full.spatial_gradients_robot_xy,
                query_position=query_position,
                grid=reference_grid,
                lower_yaw=lower_yaw,
                upper_yaw=upper_yaw,
                yaw_weight=yaw_weight,
            ),
            dtype=np.float64,
        )
        sampled_yaw_derivative = float(
            _sample_periodic_field(
                h_full.yaw_derivatives,
                query_position=query_position,
                grid=reference_grid,
                lower_yaw=lower_yaw,
                upper_yaw=upper_yaw,
                yaw_weight=yaw_weight,
            )
        )
        sampled_gradient = np.array(
            [
                sampled_spatial_gradient[0],
                sampled_spatial_gradient[1],
                sampled_yaw_derivative,
            ],
            dtype=np.float64,
        )

        active_yaw_slices = [lower_yaw]
        if yaw_weight > self.config.numerical_tolerance:
            active_yaw_slices.append(upper_yaw)
        query_on_boundary = any(
            bool(
                h_full.dirichlet_boundary_masks[
                    yaw_index,
                    query_cell[0],
                    query_cell[1],
                ]
            )
            for yaw_index in active_yaw_slices
        )

        cosine = math.cos(query_yaw)
        sine = math.sin(query_yaw)
        command_normal = np.array(
            [
                sampled_gradient[0] * cosine
                + sampled_gradient[1] * sine,
                sampled_gradient[2],
            ],
            dtype=np.float64,
        )
        constraint_rhs = (
            -self.config.gamma * sampled_safety
            - temporal_derivative
        )

        if query_on_boundary:
            return self._emergency_result(
                nominal_linear,
                nominal_angular,
                query_position=query_position,
                query_yaw=query_yaw,
                query_cell=query_cell,
                lower_yaw=lower_yaw,
                upper_yaw=upper_yaw,
                yaw_weight=yaw_weight,
                sampled_safety=sampled_safety,
                sampled_gradient=sampled_gradient,
                temporal_derivative=temporal_derivative,
                command_normal=command_normal,
                constraint_rhs=constraint_rhs,
                query_on_boundary=True,
                reason="unsafe_configuration_boundary",
            )

        tolerance = self.config.numerical_tolerance
        if sampled_safety < -tolerance:
            return self._emergency_result(
                nominal_linear,
                nominal_angular,
                query_position=query_position,
                query_yaw=query_yaw,
                query_cell=query_cell,
                lower_yaw=lower_yaw,
                upper_yaw=upper_yaw,
                yaw_weight=yaw_weight,
                sampled_safety=sampled_safety,
                sampled_gradient=sampled_gradient,
                temporal_derivative=temporal_derivative,
                command_normal=command_normal,
                constraint_rhs=constraint_rhs,
                query_on_boundary=False,
                reason="negative_safety_value",
            )
        sampled_safety = max(0.0, sampled_safety)
        constraint_rhs = (
            -self.config.gamma * sampled_safety
            - temporal_derivative
        )

        nominal_command = np.array(
            [nominal_linear, nominal_angular],
            dtype=np.float64,
        )
        lower_bounds = np.array(
            [
                self.config.minimum_linear_velocity,
                self.config.minimum_angular_velocity,
            ],
            dtype=np.float64,
        )
        upper_bounds = np.array(
            [
                self.config.maximum_linear_velocity,
                self.config.maximum_angular_velocity,
            ],
            dtype=np.float64,
        )
        weights = np.array(
            [
                self.config.linear_velocity_weight,
                self.config.angular_velocity_weight,
            ],
            dtype=np.float64,
        )
        safe_command, feasible = _project_weighted_box_halfspace(
            nominal_command,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            weights=weights,
            constraint_normal=command_normal,
            constraint_rhs=constraint_rhs,
            numerical_tolerance=tolerance,
        )
        if not feasible:
            return self._emergency_result(
                nominal_linear,
                nominal_angular,
                query_position=query_position,
                query_yaw=query_yaw,
                query_cell=query_cell,
                lower_yaw=lower_yaw,
                upper_yaw=upper_yaw,
                yaw_weight=yaw_weight,
                sampled_safety=sampled_safety,
                sampled_gradient=sampled_gradient,
                temporal_derivative=temporal_derivative,
                command_normal=command_normal,
                constraint_rhs=constraint_rhs,
                query_on_boundary=False,
                reason="infeasible_cbf_constraint",
            )

        safe_linear = float(safe_command[0])
        safe_angular = float(safe_command[1])
        nominal_lhs = float(np.dot(command_normal, nominal_command))
        safe_lhs = float(np.dot(command_normal, safe_command))
        cbf_margin = safe_lhs - constraint_rhs
        constraint_satisfied = cbf_margin >= -tolerance
        intervention_active = (
            abs(safe_linear - nominal_linear)
            > self.config.intervention_tolerance
            or abs(safe_angular - nominal_angular)
            > self.config.intervention_tolerance
        )

        return SafeSAGECBFResult(
            nominal_linear_velocity=nominal_linear,
            nominal_angular_velocity=nominal_angular,
            safe_linear_velocity=safe_linear,
            safe_angular_velocity=safe_angular,
            query_position_grid_xy=(
                float(query_position[0]),
                float(query_position[1]),
            ),
            query_yaw_radians=query_yaw,
            query_cell=query_cell,
            lower_yaw_slice=lower_yaw,
            upper_yaw_slice=upper_yaw,
            yaw_interpolation_weight=yaw_weight,
            sampled_safety_value=sampled_safety,
            sampled_gradient=(
                float(sampled_gradient[0]),
                float(sampled_gradient[1]),
                float(sampled_gradient[2]),
            ),
            temporal_safety_derivative=temporal_derivative,
            linear_velocity_coefficient=float(command_normal[0]),
            angular_velocity_coefficient=float(command_normal[1]),
            constraint_right_hand_side=constraint_rhs,
            nominal_control_left_hand_side=nominal_lhs,
            safe_control_left_hand_side=safe_lhs,
            cbf_margin=cbf_margin,
            constraint_feasible=True,
            constraint_satisfied=constraint_satisfied,
            intervention_active=intervention_active,
            emergency_stop_active=False,
            query_on_unsafe_boundary=False,
            emergency_reason=None,
        )

    def _emergency_result(
        self,
        nominal_linear: float,
        nominal_angular: float,
        *,
        query_position: np.ndarray,
        query_yaw: float,
        query_cell: tuple[int, int],
        lower_yaw: int,
        upper_yaw: int,
        yaw_weight: float,
        sampled_safety: float,
        sampled_gradient: np.ndarray,
        temporal_derivative: float,
        command_normal: np.ndarray,
        constraint_rhs: float,
        query_on_boundary: bool,
        reason: str,
    ) -> SafeSAGECBFResult:
        """Return a stationary fallback with preserved diagnostics."""

        nominal_command = np.array(
            [nominal_linear, nominal_angular],
            dtype=np.float64,
        )
        nominal_lhs = float(np.dot(command_normal, nominal_command))
        safe_lhs = 0.0
        intervention_active = (
            abs(nominal_linear) > self.config.intervention_tolerance
            or abs(nominal_angular)
            > self.config.intervention_tolerance
        )

        return SafeSAGECBFResult(
            nominal_linear_velocity=nominal_linear,
            nominal_angular_velocity=nominal_angular,
            safe_linear_velocity=0.0,
            safe_angular_velocity=0.0,
            query_position_grid_xy=(
                float(query_position[0]),
                float(query_position[1]),
            ),
            query_yaw_radians=query_yaw,
            query_cell=query_cell,
            lower_yaw_slice=lower_yaw,
            upper_yaw_slice=upper_yaw,
            yaw_interpolation_weight=yaw_weight,
            sampled_safety_value=sampled_safety,
            sampled_gradient=(
                float(sampled_gradient[0]),
                float(sampled_gradient[1]),
                float(sampled_gradient[2]),
            ),
            temporal_safety_derivative=temporal_derivative,
            linear_velocity_coefficient=float(command_normal[0]),
            angular_velocity_coefficient=float(command_normal[1]),
            constraint_right_hand_side=constraint_rhs,
            nominal_control_left_hand_side=nominal_lhs,
            safe_control_left_hand_side=safe_lhs,
            cbf_margin=safe_lhs - constraint_rhs,
            constraint_feasible=False,
            constraint_satisfied=False,
            intervention_active=intervention_active,
            emergency_stop_active=True,
            query_on_unsafe_boundary=query_on_boundary,
            emergency_reason=reason,
        )


def _validated_point(value: object) -> np.ndarray:
    """Return a finite two-dimensional query point."""

    try:
        point = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "query_position_grid_xy must contain real numbers."
        ) from error
    if point.shape != (2,):
        raise ValueError("query_position_grid_xy must have shape (2,).")
    if not np.isfinite(point).all():
        raise ValueError("query_position_grid_xy must be finite.")
    return np.ascontiguousarray(point, dtype=np.float64)


def _validate_field_alignment(
    configuration_space: SafeSAGEConfigurationSpaceResult,
    h_full: SafeSAGEHFullResult,
) -> None:
    """Validate yaw, grid, shape, resolution, and numerical alignment."""

    if not isinstance(
        configuration_space,
        SafeSAGEConfigurationSpaceResult,
    ):
        raise TypeError(
            "configuration_space must be a "
            "SafeSAGEConfigurationSpaceResult."
        )
    if not isinstance(h_full, SafeSAGEHFullResult):
        raise TypeError("h_full must be a SafeSAGEHFullResult.")
    yaw_slice_count = configuration_space.yaw_slice_count
    if yaw_slice_count < 2:
        raise ValueError(
            "configuration_space must contain at least two yaw slices."
        )
    if np.asarray(configuration_space.yaw_angles_radians).shape != (
        yaw_slice_count,
    ):
        raise ValueError(
            "configuration_space yaw angles must match its grid count."
        )
    if np.asarray(h_full.yaw_angles_radians).shape != (
        yaw_slice_count,
    ):
        raise ValueError(
            "h_full yaw angles must match the configuration-space "
            "grid count."
        )
    expected_yaw_angles = np.linspace(
        -math.pi,
        math.pi,
        num=yaw_slice_count,
        endpoint=False,
        dtype=np.float64,
    )
    if not np.allclose(
        configuration_space.yaw_angles_radians,
        expected_yaw_angles,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "configuration_space yaw angles must be uniformly spaced "
            "on [-pi, pi)."
        )
    if h_full.values.ndim != 3:
        raise ValueError(
            "h_full values must have shape (yaw, rows, columns)."
        )
    if yaw_slice_count != h_full.yaw_slice_count:
        raise ValueError(
            "configuration_space and h_full yaw counts must match."
        )
    if not np.allclose(
        configuration_space.yaw_angles_radians,
        h_full.yaw_angles_radians,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "configuration_space and h_full yaw angles must match."
        )

    reference_grid = configuration_space.occupancy_grids[0]
    expected_scalar_shape = (yaw_slice_count,) + reference_grid.shape
    expected_spatial_gradient_shape = expected_scalar_shape + (2,)
    if h_full.values.shape != expected_scalar_shape:
        raise ValueError(
            "h_full values must match the configuration-space shape."
        )
    if (
        h_full.spatial_gradients_robot_xy.shape
        != expected_spatial_gradient_shape
    ):
        raise ValueError(
            "h_full spatial gradients must have shape "
            "(yaw, rows, columns, 2)."
        )
    if h_full.yaw_derivatives.shape != expected_scalar_shape:
        raise ValueError(
            "h_full yaw derivatives must match the values shape."
        )
    if h_full.dirichlet_boundary_masks.shape != expected_scalar_shape:
        raise ValueError(
            "h_full boundary masks must match the values shape."
        )
    if not math.isclose(
        h_full.resolution,
        reference_grid.config.resolution,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "h_full resolution must match the configuration space."
        )
    for grid in configuration_space.occupancy_grids:
        if grid.config != reference_grid.config:
            raise ValueError(
                "All configuration-space grids must share one config."
            )

    arrays = (
        h_full.values,
        h_full.spatial_gradients_robot_xy,
        h_full.yaw_derivatives,
    )
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("h_full arrays must contain finite values.")


def _yaw_interpolation(
    yaw_radians: float,
    *,
    yaw_slice_count: int,
) -> tuple[int, int, float]:
    """Return periodic yaw indices and linear interpolation weight."""

    wrapped_yaw = (yaw_radians + math.pi) % math.tau - math.pi
    spacing = math.tau / yaw_slice_count
    position = (wrapped_yaw + math.pi) / spacing
    floored_position = math.floor(position)
    lower_index = floored_position % yaw_slice_count
    upper_index = (lower_index + 1) % yaw_slice_count
    return lower_index, upper_index, float(position - floored_position)


def _sample_periodic_field(
    values: np.ndarray,
    *,
    query_position: np.ndarray,
    grid,
    lower_yaw: int,
    upper_yaw: int,
    yaw_weight: float,
) -> np.ndarray | float:
    """Bilinearly sample space and linearly interpolate periodic yaw."""

    lower_value = _bilinear_sample_cell_centers(
        values[lower_yaw],
        point_grid_xy=query_position,
        grid=grid,
    )
    upper_value = _bilinear_sample_cell_centers(
        values[upper_yaw],
        point_grid_xy=query_position,
        grid=grid,
    )
    return (1.0 - yaw_weight) * lower_value + yaw_weight * upper_value


def _bilinear_sample_cell_centers(
    values: np.ndarray,
    *,
    point_grid_xy: np.ndarray,
    grid,
) -> np.ndarray | float:
    """Bilinearly sample scalar or vector data at grid-cell centers."""

    config = grid.config
    column_position = (
        (point_grid_xy[0] - config.minimum_x) / config.resolution
        - 0.5
    )
    row_position = (
        (point_grid_xy[1] - config.minimum_y) / config.resolution
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
            "query_position_grid_xy must lie within the grid-cell-center "
            "domain."
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

    lower_value = (
        (1.0 - column_fraction) * values[row_zero, column_zero]
        + column_fraction * values[row_zero, column_one]
    )
    upper_value = (
        (1.0 - column_fraction) * values[row_one, column_zero]
        + column_fraction * values[row_one, column_one]
    )
    return (
        (1.0 - row_fraction) * lower_value
        + row_fraction * upper_value
    )


def _project_weighted_box_halfspace(
    nominal: np.ndarray,
    *,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    weights: np.ndarray,
    constraint_normal: np.ndarray,
    constraint_rhs: float,
    numerical_tolerance: float,
) -> tuple[np.ndarray, bool]:
    """Solve a weighted two-command projection by scaling coordinates."""

    square_root_weights = np.sqrt(weights)
    scaled_command, feasible = _project_onto_box_halfspace(
        nominal * square_root_weights,
        lower_bounds=lower_bounds * square_root_weights,
        upper_bounds=upper_bounds * square_root_weights,
        constraint_normal=constraint_normal / square_root_weights,
        constraint_rhs=constraint_rhs,
        numerical_tolerance=numerical_tolerance,
    )
    return scaled_command / square_root_weights, feasible


def _project_onto_box_halfspace(
    nominal: np.ndarray,
    *,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    constraint_normal: np.ndarray,
    constraint_rhs: float,
    numerical_tolerance: float,
) -> tuple[np.ndarray, bool]:
    """Project one 2D point onto a box intersected with a half-space."""

    clipped_nominal = np.clip(nominal, lower_bounds, upper_bounds)
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
    maximum_lhs = float(np.dot(constraint_normal, maximum_point))
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
        constraint_rhs - float(np.dot(constraint_normal, nominal))
    ) / normal_squared
    add_candidate(nominal + projection_scale * constraint_normal)

    for fixed_index in range(2):
        free_index = 1 - fixed_index
        for fixed_value in (
            lower_bounds[fixed_index],
            upper_bounds[fixed_index],
        ):
            candidate = np.array(nominal, copy=True)
            candidate[fixed_index] = fixed_value
            free_normal = float(constraint_normal[free_index])
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

            if feasible_lower > feasible_upper + numerical_tolerance:
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
    return candidates[int(np.argmin(objective_values))], True

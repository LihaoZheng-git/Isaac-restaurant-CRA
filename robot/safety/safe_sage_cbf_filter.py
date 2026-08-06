"""Paper-faithful analytical Safe-SAGE safety filter.

Safe-SAGE constructs two distinct field products:

* ``h_full(x, y, psi)`` supplies the scalar safety value and yaw derivative;
* ``v_sem(x, y, psi)`` supplies the normal-and-tangential semantic guidance.

For the reduced-order command ``u = (u_x, u_y, omega)``, the analytical
constraint from Safe-SAGE is

    [v_sem_x, v_sem_y, partial(h_full)/partial(psi)] dot u
        + temporal_scale * partial(h_full)/partial(t)
        >= -gamma * h_full.

The spatial gradient of ``h_full`` is retained only for diagnostics and the
paper's temporal scaling. It deliberately does not replace ``v_sem`` in the
constraint: doing so would discard the non-conservative tangential flow that
encodes passing-side semantics.

This module operates in the paper's planar reduced-order command space.
Embodiment-specific control interfaces, including Nova Carter's
differential-drive mapping, live in separate modules.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from numbers import Real

import numpy as np
from robot.safety.laplace_guidance_field import (
    LaplaceGuidanceFieldResult,
)
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
    """Barrier, reduced-order bounds, and projection settings."""

    gamma: float = 1.0
    minimum_planar_velocity: float = -0.25
    maximum_planar_velocity: float = 0.25
    minimum_angular_velocity: float = -0.8
    maximum_angular_velocity: float = 0.8
    planar_velocity_weight: float = 1.0
    angular_velocity_weight: float = 1.0
    temporal_scaling_epsilon: float = 1e-6
    temporal_scaling_kappa: float = 1.0
    numerical_tolerance: float = 1e-9
    intervention_tolerance: float = 1e-6

    def __post_init__(self) -> None:
        values = {
            name: _validated_real(value, name=name)
            for name, value in (
                ("gamma", self.gamma),
                (
                    "minimum_planar_velocity",
                    self.minimum_planar_velocity,
                ),
                (
                    "maximum_planar_velocity",
                    self.maximum_planar_velocity,
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
                    "planar_velocity_weight",
                    self.planar_velocity_weight,
                ),
                (
                    "angular_velocity_weight",
                    self.angular_velocity_weight,
                ),
                (
                    "temporal_scaling_epsilon",
                    self.temporal_scaling_epsilon,
                ),
                (
                    "temporal_scaling_kappa",
                    self.temporal_scaling_kappa,
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
        if values["minimum_planar_velocity"] > values["maximum_planar_velocity"]:
            raise ValueError(
                "minimum_planar_velocity cannot exceed maximum_planar_velocity."
            )
        if values["minimum_angular_velocity"] > values["maximum_angular_velocity"]:
            raise ValueError(
                "minimum_angular_velocity cannot exceed maximum_angular_velocity."
            )
        if not (
            values["minimum_planar_velocity"]
            <= 0.0
            <= values["maximum_planar_velocity"]
        ):
            raise ValueError("Planar-velocity limits must contain zero.")
        if not (
            values["minimum_angular_velocity"]
            <= 0.0
            <= values["maximum_angular_velocity"]
        ):
            raise ValueError("Angular-velocity limits must contain zero.")
        if values["planar_velocity_weight"] <= 0.0:
            raise ValueError("planar_velocity_weight must be greater than zero.")
        if values["angular_velocity_weight"] <= 0.0:
            raise ValueError("angular_velocity_weight must be greater than zero.")
        if values["temporal_scaling_epsilon"] <= 0.0:
            raise ValueError("temporal_scaling_epsilon must be greater than zero.")
        if values["temporal_scaling_kappa"] <= 0.0:
            raise ValueError("temporal_scaling_kappa must be greater than zero.")
        if values["numerical_tolerance"] <= 0.0:
            raise ValueError("numerical_tolerance must be greater than zero.")
        if values["intervention_tolerance"] < 0.0:
            raise ValueError("intervention_tolerance cannot be negative.")

        for name, value in values.items():
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class SafeSAGEConstraintEvaluation:
    """Interpolated fields and one paper-faithful CBF half-space."""

    query_position_grid_xy: tuple[float, float]
    query_yaw_radians: float
    query_cell: tuple[int, int]
    lower_yaw_slice: int
    upper_yaw_slice: int
    yaw_interpolation_weight: float
    sampled_safety_value: float
    sampled_gradient: tuple[float, float, float]
    interpolant_gradient: tuple[float, float, float]
    sampled_guidance_grid_xy: tuple[float, float]
    temporal_safety_derivative: float
    temporal_scaling: float
    scaled_temporal_safety_derivative: float
    constraint_normal: tuple[float, float, float]
    constraint_right_hand_side: float
    query_on_unsafe_boundary: bool
    unsafe_reason: str | None


@dataclass(frozen=True)
class SafeSAGECBFResult:
    """Safe reduced-order command and complete CBF diagnostics."""

    nominal_planar_velocity_grid_xy: tuple[float, float]
    nominal_angular_velocity: float
    safe_planar_velocity_grid_xy: tuple[float, float]
    safe_angular_velocity: float
    constraint: SafeSAGEConstraintEvaluation
    nominal_control_left_hand_side: float
    safe_control_left_hand_side: float
    cbf_margin: float
    constraint_feasible: bool
    constraint_satisfied: bool
    intervention_active: bool
    emergency_stop_active: bool
    emergency_reason: str | None

    @property
    def query_position_grid_xy(self) -> tuple[float, float]:
        return self.constraint.query_position_grid_xy

    @property
    def query_yaw_radians(self) -> float:
        return self.constraint.query_yaw_radians

    @property
    def query_cell(self) -> tuple[int, int]:
        return self.constraint.query_cell

    @property
    def lower_yaw_slice(self) -> int:
        return self.constraint.lower_yaw_slice

    @property
    def upper_yaw_slice(self) -> int:
        return self.constraint.upper_yaw_slice

    @property
    def yaw_interpolation_weight(self) -> float:
        return self.constraint.yaw_interpolation_weight

    @property
    def sampled_safety_value(self) -> float:
        return self.constraint.sampled_safety_value

    @property
    def sampled_gradient(self) -> tuple[float, float, float]:
        return self.constraint.sampled_gradient

    @property
    def interpolant_gradient(self) -> tuple[float, float, float]:
        return self.constraint.interpolant_gradient

    @property
    def sampled_guidance_grid_xy(self) -> tuple[float, float]:
        return self.constraint.sampled_guidance_grid_xy

    @property
    def temporal_safety_derivative(self) -> float:
        return self.constraint.temporal_safety_derivative

    @property
    def temporal_scaling(self) -> float:
        return self.constraint.temporal_scaling

    @property
    def scaled_temporal_safety_derivative(self) -> float:
        return self.constraint.scaled_temporal_safety_derivative

    @property
    def constraint_normal(self) -> tuple[float, float, float]:
        return self.constraint.constraint_normal

    @property
    def constraint_right_hand_side(self) -> float:
        return self.constraint.constraint_right_hand_side

    @property
    def query_on_unsafe_boundary(self) -> bool:
        return self.constraint.query_on_unsafe_boundary


class SafeSAGECBFSafetyFilter:
    """Project planar reduced-order commands using Safe-SAGE Eq. (19)."""

    def __init__(self, config: SafeSAGECBFConfig | None = None) -> None:
        self.config = config or SafeSAGECBFConfig()
        if not isinstance(self.config, SafeSAGECBFConfig):
            raise TypeError("config must be a SafeSAGECBFConfig.")

    def filter_command(
        self,
        nominal_planar_velocity_grid_xy: np.ndarray,
        nominal_angular_velocity: float,
        *,
        query_position_grid_xy: np.ndarray,
        query_yaw_radians: float,
        configuration_space: SafeSAGEConfigurationSpaceResult,
        h_full: SafeSAGEHFullResult,
        guidance_fields: Sequence[LaplaceGuidanceFieldResult],
        temporal_safety_derivative: float = 0.0,
    ) -> SafeSAGECBFResult:
        """Return the nearest reduced-order command satisfying Eq. (19)."""

        nominal_planar = _validated_point(
            nominal_planar_velocity_grid_xy,
            name="nominal_planar_velocity_grid_xy",
        )
        nominal_angular = _validated_real(
            nominal_angular_velocity,
            name="nominal_angular_velocity",
        )
        constraint = self.evaluate_constraint(
            query_position_grid_xy=query_position_grid_xy,
            query_yaw_radians=query_yaw_radians,
            configuration_space=configuration_space,
            h_full=h_full,
            guidance_fields=guidance_fields,
            temporal_safety_derivative=temporal_safety_derivative,
        )
        nominal_command = np.array(
            [nominal_planar[0], nominal_planar[1], nominal_angular],
            dtype=np.float64,
        )
        if constraint.unsafe_reason is not None:
            return self._emergency_result(
                nominal_command,
                constraint=constraint,
                reason=constraint.unsafe_reason,
            )

        lower_bounds = np.array(
            [
                self.config.minimum_planar_velocity,
                self.config.minimum_planar_velocity,
                self.config.minimum_angular_velocity,
            ],
            dtype=np.float64,
        )
        upper_bounds = np.array(
            [
                self.config.maximum_planar_velocity,
                self.config.maximum_planar_velocity,
                self.config.maximum_angular_velocity,
            ],
            dtype=np.float64,
        )
        weights = np.array(
            [
                self.config.planar_velocity_weight,
                self.config.planar_velocity_weight,
                self.config.angular_velocity_weight,
            ],
            dtype=np.float64,
        )
        command_normal = np.asarray(
            constraint.constraint_normal,
            dtype=np.float64,
        )
        tolerance = self.config.numerical_tolerance
        safe_command, feasible = project_weighted_box_halfspace(
            nominal_command,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            weights=weights,
            constraint_normal=command_normal,
            constraint_rhs=constraint.constraint_right_hand_side,
            numerical_tolerance=tolerance,
        )
        if not feasible:
            return self._emergency_result(
                nominal_command,
                constraint=constraint,
                reason="infeasible_cbf_constraint",
            )

        nominal_lhs = float(np.dot(command_normal, nominal_command))
        safe_lhs = float(np.dot(command_normal, safe_command))
        cbf_margin = safe_lhs - constraint.constraint_right_hand_side
        constraint_satisfied = cbf_margin >= -tolerance
        intervention_active = bool(
            np.any(
                np.abs(safe_command - nominal_command)
                > self.config.intervention_tolerance
            )
        )

        return SafeSAGECBFResult(
            nominal_planar_velocity_grid_xy=(
                float(nominal_command[0]),
                float(nominal_command[1]),
            ),
            nominal_angular_velocity=nominal_angular,
            safe_planar_velocity_grid_xy=(
                float(safe_command[0]),
                float(safe_command[1]),
            ),
            safe_angular_velocity=float(safe_command[2]),
            constraint=constraint,
            nominal_control_left_hand_side=nominal_lhs,
            safe_control_left_hand_side=safe_lhs,
            cbf_margin=cbf_margin,
            constraint_feasible=True,
            constraint_satisfied=constraint_satisfied,
            intervention_active=intervention_active,
            emergency_stop_active=False,
            emergency_reason=None,
        )

    def evaluate_constraint(
        self,
        *,
        query_position_grid_xy: np.ndarray,
        query_yaw_radians: float,
        configuration_space: SafeSAGEConfigurationSpaceResult,
        h_full: SafeSAGEHFullResult,
        guidance_fields: Sequence[LaplaceGuidanceFieldResult],
        temporal_safety_derivative: float = 0.0,
    ) -> SafeSAGEConstraintEvaluation:
        """Interpolate fields and construct the Safe-SAGE half-space."""

        query_position = _validated_point(
            query_position_grid_xy,
            name="query_position_grid_xy",
        )
        query_yaw = _validated_real(
            query_yaw_radians,
            name="query_yaw_radians",
        )
        temporal_derivative = _validated_real(
            temporal_safety_derivative,
            name="temporal_safety_derivative",
        )
        guidance_sequence = tuple(guidance_fields)
        _validate_field_alignment(
            configuration_space,
            h_full,
            guidance_sequence,
        )

        reference_grid = configuration_space.occupancy_grids[0]
        query_cell = reference_grid.robot_point_to_cell(query_position)
        if query_cell is None:
            raise ValueError("query_position_grid_xy lies outside the local grid.")
        lower_yaw, upper_yaw, yaw_weight = _yaw_interpolation(
            query_yaw,
            yaw_slice_count=h_full.yaw_slice_count,
        )
        (
            sampled_safety,
            sampled_spatial_gradient,
            interpolant_yaw_derivative,
        ) = _sample_periodic_scalar_and_spatial_gradient(
            h_full.values,
            query_position=query_position,
            grid=reference_grid,
            lower_yaw=lower_yaw,
            upper_yaw=upper_yaw,
            yaw_weight=yaw_weight,
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
        if yaw_weight <= self.config.numerical_tolerance:
            # The piecewise-linear yaw interpolant is nondifferentiable at a
            # slice knot. Use the precomputed centered derivative there,
            # matching a symmetric finite-difference/SQP linearization.
            interpolant_yaw_derivative = sampled_yaw_derivative
        sampled_guidance = _sample_periodic_guidance(
            guidance_sequence,
            query_position=query_position,
            grid=reference_grid,
            lower_yaw=lower_yaw,
            upper_yaw=upper_yaw,
            yaw_weight=yaw_weight,
        )

        tolerance = self.config.numerical_tolerance
        if yaw_weight < 0.5 - tolerance:
            active_yaw_slices = (lower_yaw,)
        elif yaw_weight > 0.5 + tolerance:
            active_yaw_slices = (upper_yaw,)
        else:
            active_yaw_slices = (lower_yaw, upper_yaw)
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

        temporal_scaling = _temporal_scaling(
            sampled_guidance,
            sampled_spatial_gradient,
            safety_value=sampled_safety,
            epsilon=self.config.temporal_scaling_epsilon,
            kappa=self.config.temporal_scaling_kappa,
        )
        scaled_temporal_derivative = temporal_scaling * temporal_derivative
        command_normal = np.array(
            [
                sampled_guidance[0],
                sampled_guidance[1],
                sampled_yaw_derivative,
            ],
            dtype=np.float64,
        )
        safety_for_constraint = max(0.0, sampled_safety)
        constraint_rhs = (
            -self.config.gamma * safety_for_constraint - scaled_temporal_derivative
        )
        unsafe_reason: str | None = None
        if query_on_boundary:
            unsafe_reason = "unsafe_configuration_boundary"
        elif sampled_safety < -tolerance:
            unsafe_reason = "negative_safety_value"

        return SafeSAGEConstraintEvaluation(
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
                float(sampled_spatial_gradient[0]),
                float(sampled_spatial_gradient[1]),
                sampled_yaw_derivative,
            ),
            interpolant_gradient=(
                float(sampled_spatial_gradient[0]),
                float(sampled_spatial_gradient[1]),
                interpolant_yaw_derivative,
            ),
            sampled_guidance_grid_xy=(
                float(sampled_guidance[0]),
                float(sampled_guidance[1]),
            ),
            temporal_safety_derivative=temporal_derivative,
            temporal_scaling=temporal_scaling,
            scaled_temporal_safety_derivative=(scaled_temporal_derivative),
            constraint_normal=(
                float(command_normal[0]),
                float(command_normal[1]),
                float(command_normal[2]),
            ),
            constraint_right_hand_side=constraint_rhs,
            query_on_unsafe_boundary=query_on_boundary,
            unsafe_reason=unsafe_reason,
        )

    def _emergency_result(
        self,
        nominal_command: np.ndarray,
        *,
        constraint: SafeSAGEConstraintEvaluation,
        reason: str,
    ) -> SafeSAGECBFResult:
        """Return a stationary fallback with preserved diagnostics."""

        command_normal = np.asarray(
            constraint.constraint_normal,
            dtype=np.float64,
        )
        nominal_lhs = float(np.dot(command_normal, nominal_command))
        safe_lhs = 0.0
        intervention_active = bool(
            np.any(np.abs(nominal_command) > self.config.intervention_tolerance)
        )

        return SafeSAGECBFResult(
            nominal_planar_velocity_grid_xy=(
                float(nominal_command[0]),
                float(nominal_command[1]),
            ),
            nominal_angular_velocity=float(nominal_command[2]),
            safe_planar_velocity_grid_xy=(0.0, 0.0),
            safe_angular_velocity=0.0,
            constraint=constraint,
            nominal_control_left_hand_side=nominal_lhs,
            safe_control_left_hand_side=safe_lhs,
            cbf_margin=(safe_lhs - constraint.constraint_right_hand_side),
            constraint_feasible=False,
            constraint_satisfied=False,
            intervention_active=intervention_active,
            emergency_stop_active=True,
            emergency_reason=reason,
        )


def _validated_point(value: object, *, name: str) -> np.ndarray:
    """Return a finite two-dimensional query point."""

    try:
        point = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must contain real numbers.") from error
    if point.shape != (2,):
        raise ValueError(f"{name} must have shape (2,).")
    if not np.isfinite(point).all():
        raise ValueError(f"{name} must be finite.")
    return np.ascontiguousarray(point, dtype=np.float64)


def _validate_field_alignment(
    configuration_space: SafeSAGEConfigurationSpaceResult,
    h_full: SafeSAGEHFullResult,
    guidance_fields: tuple[LaplaceGuidanceFieldResult, ...],
) -> None:
    """Validate configuration, safety, and guidance-field alignment."""

    if not isinstance(
        configuration_space,
        SafeSAGEConfigurationSpaceResult,
    ):
        raise TypeError(
            "configuration_space must be a SafeSAGEConfigurationSpaceResult."
        )
    if not isinstance(h_full, SafeSAGEHFullResult):
        raise TypeError("h_full must be a SafeSAGEHFullResult.")
    yaw_slice_count = configuration_space.yaw_slice_count
    if yaw_slice_count < 2:
        raise ValueError("configuration_space must contain at least two yaw slices.")
    if len(guidance_fields) != yaw_slice_count:
        raise ValueError("guidance_fields length must match the yaw-slice count.")
    if np.asarray(configuration_space.yaw_angles_radians).shape != (yaw_slice_count,):
        raise ValueError("configuration_space yaw angles must match its grid count.")
    if np.asarray(h_full.yaw_angles_radians).shape != (yaw_slice_count,):
        raise ValueError(
            "h_full yaw angles must match the configuration-space grid count."
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
            "configuration_space yaw angles must be uniformly spaced on [-pi, pi)."
        )
    if h_full.values.ndim != 3:
        raise ValueError("h_full values must have shape (yaw, rows, columns).")
    if yaw_slice_count != h_full.yaw_slice_count:
        raise ValueError("configuration_space and h_full yaw counts must match.")
    if not np.allclose(
        configuration_space.yaw_angles_radians,
        h_full.yaw_angles_radians,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("configuration_space and h_full yaw angles must match.")

    reference_grid = configuration_space.occupancy_grids[0]
    expected_scalar_shape = (yaw_slice_count,) + reference_grid.shape
    expected_spatial_gradient_shape = expected_scalar_shape + (2,)
    if h_full.values.shape != expected_scalar_shape:
        raise ValueError("h_full values must match the configuration-space shape.")
    if h_full.spatial_gradients_robot_xy.shape != expected_spatial_gradient_shape:
        raise ValueError(
            "h_full spatial gradients must have shape (yaw, rows, columns, 2)."
        )
    if h_full.yaw_derivatives.shape != expected_scalar_shape:
        raise ValueError("h_full yaw derivatives must match the values shape.")
    if h_full.dirichlet_boundary_masks.shape != expected_scalar_shape:
        raise ValueError("h_full boundary masks must match the values shape.")
    if not math.isclose(
        h_full.resolution,
        reference_grid.config.resolution,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise ValueError("h_full resolution must match the configuration space.")
    for grid in configuration_space.occupancy_grids:
        if grid.config != reference_grid.config:
            raise ValueError("All configuration-space grids must share one config.")

    for field_index, guidance_field in enumerate(guidance_fields):
        if not isinstance(guidance_field, LaplaceGuidanceFieldResult):
            raise TypeError(
                "Every guidance field must be a LaplaceGuidanceFieldResult."
            )
        if guidance_field.shape != reference_grid.shape:
            raise ValueError(
                "Guidance-field shape must match the configuration "
                f"space at yaw index {field_index}."
            )
        if not math.isclose(
            guidance_field.resolution,
            reference_grid.config.resolution,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Guidance-field resolution must match the "
                f"configuration space at yaw index {field_index}."
            )
        if not np.isfinite(guidance_field.vectors_robot_xy).all():
            raise ValueError("Guidance-field vectors must contain finite values.")

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


def _sample_periodic_guidance(
    guidance_fields: tuple[LaplaceGuidanceFieldResult, ...],
    *,
    query_position: np.ndarray,
    grid,
    lower_yaw: int,
    upper_yaw: int,
    yaw_weight: float,
) -> np.ndarray:
    """Trilinearly sample the semantic LGF at one configuration."""

    lower_value = _bilinear_sample_cell_centers(
        guidance_fields[lower_yaw].vectors_robot_xy,
        point_grid_xy=query_position,
        grid=grid,
    )
    upper_value = _bilinear_sample_cell_centers(
        guidance_fields[upper_yaw].vectors_robot_xy,
        point_grid_xy=query_position,
        grid=grid,
    )
    sampled = (1.0 - yaw_weight) * lower_value + yaw_weight * upper_value
    return np.asarray(sampled, dtype=np.float64)


def _temporal_scaling(
    sampled_guidance: np.ndarray,
    sampled_spatial_gradient: np.ndarray,
    *,
    safety_value: float,
    epsilon: float,
    kappa: float,
) -> float:
    """Return the bounded temporal multiplier from Safe-SAGE Eq. (17)."""

    guidance_norm = float(np.linalg.norm(sampled_guidance))
    gradient_norm = float(np.linalg.norm(sampled_spatial_gradient))
    open_space_regularizer = epsilon * (1.0 - math.exp(-kappa * max(0.0, safety_value)))
    denominator = gradient_norm + open_space_regularizer
    if denominator <= np.finfo(np.float64).eps:
        return 1.0 if guidance_norm > 0.0 else 0.0
    return min(guidance_norm / denominator, 1.0)


def _sample_periodic_scalar_and_spatial_gradient(
    values: np.ndarray,
    *,
    query_position: np.ndarray,
    grid,
    lower_yaw: int,
    upper_yaw: int,
    yaw_weight: float,
) -> tuple[float, np.ndarray, float]:
    """Sample one scalar field and differentiate its spatial/yaw interpolant."""

    lower_value, lower_gradient = _bilinear_sample_scalar_with_gradient(
        values[lower_yaw],
        point_grid_xy=query_position,
        grid=grid,
    )
    upper_value, upper_gradient = _bilinear_sample_scalar_with_gradient(
        values[upper_yaw],
        point_grid_xy=query_position,
        grid=grid,
    )
    sampled_value = (1.0 - yaw_weight) * lower_value + yaw_weight * upper_value
    sampled_gradient = (1.0 - yaw_weight) * lower_gradient + yaw_weight * upper_gradient
    yaw_spacing = math.tau / values.shape[0]
    yaw_derivative = (upper_value - lower_value) / yaw_spacing
    return (
        float(sampled_value),
        np.asarray(sampled_gradient, dtype=np.float64),
        float(yaw_derivative),
    )


def _bilinear_sample_cell_centers(
    values: np.ndarray,
    *,
    point_grid_xy: np.ndarray,
    grid,
) -> np.ndarray | float:
    """Bilinearly sample scalar or vector data at grid-cell centers."""

    (
        row_zero,
        row_one,
        column_zero,
        column_one,
        row_fraction,
        column_fraction,
    ) = _bilinear_coordinates(point_grid_xy, grid=grid)

    lower_value = (1.0 - column_fraction) * values[
        row_zero, column_zero
    ] + column_fraction * values[row_zero, column_one]
    upper_value = (1.0 - column_fraction) * values[
        row_one, column_zero
    ] + column_fraction * values[row_one, column_one]
    return (1.0 - row_fraction) * lower_value + row_fraction * upper_value


def _bilinear_sample_scalar_with_gradient(
    values: np.ndarray,
    *,
    point_grid_xy: np.ndarray,
    grid,
) -> tuple[float, np.ndarray]:
    """Return a scalar value and exact gradient of its bilinear patch."""

    if values.ndim != 2:
        raise ValueError("Bilinear scalar sampling requires a 2D array.")
    (
        row_zero,
        row_one,
        column_zero,
        column_one,
        row_fraction,
        column_fraction,
    ) = _bilinear_coordinates(point_grid_xy, grid=grid)

    value_00 = float(values[row_zero, column_zero])
    value_01 = float(values[row_zero, column_one])
    value_10 = float(values[row_one, column_zero])
    value_11 = float(values[row_one, column_one])

    lower_value = (1.0 - column_fraction) * value_00 + column_fraction * value_01
    upper_value = (1.0 - column_fraction) * value_10 + column_fraction * value_11
    sampled_value = (1.0 - row_fraction) * lower_value + row_fraction * upper_value

    resolution = grid.config.resolution
    derivative_x = (
        (1.0 - row_fraction) * (value_01 - value_00)
        + row_fraction * (value_11 - value_10)
    ) / resolution
    derivative_y = (
        (1.0 - column_fraction) * (value_10 - value_00)
        + column_fraction * (value_11 - value_01)
    ) / resolution
    return float(sampled_value), np.array(
        [derivative_x, derivative_y],
        dtype=np.float64,
    )


def _bilinear_coordinates(
    point_grid_xy: np.ndarray,
    *,
    grid,
) -> tuple[int, int, int, int, float, float]:
    """Return checked cell-center interpolation indices and fractions."""

    config = grid.config
    column_position = (point_grid_xy[0] - config.minimum_x) / config.resolution - 0.5
    row_position = (point_grid_xy[1] - config.minimum_y) / config.resolution - 0.5
    tolerance = 1e-9
    if not (
        -tolerance <= row_position <= config.row_count - 1 + tolerance
        and -tolerance <= column_position <= config.column_count - 1 + tolerance
    ):
        raise ValueError(
            "query_position_grid_xy must lie within the grid-cell-center domain."
        )

    row_position = float(np.clip(row_position, 0.0, config.row_count - 1))
    column_position = float(np.clip(column_position, 0.0, config.column_count - 1))
    row_zero = math.floor(row_position)
    column_zero = math.floor(column_position)
    row_one = min(row_zero + 1, config.row_count - 1)
    column_one = min(column_zero + 1, config.column_count - 1)
    return (
        row_zero,
        row_one,
        column_zero,
        column_one,
        row_position - row_zero,
        column_position - column_zero,
    )


def project_weighted_box_halfspace(
    nominal: np.ndarray,
    *,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    weights: np.ndarray,
    constraint_normal: np.ndarray,
    constraint_rhs: float,
    numerical_tolerance: float,
) -> tuple[np.ndarray, bool]:
    """Solve a weighted finite-dimensional box/half-space projection."""

    arrays = tuple(
        np.asarray(value, dtype=np.float64)
        for value in (
            nominal,
            lower_bounds,
            upper_bounds,
            weights,
            constraint_normal,
        )
    )
    nominal_array, lower_array, upper_array, weight_array, normal_array = arrays
    if nominal_array.ndim != 1 or nominal_array.size == 0:
        raise ValueError("nominal must be a non-empty one-dimensional array.")
    if any(array.shape != nominal_array.shape for array in arrays[1:]):
        raise ValueError("Projection arrays must have identical shapes.")
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("Projection arrays must contain finite values.")
    if np.any(lower_array > upper_array):
        raise ValueError("lower_bounds cannot exceed upper_bounds.")
    if np.any(weight_array <= 0.0):
        raise ValueError("weights must be strictly positive.")

    square_root_weights = np.sqrt(weight_array)
    scaled_command, feasible = _project_onto_box_halfspace(
        nominal_array * square_root_weights,
        lower_bounds=lower_array * square_root_weights,
        upper_bounds=upper_array * square_root_weights,
        constraint_normal=normal_array / square_root_weights,
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
    """Project one point onto a box intersected with one half-space."""

    clipped_nominal = np.clip(nominal, lower_bounds, upper_bounds)
    normal_squared = float(np.dot(constraint_normal, constraint_normal))
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
        return np.zeros_like(nominal, dtype=np.float64), False

    clipped_lhs = float(np.dot(constraint_normal, clipped_nominal))
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
        candidate_lhs = float(np.dot(constraint_normal, bounded_candidate))
        if candidate_lhs >= constraint_rhs - numerical_tolerance:
            candidates.append(bounded_candidate)

    # Each coordinate is either free (0), fixed to its lower bound (-1),
    # or fixed to its upper bound (+1). Projecting onto every box face and
    # retaining the feasible minimum gives the exact convex solution.
    dimension = nominal.size
    for states in product((-1, 0, 1), repeat=dimension):
        free_indices = np.array(
            [index for index, state in enumerate(states) if state == 0],
            dtype=np.int64,
        )
        fixed_indices = np.array(
            [index for index, state in enumerate(states) if state != 0],
            dtype=np.int64,
        )
        candidate = np.array(nominal, copy=True)
        for index in fixed_indices:
            candidate[index] = (
                lower_bounds[index] if states[index] < 0 else upper_bounds[index]
            )

        fixed_contribution = float(
            np.dot(
                constraint_normal[fixed_indices],
                candidate[fixed_indices],
            )
        )
        if free_indices.size:
            free_normal = constraint_normal[free_indices]
            free_normal_squared = float(np.dot(free_normal, free_normal))
            if free_normal_squared <= numerical_tolerance**2:
                if fixed_contribution < (constraint_rhs - numerical_tolerance):
                    continue
            else:
                free_nominal = nominal[free_indices]
                scale = (
                    constraint_rhs
                    - fixed_contribution
                    - float(np.dot(free_normal, free_nominal))
                ) / free_normal_squared
                candidate[free_indices] = free_nominal + scale * free_normal
        add_candidate(candidate)

    add_candidate(maximum_point)
    if not candidates:
        return np.zeros_like(nominal, dtype=np.float64), False

    objective_values = [
        float(np.sum(np.square(candidate - nominal))) for candidate in candidates
    ]
    return candidates[int(np.argmin(objective_values))], True

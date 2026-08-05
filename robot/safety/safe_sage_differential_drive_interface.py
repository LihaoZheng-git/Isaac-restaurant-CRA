"""Nova Carter control interface for the Safe-SAGE analytical filter.

The Safe-SAGE paper filters the reduced-order command
``(u_x, u_y, omega)``. A differential-drive robot cannot realize arbitrary
``u_x`` and ``u_y`` at its chassis origin. This module uses the standard
forward look-ahead point

    q_l = q + l [cos(psi), sin(psi)]

whose velocity is an exact linear function of the realizable command
``(v, omega)``. Pushing the paper's half-space through that map makes the
tangential LGF contribute to angular velocity without placing robot-specific
kinematics inside the paper-space filter.

The configuration-space footprint must use the same look-ahead point as its
reference. ``shift_rectangular_footprint_to_lookahead`` performs that change
without altering the physical rectangle.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

import numpy as np
from robot.safety.laplace_guidance_field import (
    LaplaceGuidanceFieldResult,
)
from robot.safety.safe_sage_cbf_filter import (
    SafeSAGECBFSafetyFilter,
    SafeSAGEConstraintEvaluation,
    project_weighted_box_halfspace,
)
from robot.safety.safe_sage_configuration_space import (
    RectangularRobotFootprint,
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
class SafeSAGEDifferentialDriveConfig:
    """Look-ahead geometry, actuator limits, and projection weights."""

    lookahead_distance: float = 0.10
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
                ("lookahead_distance", self.lookahead_distance),
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
                ("linear_velocity_weight", self.linear_velocity_weight),
                ("angular_velocity_weight", self.angular_velocity_weight),
                ("numerical_tolerance", self.numerical_tolerance),
                ("intervention_tolerance", self.intervention_tolerance),
            )
        }
        if values["lookahead_distance"] <= 0.0:
            raise ValueError("lookahead_distance must be greater than zero.")
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
            raise ValueError("Linear-velocity bounds must contain zero.")
        if not (
            values["minimum_angular_velocity"]
            <= 0.0
            <= values["maximum_angular_velocity"]
        ):
            raise ValueError("Angular-velocity bounds must contain zero.")
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
class SafeSAGEDifferentialDriveResult:
    """Realizable command plus paper-space and mapped diagnostics."""

    nominal_linear_velocity: float
    nominal_angular_velocity: float
    safe_linear_velocity: float
    safe_angular_velocity: float
    constraint: SafeSAGEConstraintEvaluation
    control_mapping: tuple[tuple[float, float], ...]
    linear_velocity_coefficient: float
    angular_velocity_coefficient: float
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
    def constraint_right_hand_side(self) -> float:
        return self.constraint.constraint_right_hand_side

    @property
    def query_on_unsafe_boundary(self) -> bool:
        return self.constraint.query_on_unsafe_boundary


def differential_drive_control_mapping(
    yaw_radians: float,
    *,
    lookahead_distance: float,
) -> np.ndarray:
    """Map ``(v, omega)`` to ``(q_l_dot_x, q_l_dot_y, psi_dot)``."""

    yaw = _validated_real(yaw_radians, name="yaw_radians")
    distance = _validated_real(
        lookahead_distance,
        name="lookahead_distance",
    )
    if distance <= 0.0:
        raise ValueError("lookahead_distance must be greater than zero.")
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return np.array(
        [
            [cosine, -distance * sine],
            [sine, distance * cosine],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )


def lookahead_position_grid_xy(
    chassis_position_grid_xy: np.ndarray,
    *,
    yaw_radians: float,
    lookahead_distance: float,
) -> np.ndarray:
    """Return the look-ahead control point in the field frame."""

    position = np.asarray(chassis_position_grid_xy, dtype=np.float64)
    if position.shape != (2,):
        raise ValueError("chassis_position_grid_xy must have shape (2,).")
    if not np.isfinite(position).all():
        raise ValueError("chassis_position_grid_xy must be finite.")
    yaw = _validated_real(yaw_radians, name="yaw_radians")
    distance = _validated_real(
        lookahead_distance,
        name="lookahead_distance",
    )
    if distance <= 0.0:
        raise ValueError("lookahead_distance must be greater than zero.")
    return np.ascontiguousarray(
        position
        + distance * np.array([math.cos(yaw), math.sin(yaw)]),
        dtype=np.float64,
    )


def shift_rectangular_footprint_to_lookahead(
    footprint: RectangularRobotFootprint,
    *,
    lookahead_distance: float,
) -> RectangularRobotFootprint:
    """Express one physical rectangle about its look-ahead point."""

    if not isinstance(footprint, RectangularRobotFootprint):
        raise TypeError("footprint must be a RectangularRobotFootprint.")
    distance = _validated_real(
        lookahead_distance,
        name="lookahead_distance",
    )
    if distance <= 0.0:
        raise ValueError("lookahead_distance must be greater than zero.")
    if distance > footprint.front_extent:
        raise ValueError(
            "lookahead_distance cannot exceed front_extent with the "
            "current rectangular-footprint representation."
        )
    return RectangularRobotFootprint(
        front_extent=footprint.front_extent - distance,
        rear_extent=footprint.rear_extent + distance,
        left_extent=footprint.left_extent,
        right_extent=footprint.right_extent,
        safety_margin=footprint.safety_margin,
    )


class SafeSAGEDifferentialDriveSafetyFilter:
    """Push Safe-SAGE Eq. (19) through look-ahead kinematics."""

    def __init__(
        self,
        paper_filter: SafeSAGECBFSafetyFilter,
        config: SafeSAGEDifferentialDriveConfig | None = None,
    ) -> None:
        if not isinstance(paper_filter, SafeSAGECBFSafetyFilter):
            raise TypeError("paper_filter must be a SafeSAGECBFSafetyFilter.")
        self.paper_filter = paper_filter
        self.config = config or SafeSAGEDifferentialDriveConfig()
        if not isinstance(self.config, SafeSAGEDifferentialDriveConfig):
            raise TypeError(
                "config must be a SafeSAGEDifferentialDriveConfig."
            )

    def filter_command(
        self,
        nominal_linear_velocity: float,
        nominal_angular_velocity: float,
        *,
        query_position_grid_xy: np.ndarray,
        query_yaw_radians: float,
        configuration_space: SafeSAGEConfigurationSpaceResult,
        h_full: SafeSAGEHFullResult,
        guidance_fields: Sequence[LaplaceGuidanceFieldResult],
        temporal_safety_derivative: float = 0.0,
    ) -> SafeSAGEDifferentialDriveResult:
        """Return the nearest realizable command satisfying Safe-SAGE."""

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
        constraint = self.paper_filter.evaluate_constraint(
            query_position_grid_xy=query_position_grid_xy,
            query_yaw_radians=query_yaw,
            configuration_space=configuration_space,
            h_full=h_full,
            guidance_fields=guidance_fields,
            temporal_safety_derivative=temporal_safety_derivative,
        )
        mapping = differential_drive_control_mapping(
            query_yaw,
            lookahead_distance=self.config.lookahead_distance,
        )
        paper_normal = np.asarray(
            constraint.constraint_normal,
            dtype=np.float64,
        )
        drive_normal = mapping.T @ paper_normal
        nominal_command = np.array(
            [nominal_linear, nominal_angular],
            dtype=np.float64,
        )
        if constraint.unsafe_reason is not None:
            return self._emergency_result(
                nominal_command,
                constraint=constraint,
                mapping=mapping,
                drive_normal=drive_normal,
                reason=constraint.unsafe_reason,
            )

        safe_command, feasible = project_weighted_box_halfspace(
            nominal_command,
            lower_bounds=np.array(
                [
                    self.config.minimum_linear_velocity,
                    self.config.minimum_angular_velocity,
                ],
                dtype=np.float64,
            ),
            upper_bounds=np.array(
                [
                    self.config.maximum_linear_velocity,
                    self.config.maximum_angular_velocity,
                ],
                dtype=np.float64,
            ),
            weights=np.array(
                [
                    self.config.linear_velocity_weight,
                    self.config.angular_velocity_weight,
                ],
                dtype=np.float64,
            ),
            constraint_normal=drive_normal,
            constraint_rhs=constraint.constraint_right_hand_side,
            numerical_tolerance=self.config.numerical_tolerance,
        )
        if not feasible:
            return self._emergency_result(
                nominal_command,
                constraint=constraint,
                mapping=mapping,
                drive_normal=drive_normal,
                reason="infeasible_cbf_constraint",
            )

        nominal_lhs = float(np.dot(drive_normal, nominal_command))
        safe_lhs = float(np.dot(drive_normal, safe_command))
        margin = safe_lhs - constraint.constraint_right_hand_side
        constraint_satisfied = (
            margin >= -self.config.numerical_tolerance
        )
        intervention = bool(
            np.any(
                np.abs(safe_command - nominal_command)
                > self.config.intervention_tolerance
            )
        )
        return SafeSAGEDifferentialDriveResult(
            nominal_linear_velocity=nominal_linear,
            nominal_angular_velocity=nominal_angular,
            safe_linear_velocity=float(safe_command[0]),
            safe_angular_velocity=float(safe_command[1]),
            constraint=constraint,
            control_mapping=tuple(
                tuple(float(value) for value in row)
                for row in mapping
            ),
            linear_velocity_coefficient=float(drive_normal[0]),
            angular_velocity_coefficient=float(drive_normal[1]),
            nominal_control_left_hand_side=nominal_lhs,
            safe_control_left_hand_side=safe_lhs,
            cbf_margin=margin,
            constraint_feasible=True,
            constraint_satisfied=constraint_satisfied,
            intervention_active=intervention,
            emergency_stop_active=False,
            emergency_reason=None,
        )

    def _emergency_result(
        self,
        nominal_command: np.ndarray,
        *,
        constraint: SafeSAGEConstraintEvaluation,
        mapping: np.ndarray,
        drive_normal: np.ndarray,
        reason: str,
    ) -> SafeSAGEDifferentialDriveResult:
        """Return a stationary fail-closed command with diagnostics."""

        nominal_lhs = float(np.dot(drive_normal, nominal_command))
        intervention = bool(
            np.any(
                np.abs(nominal_command)
                > self.config.intervention_tolerance
            )
        )
        return SafeSAGEDifferentialDriveResult(
            nominal_linear_velocity=float(nominal_command[0]),
            nominal_angular_velocity=float(nominal_command[1]),
            safe_linear_velocity=0.0,
            safe_angular_velocity=0.0,
            constraint=constraint,
            control_mapping=tuple(
                tuple(float(value) for value in row)
                for row in mapping
            ),
            linear_velocity_coefficient=float(drive_normal[0]),
            angular_velocity_coefficient=float(drive_normal[1]),
            nominal_control_left_hand_side=nominal_lhs,
            safe_control_left_hand_side=0.0,
            cbf_margin=-constraint.constraint_right_hand_side,
            constraint_feasible=False,
            constraint_satisfied=False,
            intervention_active=intervention,
            emergency_stop_active=True,
            emergency_reason=reason,
        )

"""Class-dependent boundary policy for the tangential-LGF milestone.

The policy materializes two independent parts of a Safe-SAGE-style Laplace
boundary condition:

    obstacle boundary:  v = b n
    buffered interface: v = lambda tau

Normalized semantic risk is mapped monotonically to the magnitude of the
negative normal flux ``b``. Tangential strength is configured separately
because it describes passing behavior rather than danger alone. Passing side
is represented explicitly instead of hiding the convention inside the sign
of ``lambda``.

This module only defines and validates policy data. It does not construct a
buffered interface, orient tangent vectors, solve a Laplace equation, or
propagate semantics into a Poisson forcing function.
"""

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from numbers import Real
from pathlib import Path
from types import MappingProxyType

from robot.safety.class_safety_policy import (
    ClassSafetyParameters,
    LinearRiskFluxMapping,
)
from robot.safety.semantic_obstacle import (
    SemanticObstacle,
    normalize_semantic_name,
)

SUPPORTED_SCHEMA_VERSION = 1


def _validated_real(
    value: object,
    *,
    name: str,
) -> float:
    """Return ``value`` as a finite float, rejecting booleans and strings."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")

    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")

    return result


class PassingSide(str, Enum):
    """Supported obstacle-passing conventions.

    The tangential field solver will translate this semantic choice into an
    oriented unit tangent in robot-frame coordinates.
    """

    LEFT = "left"
    RIGHT = "right"

    @classmethod
    def from_value(cls, value: object) -> "PassingSide":
        """Return a normalized passing side from a string or enum member."""

        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError("passing_side must be a string.")

        normalized = value.strip().casefold()
        try:
            return cls(normalized)
        except ValueError as error:
            allowed = ", ".join(side.value for side in cls)
            raise ValueError(
                f"passing_side must be one of: {allowed}."
            ) from error


@dataclass(frozen=True)
class SafeSAGEGuidanceParameters:
    """Materialized normal and tangential parameters for one class."""

    safety_parameters: ClassSafetyParameters
    tangential_strength: float
    passing_side: PassingSide | str

    def __post_init__(self) -> None:
        if not isinstance(
            self.safety_parameters,
            ClassSafetyParameters,
        ):
            raise TypeError(
                "safety_parameters must be a ClassSafetyParameters instance."
            )

        strength = _validated_real(
            self.tangential_strength,
            name="tangential_strength",
        )
        if strength < 0.0:
            raise ValueError(
                "tangential_strength must be greater than or equal to zero."
            )

        side = PassingSide.from_value(self.passing_side)

        object.__setattr__(self, "tangential_strength", strength)
        object.__setattr__(self, "passing_side", side)

    @property
    def risk(self) -> float:
        """Return the normalized semantic risk."""

        return self.safety_parameters.risk

    @property
    def boundary_flux_magnitude(self) -> float:
        """Return the magnitude of the normal obstacle-boundary flux."""

        return self.safety_parameters.boundary_flux_magnitude

    @property
    def boundary_flux(self) -> float:
        """Return the signed negative normal obstacle-boundary flux."""

        return self.safety_parameters.boundary_flux

    @property
    def has_tangential_guidance(self) -> bool:
        """Return whether this class requests a tangential boundary term."""

        return self.tangential_strength > 0.0


class SafeSAGEGuidancePolicy:
    """Immutable lookup for class-dependent normal and tangential data."""

    def __init__(
        self,
        class_parameters: Mapping[str, SafeSAGEGuidanceParameters],
        *,
        unknown_class: str = "unknown",
        risk_flux_mapping: LinearRiskFluxMapping | None = None,
    ) -> None:
        if not isinstance(class_parameters, Mapping):
            raise TypeError("class_parameters must be a mapping.")
        if not class_parameters:
            raise ValueError("class_parameters cannot be empty.")

        normalized_parameters: dict[
            str,
            SafeSAGEGuidanceParameters,
        ] = {}
        for raw_class, parameters in class_parameters.items():
            semantic_class = normalize_semantic_name(
                raw_class,
                name="semantic class",
            )
            if semantic_class in normalized_parameters:
                raise ValueError(
                    "class_parameters contains duplicate normalized class "
                    f"'{semantic_class}'."
                )
            if not isinstance(parameters, SafeSAGEGuidanceParameters):
                raise TypeError(
                    "Every class parameter must be a "
                    "SafeSAGEGuidanceParameters instance."
                )

            normalized_parameters[semantic_class] = parameters

        normalized_unknown = normalize_semantic_name(
            unknown_class,
            name="unknown_class",
        )
        if normalized_unknown not in normalized_parameters:
            raise ValueError(
                "unknown_class must have an entry in class_parameters."
            )
        if (
            risk_flux_mapping is not None
            and not isinstance(
                risk_flux_mapping,
                LinearRiskFluxMapping,
            )
        ):
            raise TypeError(
                "risk_flux_mapping must be a LinearRiskFluxMapping."
            )

        self._class_parameters = MappingProxyType(normalized_parameters)
        self._unknown_class = normalized_unknown
        self._risk_flux_mapping = risk_flux_mapping

    @property
    def class_parameters(
        self,
    ) -> Mapping[str, SafeSAGEGuidanceParameters]:
        """Return the read-only class-to-parameter mapping."""

        return self._class_parameters

    @property
    def known_classes(self) -> tuple[str, ...]:
        """Return configured semantic classes in deterministic order."""

        return tuple(sorted(self._class_parameters))

    @property
    def unknown_class(self) -> str:
        """Return the class used for unconfigured semantic labels."""

        return self._unknown_class

    @property
    def risk_flux_mapping(
        self,
    ) -> LinearRiskFluxMapping | None:
        """Return the configured risk-to-normal-flux mapping."""

        return self._risk_flux_mapping

    def parameters_for(
        self,
        semantic_class: str,
    ) -> SafeSAGEGuidanceParameters:
        """Return guidance parameters for a class or the fallback."""

        normalized_class = normalize_semantic_name(
            semantic_class,
            name="semantic_class",
        )
        return self._class_parameters.get(
            normalized_class,
            self._class_parameters[self._unknown_class],
        )

    def parameters_for_obstacle(
        self,
        obstacle: SemanticObstacle,
    ) -> SafeSAGEGuidanceParameters:
        """Return guidance parameters for one semantic obstacle."""

        if not isinstance(obstacle, SemanticObstacle):
            raise TypeError("obstacle must be a SemanticObstacle.")

        return self.parameters_for(obstacle.semantic_class)

    @classmethod
    def from_json(
        cls,
        path: str | Path,
    ) -> "SafeSAGEGuidancePolicy":
        """Load and strictly validate a Safe-SAGE guidance policy."""

        config_path = Path(path)
        try:
            raw_config = json.loads(
                config_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON in Safe-SAGE guidance policy: {config_path}"
            ) from error

        if not isinstance(raw_config, dict):
            raise TypeError("Safe-SAGE guidance policy root must be an object.")

        required_root_fields = {
            "schema_version",
            "unknown_class",
            "risk_flux_mapping",
            "classes",
        }
        _validate_exact_fields(
            raw_config,
            required_root_fields,
            context="Safe-SAGE guidance policy root",
        )

        schema_version = raw_config["schema_version"]
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version != SUPPORTED_SCHEMA_VERSION
        ):
            raise ValueError(
                "Unsupported Safe-SAGE guidance policy schema_version; "
                f"expected {SUPPORTED_SCHEMA_VERSION}."
            )

        unknown_class = raw_config["unknown_class"]
        if not isinstance(unknown_class, str):
            raise TypeError("unknown_class must be a string.")

        raw_mapping = raw_config["risk_flux_mapping"]
        risk_flux_mapping = _load_risk_flux_mapping(raw_mapping)

        raw_classes = raw_config["classes"]
        if not isinstance(raw_classes, dict):
            raise TypeError("classes must be an object.")

        class_parameters: dict[str, SafeSAGEGuidanceParameters] = {}
        for semantic_class, raw_parameters in raw_classes.items():
            if not isinstance(semantic_class, str):
                raise TypeError("Every class name must be a string.")
            if not isinstance(raw_parameters, dict):
                raise TypeError(
                    f"Parameters for '{semantic_class}' must be an object."
                )

            required_class_fields = {
                "risk",
                "tangential_strength",
                "passing_side",
            }
            _validate_exact_fields(
                raw_parameters,
                required_class_fields,
                context=f"class '{semantic_class}'",
            )

            risk = _validated_real(
                raw_parameters["risk"],
                name=f"risk for '{semantic_class}'",
            )
            safety_parameters = ClassSafetyParameters(
                risk=risk,
                boundary_flux_magnitude=(
                    risk_flux_mapping.magnitude_for_risk(risk)
                ),
            )
            class_parameters[semantic_class] = SafeSAGEGuidanceParameters(
                safety_parameters=safety_parameters,
                tangential_strength=raw_parameters["tangential_strength"],
                passing_side=raw_parameters["passing_side"],
            )

        return cls(
            class_parameters,
            unknown_class=unknown_class,
            risk_flux_mapping=risk_flux_mapping,
        )


def _validate_exact_fields(
    raw_mapping: Mapping[object, object],
    required_fields: set[str],
    *,
    context: str,
) -> None:
    """Reject missing and unexpected JSON object fields."""

    actual_fields = set(raw_mapping)
    if actual_fields == required_fields:
        return

    missing = sorted(required_fields - actual_fields)
    unexpected = sorted(actual_fields - required_fields)
    raise ValueError(
        f"Invalid fields for {context}: "
        f"missing={missing}, unexpected={unexpected}."
    )


def _load_risk_flux_mapping(
    raw_mapping: object,
) -> LinearRiskFluxMapping:
    """Return a validated linear normal-flux mapping from JSON data."""

    if not isinstance(raw_mapping, dict):
        raise TypeError("risk_flux_mapping must be an object.")

    required_fields = {
        "type",
        "minimum_magnitude",
        "maximum_magnitude",
    }
    _validate_exact_fields(
        raw_mapping,
        required_fields,
        context="risk_flux_mapping",
    )
    if raw_mapping["type"] != "linear":
        raise ValueError("risk_flux_mapping type must be 'linear'.")

    return LinearRiskFluxMapping(
        minimum_magnitude=raw_mapping["minimum_magnitude"],
        maximum_magnitude=raw_mapping["maximum_magnitude"],
    )

"""Semantic risk policy for the Risk-Aware PSF/LGF CBF baseline.

The production JSON configuration assigns a normalized risk to each semantic
class and maps that risk to a Laplace boundary-flux magnitude through one
global monotone linear map:

    semantic class -> risk -> boundary flux

This makes the relationship used by the risk-aware baseline explicit and
prevents independently configured risk and flux values from disagreeing.
The materialized class parameters retain both values for efficient grid
rasterization. Negative signed flux is consumed by the normal-only Laplace
guidance field; the Poisson safety function is solved independently.
"""

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from types import MappingProxyType

from robot.safety.semantic_obstacle import (
    SemanticObstacle,
    normalize_semantic_name,
)


SUPPORTED_SCHEMA_VERSION = 2


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


@dataclass(frozen=True)
class LinearRiskFluxMapping:
    """Monotone linear map from normalized risk to flux magnitude."""

    minimum_magnitude: float
    maximum_magnitude: float

    def __post_init__(self) -> None:
        minimum = _validated_real(
            self.minimum_magnitude,
            name="minimum_magnitude",
        )
        maximum = _validated_real(
            self.maximum_magnitude,
            name="maximum_magnitude",
        )

        if minimum <= 0.0:
            raise ValueError(
                "minimum_magnitude must be greater than zero."
            )
        if maximum < minimum:
            raise ValueError(
                "maximum_magnitude cannot be smaller than "
                "minimum_magnitude."
            )

        object.__setattr__(self, "minimum_magnitude", minimum)
        object.__setattr__(self, "maximum_magnitude", maximum)

    def magnitude_for_risk(self, risk: float) -> float:
        """Return the flux magnitude corresponding to one risk value."""

        normalized_risk = _validated_real(risk, name="risk")
        if not 0.0 <= normalized_risk <= 1.0:
            raise ValueError("risk must be in [0, 1].")

        return self.minimum_magnitude + normalized_risk * (
            self.maximum_magnitude - self.minimum_magnitude
        )


@dataclass(frozen=True)
class ClassSafetyParameters:
    """Materialized risk-aware parameters for one semantic class."""

    risk: float
    boundary_flux_magnitude: float

    def __post_init__(self) -> None:
        risk = _validated_real(self.risk, name="risk")
        flux_magnitude = _validated_real(
            self.boundary_flux_magnitude,
            name="boundary_flux_magnitude",
        )

        if not 0.0 <= risk <= 1.0:
            raise ValueError("risk must be in [0, 1].")
        if flux_magnitude <= 0.0:
            raise ValueError(
                "boundary_flux_magnitude must be greater than zero."
            )

        object.__setattr__(self, "risk", risk)
        object.__setattr__(
            self,
            "boundary_flux_magnitude",
            flux_magnitude,
        )

    @property
    def boundary_flux(self) -> float:
        """Return the signed Laplace boundary flux for this class."""

        return -self.boundary_flux_magnitude


class ClassSafetyPolicy:
    """Immutable lookup from semantic classes to fixed safety parameters."""

    def __init__(
        self,
        class_parameters: Mapping[str, ClassSafetyParameters],
        *,
        unknown_class: str = "unknown",
        risk_flux_mapping: LinearRiskFluxMapping | None = None,
    ) -> None:
        if not isinstance(class_parameters, Mapping):
            raise TypeError("class_parameters must be a mapping.")
        if not class_parameters:
            raise ValueError("class_parameters cannot be empty.")

        normalized_parameters: dict[str, ClassSafetyParameters] = {}
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
            if not isinstance(parameters, ClassSafetyParameters):
                raise TypeError(
                    "Every class parameter must be a "
                    "ClassSafetyParameters instance."
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

        self._class_parameters = MappingProxyType(
            normalized_parameters
        )
        self._unknown_class = normalized_unknown
        self._risk_flux_mapping = risk_flux_mapping

    @property
    def class_parameters(
        self,
    ) -> Mapping[str, ClassSafetyParameters]:
        """Return the read-only class-to-parameter mapping."""

        return self._class_parameters

    @property
    def known_classes(self) -> tuple[str, ...]:
        """Return configured semantic classes in deterministic order."""

        return tuple(sorted(self._class_parameters))

    @property
    def unknown_class(self) -> str:
        """Return the class used when a requested label is not configured."""

        return self._unknown_class

    @property
    def risk_flux_mapping(
        self,
    ) -> LinearRiskFluxMapping | None:
        """Return the production risk-to-flux map, when configured."""

        return self._risk_flux_mapping

    def parameters_for(
        self,
        semantic_class: str,
    ) -> ClassSafetyParameters:
        """Return parameters for a class or the conservative fallback."""

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
    ) -> ClassSafetyParameters:
        """Return parameters for one semantically labeled obstacle."""

        if not isinstance(obstacle, SemanticObstacle):
            raise TypeError("obstacle must be a SemanticObstacle.")

        return self.parameters_for(obstacle.semantic_class)

    @classmethod
    def from_json(
        cls,
        path: str | Path,
    ) -> "ClassSafetyPolicy":
        """Load and validate a class safety policy from a JSON file."""

        config_path = Path(path)
        try:
            raw_config = json.loads(
                config_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON in class safety policy: {config_path}"
            ) from error

        if not isinstance(raw_config, dict):
            raise TypeError("Class safety policy root must be an object.")

        schema_version = raw_config.get("schema_version")
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version != SUPPORTED_SCHEMA_VERSION
        ):
            raise ValueError(
                "Unsupported class safety policy schema_version; "
                f"expected {SUPPORTED_SCHEMA_VERSION}."
            )

        unknown_class = raw_config.get("unknown_class")
        if not isinstance(unknown_class, str):
            raise TypeError("unknown_class must be a string.")

        raw_classes = raw_config.get("classes")
        if not isinstance(raw_classes, dict):
            raise TypeError("classes must be an object.")

        raw_mapping = raw_config.get("risk_flux_mapping")
        if not isinstance(raw_mapping, dict):
            raise TypeError("risk_flux_mapping must be an object.")
        required_mapping_fields = {
            "type",
            "minimum_magnitude",
            "maximum_magnitude",
        }
        actual_mapping_fields = set(raw_mapping)
        if actual_mapping_fields != required_mapping_fields:
            missing = sorted(
                required_mapping_fields - actual_mapping_fields
            )
            unexpected = sorted(
                actual_mapping_fields - required_mapping_fields
            )
            raise ValueError(
                "Invalid risk_flux_mapping fields: "
                f"missing={missing}, unexpected={unexpected}."
            )
        if raw_mapping["type"] != "linear":
            raise ValueError(
                "risk_flux_mapping type must be 'linear'."
            )
        risk_flux_mapping = LinearRiskFluxMapping(
            minimum_magnitude=raw_mapping["minimum_magnitude"],
            maximum_magnitude=raw_mapping["maximum_magnitude"],
        )

        class_parameters: dict[str, ClassSafetyParameters] = {}
        for semantic_class, raw_parameters in raw_classes.items():
            if not isinstance(semantic_class, str):
                raise TypeError("Every class name must be a string.")
            if not isinstance(raw_parameters, dict):
                raise TypeError(
                    f"Parameters for '{semantic_class}' must be an object."
                )

            required_fields = {"risk"}
            actual_fields = set(raw_parameters)
            if actual_fields != required_fields:
                missing = sorted(required_fields - actual_fields)
                unexpected = sorted(actual_fields - required_fields)
                raise ValueError(
                    f"Invalid fields for class '{semantic_class}': "
                    f"missing={missing}, unexpected={unexpected}."
                )

            risk = _validated_real(
                raw_parameters["risk"],
                name=f"risk for '{semantic_class}'",
            )
            class_parameters[semantic_class] = ClassSafetyParameters(
                risk=risk,
                boundary_flux_magnitude=(
                    risk_flux_mapping.magnitude_for_risk(risk)
                ),
            )

        return cls(
            class_parameters,
            unknown_class=unknown_class,
            risk_flux_mapping=risk_flux_mapping,
        )

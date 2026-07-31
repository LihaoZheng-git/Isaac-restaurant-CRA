"""Fixed semantic-class policy for the PSF class-aware CBF baseline.

This module converts a semantic obstacle label into the two values needed by
the later Poisson safety-function pipeline:

* a normalized class risk in ``[0, 1]``;
* a positive boundary-flux magnitude.

The configuration stores flux as a positive magnitude for readability.  The
``boundary_flux`` property returns its negative signed value because the PSF
boundary condition uses negative flux to point the guidance field away from
an obstacle.
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


@dataclass(frozen=True)
class ClassSafetyParameters:
    """Fixed PSF parameters assigned to one semantic class."""

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
        """Return the signed PSF boundary flux for this class."""

        return -self.boundary_flux_magnitude


class ClassSafetyPolicy:
    """Immutable lookup from semantic classes to fixed safety parameters."""

    def __init__(
        self,
        class_parameters: Mapping[str, ClassSafetyParameters],
        *,
        unknown_class: str = "unknown",
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

        self._class_parameters = MappingProxyType(
            normalized_parameters
        )
        self._unknown_class = normalized_unknown

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

        class_parameters: dict[str, ClassSafetyParameters] = {}
        for semantic_class, raw_parameters in raw_classes.items():
            if not isinstance(semantic_class, str):
                raise TypeError("Every class name must be a string.")
            if not isinstance(raw_parameters, dict):
                raise TypeError(
                    f"Parameters for '{semantic_class}' must be an object."
                )

            required_fields = {
                "risk",
                "boundary_flux_magnitude",
            }
            actual_fields = set(raw_parameters)
            if actual_fields != required_fields:
                missing = sorted(required_fields - actual_fields)
                unexpected = sorted(actual_fields - required_fields)
                raise ValueError(
                    f"Invalid fields for class '{semantic_class}': "
                    f"missing={missing}, unexpected={unexpected}."
                )

            class_parameters[semantic_class] = ClassSafetyParameters(
                risk=raw_parameters["risk"],
                boundary_flux_magnitude=raw_parameters[
                    "boundary_flux_magnitude"
                ],
            )

        return cls(
            class_parameters,
            unknown_class=unknown_class,
        )

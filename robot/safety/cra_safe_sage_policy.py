"""Translate contextual risk assessments into Safe-SAGE field parameters.

The CRA mathematical core deliberately knows nothing about Safe-SAGE.  This
adapter is the explicit boundary between the two models:

``PhysicalRiskProfile -> CRAAssessment -> SafeSAGEGuidanceParameters``

The conservative CRA danger index ``phi`` is mapped monotonically to the
magnitude of the negative normal boundary flux ``b``.  Tangential strength
``lambda`` has a separate endpoint map because it controls passing behavior
and is not equivalent to danger.  The default tangential endpoints are equal,
which reproduces the repository's paper-baseline choice of a fixed
``lambda = 1.25``.  Experiments may configure a monotone risk-dependent map
without changing the field or controller interfaces.

This module does not assign risk from vision, choose passing side from free
space, construct a field, or execute a command.  Numerical endpoint values are
configurable research hypotheses and must be calibrated and ablated before
they are reported as experimental findings.
"""

from dataclasses import dataclass
import math
from numbers import Real

from robot.risk.cra import (
    CRAAssessment,
    CRARiskAssessor,
    PhysicalRiskProfile,
)
from robot.safety.class_safety_policy import ClassSafetyParameters
from robot.safety.safe_sage_guidance_policy import (
    PassingSide,
    SafeSAGEGuidanceParameters,
)


def _finite_real(value: object, *, name: str) -> float:
    """Return one finite real number while rejecting booleans."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _normalized_risk(value: object, *, name: str = "risk") -> float:
    """Return one finite normalized risk value."""

    result = _finite_real(value, name=name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1].")
    return result


@dataclass(frozen=True)
class CRASafeSAGEPolicyConfig:
    """Endpoint policy for converting CRA danger into Safe-SAGE data.

    Flux defaults match ``configs/safe_sage_paper_baseline.json``.  Equal
    tangential endpoints intentionally keep lambda independent of CRA risk,
    as in that repository baseline.
    """

    normal_flux_magnitude_at_zero_risk: float = 0.5
    normal_flux_magnitude_at_unit_risk: float = 1.7
    tangential_strength_at_zero_risk: float = 1.25
    tangential_strength_at_unit_risk: float = 1.25
    passing_side: PassingSide | str = PassingSide.RIGHT

    def __post_init__(self) -> None:
        zero_flux = _finite_real(
            self.normal_flux_magnitude_at_zero_risk,
            name="normal_flux_magnitude_at_zero_risk",
        )
        unit_flux = _finite_real(
            self.normal_flux_magnitude_at_unit_risk,
            name="normal_flux_magnitude_at_unit_risk",
        )
        zero_tangent = _finite_real(
            self.tangential_strength_at_zero_risk,
            name="tangential_strength_at_zero_risk",
        )
        unit_tangent = _finite_real(
            self.tangential_strength_at_unit_risk,
            name="tangential_strength_at_unit_risk",
        )
        if zero_flux <= 0.0:
            raise ValueError("normal_flux_magnitude_at_zero_risk must be positive.")
        if unit_flux < zero_flux:
            raise ValueError(
                "normal flux magnitude cannot decrease as CRA risk increases."
            )
        if zero_tangent < 0.0:
            raise ValueError("tangential_strength_at_zero_risk cannot be negative.")
        if unit_tangent < zero_tangent:
            raise ValueError(
                "tangential strength cannot decrease as CRA risk increases."
            )
        side = PassingSide.from_value(self.passing_side)
        object.__setattr__(
            self,
            "normal_flux_magnitude_at_zero_risk",
            zero_flux,
        )
        object.__setattr__(
            self,
            "normal_flux_magnitude_at_unit_risk",
            unit_flux,
        )
        object.__setattr__(
            self,
            "tangential_strength_at_zero_risk",
            zero_tangent,
        )
        object.__setattr__(
            self,
            "tangential_strength_at_unit_risk",
            unit_tangent,
        )
        object.__setattr__(self, "passing_side", side)

    def normal_flux_magnitude_for_risk(self, risk: float) -> float:
        """Return the monotone linearly interpolated normal-flux magnitude."""

        phi = _normalized_risk(risk)
        return self.normal_flux_magnitude_at_zero_risk + phi * (
            self.normal_flux_magnitude_at_unit_risk
            - self.normal_flux_magnitude_at_zero_risk
        )

    def tangential_strength_for_risk(self, risk: float) -> float:
        """Return the independently configured tangential strength."""

        phi = _normalized_risk(risk)
        return self.tangential_strength_at_zero_risk + phi * (
            self.tangential_strength_at_unit_risk
            - self.tangential_strength_at_zero_risk
        )


@dataclass(frozen=True)
class CRASafeSAGEPolicyResult:
    """One CRA assessment and the Safe-SAGE parameters derived from it."""

    assessment: CRAAssessment
    guidance_parameters: SafeSAGEGuidanceParameters

    def __post_init__(self) -> None:
        if not isinstance(self.assessment, CRAAssessment):
            raise TypeError("assessment must be a CRAAssessment.")
        if not isinstance(
            self.guidance_parameters,
            SafeSAGEGuidanceParameters,
        ):
            raise TypeError("guidance_parameters must be SafeSAGEGuidanceParameters.")
        if not math.isclose(
            self.assessment.phi,
            self.guidance_parameters.risk,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("Safe-SAGE risk must equal the conservative CRA phi.")

    @property
    def risk(self) -> float:
        """Return the conservative CRA risk supplied to Safe-SAGE."""

        return self.assessment.phi

    @property
    def boundary_flux(self) -> float:
        """Return the signed Safe-SAGE normal boundary flux."""

        return self.guidance_parameters.boundary_flux

    @property
    def tangential_strength(self) -> float:
        """Return the Safe-SAGE tangential strength."""

        return self.guidance_parameters.tangential_strength

    @property
    def passing_side(self) -> PassingSide:
        """Return the configured obstacle-passing side."""

        return self.guidance_parameters.passing_side

    def as_dict(self) -> dict[str, object]:
        """Return versioned CRA diagnostics plus Safe-SAGE parameters."""

        return {
            "cra": self.assessment.as_dict(),
            "safe_sage": {
                "risk": self.risk,
                "boundary_flux_magnitude": (
                    self.guidance_parameters.boundary_flux_magnitude
                ),
                "boundary_flux": self.boundary_flux,
                "tangential_strength": self.tangential_strength,
                "passing_side": self.passing_side.value,
            },
        }


class CRASafeSAGEPolicy:
    """Assess one physical-risk profile and materialize Safe-SAGE data."""

    def __init__(
        self,
        *,
        risk_assessor: CRARiskAssessor | None = None,
        config: CRASafeSAGEPolicyConfig | None = None,
    ) -> None:
        resolved_assessor = (
            CRARiskAssessor() if risk_assessor is None else risk_assessor
        )
        resolved_config = CRASafeSAGEPolicyConfig() if config is None else config
        if not isinstance(resolved_assessor, CRARiskAssessor):
            raise TypeError("risk_assessor must be a CRARiskAssessor.")
        if not isinstance(resolved_config, CRASafeSAGEPolicyConfig):
            raise TypeError("config must be a CRASafeSAGEPolicyConfig.")
        self.risk_assessor = resolved_assessor
        self.config = resolved_config

    def evaluate(self, profile: PhysicalRiskProfile) -> CRASafeSAGEPolicyResult:
        """Assess a physical-risk profile and return Safe-SAGE parameters."""

        if not isinstance(profile, PhysicalRiskProfile):
            raise TypeError("profile must be a PhysicalRiskProfile.")
        return self.parameters_for_assessment(self.risk_assessor.assess(profile))

    def parameters_for_assessment(
        self,
        assessment: CRAAssessment,
    ) -> CRASafeSAGEPolicyResult:
        """Map an existing CRA assessment without evaluating it again."""

        if not isinstance(assessment, CRAAssessment):
            raise TypeError("assessment must be a CRAAssessment.")
        risk = assessment.phi
        safety_parameters = ClassSafetyParameters(
            risk=risk,
            boundary_flux_magnitude=(self.config.normal_flux_magnitude_for_risk(risk)),
        )
        guidance_parameters = SafeSAGEGuidanceParameters(
            safety_parameters=safety_parameters,
            tangential_strength=(self.config.tangential_strength_for_risk(risk)),
            passing_side=self.config.passing_side,
        )
        return CRASafeSAGEPolicyResult(
            assessment=assessment,
            guidance_parameters=guidance_parameters,
        )

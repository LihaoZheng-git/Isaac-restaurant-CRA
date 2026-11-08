"""Opt-in engineering calibration for verified restaurant object evidence.

This module changes the scoring hypothesis, not the physical estimates. It
keeps source severity, interaction, confidence, priors and evidence intact in
diagnostics. It is not a probability of injury or a safety guarantee.

Supported factors use their existing upper-bound severity, a bounded contextual
modulation and a proportional uncertainty reserve. Missing-factor priors form
one shared baseline (their maximum), explicitly treating their uncertainty as
correlated for this experiment. Supported hazards combine by exponent-weighted
noisy-OR. Unassessed objects keep the original policy. A separately verified
human may receive an explicit protection floor without changing physical cues.
"""

from dataclasses import asdict, dataclass, field, replace
import math
from numbers import Real
from typing import Mapping

from robot.risk.cra import PhysicalRiskFactor, RiskFactorEstimate
from robot.perception.physical_attributes import ScalarEstimate, EstimateStatus
from robot.safety.cra_safe_sage_policy import CRASafeSAGEPolicyResult


FACTORS = ("thermal", "contact", "potential", "kinetic")
CALIBRATION_SOURCE = "restaurant_verified_object_calibration_v1"
CONTACT_SOURCES = {"visual_silhouette_contact_proxy", "visual_exposed_edge_geometry_proxy"}
CONTACT_BASES = {"visual_silhouette_geometry_proxy", "visual_exposed_edge_proxy"}


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _unit(value, name):
    value = _finite(value, name)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


@dataclass(frozen=True)
class RestaurantFactorWeights:
    """Positive hazard exponents, not percentages or a learned importance map."""

    thermal: float = 1.0
    contact: float = 1.0
    potential: float = 1.0
    kinetic: float = 1.0

    def __post_init__(self):
        for name in FACTORS:
            value = _finite(getattr(self, name), name)
            if not 0 < value <= 4:
                raise ValueError("factor exponents must be in (0, 4]")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class RestaurantCalibrationConfig:
    """Candidate parameters; explicit opt-in is required at the call site.

    Energy scales are uncalibrated engineering reference energies. Existing
    severity can be rescaled exactly only when the new scale is no larger than
    the explicitly supplied original scale, because the old adapter clipped at
    one. Confidence reserve is at most 50% of the supported factor value.
    """

    enabled: bool = False
    potential_energy_scale_j: float = 100.0
    kinetic_energy_scale_j: float = 50.0
    contact_power: float = 2.0
    context_floor: float = 0.75
    uncertainty_gain: float = 0.25
    weights: RestaurantFactorWeights = field(default_factory=RestaurantFactorWeights)
    human_protection_floor: float = 0.8
    normal_flux_magnitude_at_unit_risk: float = 1.7

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be bool")
        if not isinstance(self.weights, RestaurantFactorWeights):
            raise TypeError("weights must be RestaurantFactorWeights")
        for name in ("potential_energy_scale_j", "kinetic_energy_scale_j"):
            value = _finite(getattr(self, name), name)
            if value <= 0:
                raise ValueError("energy scales must be positive")
            object.__setattr__(self, name, value)
        power = _finite(self.contact_power, "contact_power")
        if not 1 <= power <= 3:
            raise ValueError("contact_power must be in [1, 3]")
        object.__setattr__(self, "contact_power", power)
        for name in ("context_floor", "human_protection_floor"):
            object.__setattr__(self, name, _unit(getattr(self, name), name))
        gain = _finite(self.uncertainty_gain, "uncertainty_gain")
        if not 0 <= gain <= 0.5:
            raise ValueError("uncertainty_gain must be in [0, .5]")
        object.__setattr__(self, "uncertainty_gain", gain)
        flux = _finite(self.normal_flux_magnitude_at_unit_risk,
                       "normal_flux_magnitude_at_unit_risk")
        if flux < 1.7:
            raise ValueError("restaurant boundary response cannot weaken the original 1.7 endpoint")
        if not self.enabled and flux != 1.7:
            raise ValueError("alternative restaurant boundary response requires enabled calibration")
        object.__setattr__(self, "normal_flux_magnitude_at_unit_risk", flux)

    def normal_flux_magnitude_for_risk(self, risk):
        """Map the unchanged normalized score to a separate field strength.

        The zero-risk endpoint stays 0.5. Only the verified/retained-object
        calibration path uses this map; ordinary unknown policies are untouched.
        """
        return .5 + (self.normal_flux_magnitude_at_unit_risk - .5) * _unit(risk, "risk")


@dataclass(frozen=True)
class CalibratedFactor:
    factor: str
    supported: bool
    mode: str
    basis: str
    original_severity: float
    original_interaction: float
    original_confidence: float
    original_factor_score: float
    fallback_risk: float
    normalized_severity: float | None
    context_multiplier: float | None
    confidence_multiplier: float | None
    score: float
    exponent: float
    source: str
    evidence: tuple[str, ...]
    input_rejections: tuple[str, ...]
    original_energy_scale_j: float | None = None
    energy_scale_j: float | None = None
    calibration_confidence: float | None = None
    contact_proxy: dict | None = None
    contact_context_basis: str | None = None
    contact_proxy_rejections: tuple[str, ...] = ()
    calibration_mode: str = "original_factor_policy"


@dataclass(frozen=True)
class RestaurantCalibrationAssessment:
    config: RestaurantCalibrationConfig
    eligible: bool
    verified_human: bool
    reason: str
    original_phi: float
    physical_phi: float
    phi: float
    prior_baseline: float | None
    supported_hazard: float | None
    factors: tuple[CalibratedFactor, ...]

    def as_dict(self):
        result = asdict(self)
        result.update(
            source=CALIBRATION_SOURCE,
            factors={item.factor: asdict(item) for item in self.factors},
            human_protection_prior=(self.config.human_protection_floor
                                    if self.config.enabled and self.verified_human else None),
            human_prior_uplift=self.phi - self.physical_phi,
            uncertainty_scope="maximum missing-factor prior as one correlated engineering baseline",
            aggregation="prior + (1-prior) * (1-product((1-supported_score)**positive_exponent))",
            scope="experimental normalized consequence policy; original evidence and source clocks retained",
        )
        return result


def evaluate_restaurant_calibration(
    cra: Mapping,
    factor_details: Mapping,
    config: RestaurantCalibrationConfig,
    *,
    eligible: bool,
    verified_human: bool = False,
    source_energy_scales_j: Mapping | None = None,
    contact_proxy: ScalarEstimate | Mapping | None = None,
    contact_interaction: ScalarEstimate | Mapping | None = None,
    timestamp_seconds: float | None = None,
    maximum_contact_age_seconds: float | None = None,
    maximum_interaction_age_seconds: float = 1.0,
    prior_confidence_cap: float = 0.5,
) -> RestaurantCalibrationAssessment:
    """Evaluate the candidate from the original serialized factor assessment.

    Eligibility is established by the caller's accepted image/LiDAR ownership
    or qualified retained evidence; no semantic object name is consulted here.
    Human verification is a separate current association/identity contract.
    The original energy reference scales must come from that assessment's
    adapter configuration, not from this module's candidate parameters.
    """
    if not isinstance(config, RestaurantCalibrationConfig):
        raise TypeError("config must be RestaurantCalibrationConfig")
    if not isinstance(eligible, bool) or not isinstance(verified_human, bool):
        raise TypeError("eligibility and human verification must be bool")
    original_phi = _unit(cra["phi"], "original phi")
    reason = "disabled" if not config.enabled else "unassessed_object_policy_preserved"
    physical_phi, prior, hazard, factors = original_phi, None, None, []
    if config.enabled and eligible:
        for name in FACTORS:
            original = cra["factors"][name]
            detail = factor_details[name]
            severity = _unit(original["severity"], f"{name} severity")
            interaction = _unit(original["interaction_likelihood"], f"{name} interaction")
            confidence = _unit(original["confidence"], f"{name} confidence")
            original_score = _unit(original["conservative_risk"], f"{name} original score")
            fallback = _unit(original["fallback_risk"], f"{name} fallback")
            rejections = tuple(detail.get("input_rejections", ()))
            supported = (detail["mode"] == "evidence_supported"
                         and confidence > 0 and not rejections)
            normalized = context = reserve = old_scale = new_scale = None
            used_confidence, raw_proxy, context_basis, proxy_rejections = confidence, None, None, ()
            used_severity, used_interaction = severity, interaction
            if name == "contact" and contact_proxy is not None and detail.get("basis") in CONTACT_BASES:
                recovery = _contact_inputs(contact_proxy, contact_interaction, detail,
                    timestamp_seconds, maximum_contact_age_seconds,
                    maximum_interaction_age_seconds, prior_confidence_cap)
                raw_proxy, proxy_rejections = recovery[0], recovery[1]
                if not proxy_rejections:
                    supported = True
                    used_severity, used_confidence, used_interaction, context_basis = recovery[2:]
            score = original_score
            if supported:
                normalized = used_severity
                if name in ("potential", "kinetic"):
                    if source_energy_scales_j is None or name not in source_energy_scales_j:
                        raise ValueError("original energy reference scales are required")
                    old_scale = _finite(source_energy_scales_j[name], f"original {name} scale")
                    new_scale = getattr(config, f"{name}_energy_scale_j")
                    if old_scale <= 0 or new_scale > old_scale:
                        raise ValueError("candidate scale cannot exceed the original clipped energy scale")
                    normalized = min(1.0, severity * old_scale / new_scale)
                elif name == "contact":
                    normalized = used_severity ** config.contact_power
                context = config.context_floor + (1.0 - config.context_floor) * used_interaction
                reserve = 1.0 + config.uncertainty_gain * (1.0 - used_confidence)
                score = min(1.0, normalized * context * reserve)
            factors.append(CalibratedFactor(
                name, supported, detail["mode"], detail.get("basis", "unspecified"),
                severity, interaction, confidence, original_score, fallback,
                normalized, context, reserve, score, getattr(config.weights, name),
                original["source"], tuple(original.get("evidence", ())), rejections,
                old_scale, new_scale, used_confidence if supported else None,
                raw_proxy, context_basis, proxy_rejections,
                ("supported_visual_cue_with_explicit_context_prior"
                 if supported and context_basis == "configured_upper_interaction_prior_for_unavailable_context"
                 else "supported_visual_cue_with_current_context" if supported and context_basis
                 else "supported_original_upper_evidence" if supported else "original_factor_policy"),
            ))
        usable = [item for item in factors if item.supported]
        if usable:
            prior = max((item.score for item in factors if not item.supported), default=0.0)
            hazard = 1.0 - math.prod((1.0 - item.score) ** item.exponent for item in usable)
            physical_phi = min(1.0, prior + (1.0 - prior) * hazard)
            reason = "verified_object_factor_calibration"
        else:
            reason = "no_supported_factor_original_policy_preserved"
    phi = (max(physical_phi, config.human_protection_floor)
           if config.enabled and verified_human else physical_phi)
    return RestaurantCalibrationAssessment(
        config, eligible, verified_human, reason, original_phi, physical_phi,
        phi, prior, hazard, tuple(factors),
    )


def _contact_inputs(proxy, interaction, detail, timestamp, contact_age, interaction_age, prior_cap):
    """Retain a valid intrinsic cue when only motion/context is unavailable.

    No evidence is refreshed. Invalid/stale visual evidence is never recovered.
    Unsupported route context uses an explicit upper interaction prior of one;
    the kinetic factor still follows its original evidence/prior contract.
    """
    proxy = proxy if isinstance(proxy, ScalarEstimate) else ScalarEstimate(**proxy)
    raw = asdict(proxy)
    raw["status"] = proxy.status.value
    if timestamp is None or contact_age is None:
        raise ValueError("contact proxy requires the original current time and approved maximum age")
    timestamp = _finite(timestamp, "current source time")
    contact_age = _finite(contact_age, "maximum contact age")
    interaction_age = _finite(interaction_age, "maximum interaction age")
    prior_cap = _unit(prior_cap, "prior confidence cap")
    if timestamp < 0 or contact_age <= 0 or interaction_age <= 0:
        raise ValueError("source time must be nonnegative and evidence ages positive")
    rejections = []
    if proxy.source not in CONTACT_SOURCES or proxy.unit != "1":
        rejections.append("unsupported visual contact proxy source/unit")
    if proxy.status not in {EstimateStatus.MEASURED, EstimateStatus.INFERRED, EstimateStatus.PRIOR}:
        rejections.append("visual proxy is unavailable, fallback or oracle")
    if not proxy.is_available or proxy.confidence <= 0:
        rejections.append("visual contact proxy has no supported confidence")
    elif not 0 <= proxy.lower_bound <= proxy.upper_bound <= 1:
        rejections.append("visual contact proxy bounds outside unit interval")
    if not 0 <= timestamp - proxy.timestamp_seconds <= contact_age + 1e-9:
        rejections.append("visual contact proxy source is stale or future")
    rejections.extend(reason for reason in detail.get("input_rejections", ())
                      if not reason.startswith("contact_interaction_likelihood:"))
    if rejections:
        return raw, tuple(rejections), None, None, None, None
    confidence = min(proxy.confidence, prior_cap) if proxy.status is EstimateStatus.PRIOR else proxy.confidence
    if confidence <= 0:
        return raw, ("visual proxy has no effective confidence under original prior cap",), None, None, None, None
    context, basis = 1.0, "configured_upper_interaction_prior_for_unavailable_context"
    if interaction is not None:
        interaction = interaction if isinstance(interaction, ScalarEstimate) else ScalarEstimate(**interaction)
        if (interaction.is_available and interaction.unit == "1" and interaction.confidence > 0
                and interaction.status in {EstimateStatus.MEASURED, EstimateStatus.INFERRED, EstimateStatus.PRIOR}
                and 0 <= interaction.lower_bound <= interaction.upper_bound <= 1
                and 0 <= timestamp - interaction.timestamp_seconds <= interaction_age + 1e-9):
            context = interaction.upper_bound
            basis = "current_route_interaction_upper_bound"
    return raw, (), proxy.upper_bound, confidence, context, basis


@dataclass(frozen=True)
class CalibratedObjectPolicy:
    """Policy protocol wrapper; original CRA arithmetic remains identifiable."""

    original: CRASafeSAGEPolicyResult
    calibration: RestaurantCalibrationAssessment
    guidance_parameters: object

    @property
    def assessment(self):
        """Current scores use their own schema; legacy arithmetic is separate."""
        return CalibratedCRAAssessment(self.original.assessment, self.calibration)

    @property
    def risk(self):
        return self.calibration.phi

    @property
    def boundary_flux(self):
        return self.guidance_parameters.boundary_flux

    @property
    def tangential_strength(self):
        return self.guidance_parameters.tangential_strength

    @property
    def passing_side(self):
        return self.guidance_parameters.passing_side

    def as_dict(self):
        return {
            "cra": self.assessment.as_dict(),
            "legacy_cra": self.original.assessment.as_dict(),
            "restaurant_calibration": self.calibration.as_dict(),
            "safe_sage": {
            "risk": self.risk,
            "boundary_flux_magnitude": self.guidance_parameters.boundary_flux_magnitude,
            "boundary_flux": self.boundary_flux,
            "tangential_strength": self.tangential_strength,
            "passing_side": self.passing_side.value,
            },
        }


@dataclass(frozen=True)
class CalibratedCRAFactor:
    """Readout protocol with calibrated values and explicit exponent semantics."""

    original: object
    calibrated: CalibratedFactor | None

    @property
    def factor(self):
        return self.original.factor

    @property
    def estimate(self):
        item = self.calibrated
        if item is None or not item.supported:
            return self.original.estimate
        return RiskFactorEstimate(
            item.normalized_severity, item.context_multiplier,
            item.calibration_confidence, CALIBRATION_SOURCE,
            (*item.evidence, "bounded restaurant context and proportional uncertainty reserve"),
        )

    @property
    def conservative_risk(self):
        return self.original.conservative_risk if self.calibrated is None else self.calibrated.score

    @property
    def nominal_risk(self):
        item = self.calibrated
        return (self.original.nominal_risk if item is None or not item.supported
                else item.normalized_severity * item.context_multiplier)

    @property
    def fallback_risk(self):
        return self.original.fallback_risk

    @property
    def weight(self):
        return self.original.weight if self.calibrated is None else self.calibrated.exponent

    @property
    def weighted_conservative_risk(self):
        return 1.0 - (1.0 - self.conservative_risk) ** self.weight

    @property
    def uncertainty_uplift(self):
        return max(0.0, self.conservative_risk - self.nominal_risk)

    def as_dict(self):
        estimate = self.estimate
        return dict(
            severity=estimate.severity, interaction_likelihood=estimate.interaction_likelihood,
            confidence=estimate.confidence, source=estimate.source, evidence=list(estimate.evidence),
            nominal_risk=self.nominal_risk, fallback_risk=self.fallback_risk,
            conservative_risk=self.conservative_risk, weight=self.weight,
            weight_semantics="positive noisy-OR exponent, not a percentage",
            weighted_conservative_risk=self.weighted_conservative_risk,
            uncertainty_uplift=self.uncertainty_uplift,
            calibration=None if self.calibrated is None else asdict(self.calibrated),
        )


@dataclass(frozen=True)
class CalibratedCRAAssessment:
    """Current CRA protocol, without impersonating legacy noisy-OR arithmetic."""

    original: object
    calibrated: RestaurantCalibrationAssessment

    @property
    def phi(self):
        return self.calibrated.phi

    @property
    def nominal_phi(self):
        # This candidate's confidence-free physical score uses the same pooled
        # priors. The human floor is policy, never an invented physical factor.
        rows = self.calibrated.factors
        supported = [r for r in rows if r.supported]
        if not supported:
            return self.original.nominal_phi
        p = self.calibrated.prior_baseline
        hazard = 1.0 - math.prod(
            (1.0 - r.normalized_severity * r.context_multiplier) ** r.exponent
            for r in supported)
        return p + (1.0 - p) * hazard

    @property
    def factors(self):
        calibrated = {row.factor: row for row in self.calibrated.factors}
        return tuple(CalibratedCRAFactor(row, calibrated.get(row.factor.value))
                     for row in self.original.factors)

    def factor(self, factor):
        return next(row for row in self.factors if row.factor == PhysicalRiskFactor(factor))

    @property
    def overall_confidence(self):
        return self.original.overall_confidence

    @property
    def uncertainty_uplift(self):
        # Excludes the independently identified human protection prior.
        return max(0.0, self.calibrated.physical_phi - self.nominal_phi)

    @property
    def dominant_factor(self):
        dominant = max(self.factors, key=lambda row: row.weighted_conservative_risk)
        return dominant.factor if dominant.weighted_conservative_risk > 0 else None

    def as_dict(self):
        return dict(
            schema_version=1, aggregation="restaurant_evidence_weighted_noisy_or",
            policy_schema=CALIBRATION_SOURCE,
            nominal_phi=self.nominal_phi, phi=self.phi,
            physical_phi=self.calibrated.physical_phi,
            overall_confidence=self.overall_confidence,
            uncertainty_uplift=self.uncertainty_uplift,
            human_protection_uplift=self.phi - self.calibrated.physical_phi,
            dominant_factor=None if self.dominant_factor is None else self.dominant_factor.value,
            factors={row.factor.value: row.as_dict() for row in self.factors},
        )


def calibrate_object_policy(
    original: CRASafeSAGEPolicyResult,
    factor_details: Mapping,
    config: RestaurantCalibrationConfig,
    *,
    eligible: bool,
    verified_human: bool = False,
    source_energy_scales_j: Mapping | None = None,
    contact_proxy: ScalarEstimate | Mapping | None = None,
    contact_interaction: ScalarEstimate | Mapping | None = None,
    timestamp_seconds: float | None = None,
    maximum_contact_age_seconds: float | None = None,
    maximum_interaction_age_seconds: float = 1.0,
    prior_confidence_cap: float = 0.5,
):
    """Return the legacy object unchanged when disabled or wholly unassessed.

    The caller applies this once, after the original factor/human-mass updates.
    The input must use the original shared .5..1.7 mapping. An explicit
    calibrated-object endpoint can strengthen the output without altering risk,
    lambda, passing side, evidence or unassessed-object policies.
    """
    if not isinstance(original, CRASafeSAGEPolicyResult):
        raise TypeError("original policy must be an uncalibrated CRASafeSAGEPolicyResult")
    if not isinstance(config, RestaurantCalibrationConfig):
        raise TypeError("config must be RestaurantCalibrationConfig")
    if not isinstance(eligible, bool) or not isinstance(verified_human, bool):
        raise TypeError("eligibility and human verification must be bool")
    if not config.enabled or not (eligible or verified_human):
        return original
    if not math.isclose(original.boundary_flux, -(.5 + 1.2 * original.risk), abs_tol=1e-12):
        raise ValueError("restaurant candidate requires the original shared flux scale")
    assessment = evaluate_restaurant_calibration(
        original.assessment.as_dict(), factor_details, config, eligible=eligible,
        verified_human=verified_human, source_energy_scales_j=source_energy_scales_j,
        contact_proxy=contact_proxy, contact_interaction=contact_interaction,
        timestamp_seconds=timestamp_seconds, maximum_contact_age_seconds=maximum_contact_age_seconds,
        maximum_interaction_age_seconds=maximum_interaction_age_seconds,
        prior_confidence_cap=prior_confidence_cap,
    )
    parameters = replace(original.guidance_parameters,
        safety_parameters=replace(original.guidance_parameters.safety_parameters,
            risk=assessment.phi,
            boundary_flux_magnitude=config.normal_flux_magnitude_for_risk(assessment.phi)))
    return CalibratedObjectPolicy(original, assessment, parameters)

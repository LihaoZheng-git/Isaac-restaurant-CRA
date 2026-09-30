"""Conservative, model-agnostic mass estimation for CRA perception.

The runtime may obtain mass evidence from a learned image/RGB-D estimator,
complete metric geometry plus a material-density estimate, or a semantic
class prior.  This module validates and combines those sources without
depending on a particular neural-network framework.

The LiDAR geometry adapter intentionally reports a *visible envelope*.  That
source is rejected as full object volume by default: multiplying a partial
surface envelope by density would create falsely precise mass estimates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from types import MappingProxyType
from typing import Iterable, Mapping

from robot.perception.physical_attributes import (
    EstimateStatus,
    GeometryAttributes,
    MassAttributes,
    ScalarEstimate,
    combine_estimate_statuses,
)


MASS_PRIOR_SCHEMA_VERSION = 1
MASS_PRIOR_SOURCE = "semantic_class_mass_prior"
VOLUME_SOURCE = "complete_geometry_box_volume"
VOLUME_DENSITY_MASS_SOURCE = "volume_density_mass"
MASS_FUSION_SOURCE = "mass_evidence_fusion"


def _finite_real(value: object, *, name: str) -> float:
    """Return one finite real value while rejecting booleans."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _normalized_identifier(value: object, *, name: str) -> str:
    """Return a stable lower-case semantic identifier."""

    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    normalized = "_".join(value.casefold().replace("-", " ").split())
    if not normalized:
        raise ValueError(f"{name} cannot be empty.")
    return normalized


def _nonempty_text(value: object, *, name: str) -> str:
    """Return stripped nonempty text."""

    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    result = value.strip()
    if not result:
        raise ValueError(f"{name} cannot be empty.")
    return result


@dataclass(frozen=True)
class ClassMassPrior:
    """One auditable semantic-class mass interval."""

    nominal_kg: float
    lower_kg: float
    upper_kg: float
    confidence: float
    source: str
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        nominal = _finite_real(self.nominal_kg, name="nominal_kg")
        lower = _finite_real(self.lower_kg, name="lower_kg")
        upper = _finite_real(self.upper_kg, name="upper_kg")
        confidence = _finite_real(self.confidence, name="confidence")
        if lower < 0.0:
            raise ValueError("lower_kg cannot be negative.")
        if not lower <= nominal <= upper:
            raise ValueError("Mass prior must satisfy lower <= nominal <= upper.")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must lie in [0, 1].")
        source = _nonempty_text(self.source, name="source")
        if isinstance(self.evidence, str):
            raise TypeError("evidence must be an iterable of strings.")
        evidence = tuple(
            _nonempty_text(item, name="evidence item")
            for item in self.evidence
        )
        object.__setattr__(self, "nominal_kg", nominal)
        object.__setattr__(self, "lower_kg", lower)
        object.__setattr__(self, "upper_kg", upper)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "evidence", evidence)


@dataclass(frozen=True)
class ClassMassPriorTable:
    """Immutable lookup table supplied by an experiment configuration.

    The repository deliberately does not embed universal masses for semantic
    classes.  Such values depend on the benchmark objects and must be cited or
    measured by the experiment author.
    """

    priors: Mapping[str, ClassMassPrior]
    source: str = MASS_PRIOR_SOURCE

    def __post_init__(self) -> None:
        if not isinstance(self.priors, Mapping):
            raise TypeError("priors must be a mapping.")
        normalized: dict[str, ClassMassPrior] = {}
        for semantic_class, prior in self.priors.items():
            key = _normalized_identifier(
                semantic_class,
                name="semantic class",
            )
            if key in normalized:
                raise ValueError(f"Duplicate normalized semantic class {key!r}.")
            if not isinstance(prior, ClassMassPrior):
                raise TypeError("Every prior must be a ClassMassPrior.")
            normalized[key] = prior
        object.__setattr__(self, "priors", MappingProxyType(normalized))
        object.__setattr__(
            self,
            "source",
            _nonempty_text(self.source, name="source"),
        )

    def estimate(
        self,
        semantic_class: str,
        *,
        timestamp_seconds: float,
    ) -> MassAttributes:
        """Return a configured class prior or explicit unavailability."""

        key = _normalized_identifier(semantic_class, name="semantic_class")
        timestamp = _timestamp(timestamp_seconds)
        prior = self.priors.get(key)
        if prior is None:
            return _unavailable_mass(
                timestamp,
                source=self.source,
                evidence=(f"no configured mass prior for class {key!r}",),
            )
        return MassAttributes(
            mass_kg=ScalarEstimate(
                value=prior.nominal_kg,
                lower_bound=prior.lower_kg,
                upper_bound=prior.upper_kg,
                confidence=prior.confidence,
                unit="kg",
                source=self.source,
                timestamp_seconds=timestamp,
                status=EstimateStatus.PRIOR,
                evidence=(
                    f"semantic class={key}",
                    f"prior record source={prior.source}",
                    *prior.evidence,
                ),
            )
        )


@dataclass(frozen=True)
class BoxVolumeConfig:
    """Acceptance policy for deriving volume from metric dimensions."""

    accepted_geometry_sources: tuple[str, ...] = (
        "rgbd_complete_object_geometry",
        "stereo_complete_object_geometry",
    )
    minimum_dimension_confidence: float = 0.50
    maximum_volume_m3: float = 20.0

    def __post_init__(self) -> None:
        if isinstance(self.accepted_geometry_sources, str):
            raise TypeError(
                "accepted_geometry_sources must be an iterable of strings."
            )
        sources = tuple(
            _nonempty_text(source, name="accepted geometry source")
            for source in self.accepted_geometry_sources
        )
        if not sources:
            raise ValueError("accepted_geometry_sources cannot be empty.")
        if len(set(sources)) != len(sources):
            raise ValueError("accepted_geometry_sources cannot contain duplicates.")
        minimum_confidence = _finite_real(
            self.minimum_dimension_confidence,
            name="minimum_dimension_confidence",
        )
        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError("minimum_dimension_confidence must lie in [0, 1].")
        maximum_volume = _finite_real(
            self.maximum_volume_m3,
            name="maximum_volume_m3",
        )
        if maximum_volume <= 0.0:
            raise ValueError("maximum_volume_m3 must be greater than zero.")
        object.__setattr__(self, "accepted_geometry_sources", sources)
        object.__setattr__(
            self,
            "minimum_dimension_confidence",
            minimum_confidence,
        )
        object.__setattr__(self, "maximum_volume_m3", maximum_volume)


def estimate_box_volume(
    geometry: GeometryAttributes,
    *,
    timestamp_seconds: float,
    config: BoxVolumeConfig | None = None,
) -> ScalarEstimate:
    """Estimate bounding-box volume from accepted complete geometry."""

    if not isinstance(geometry, GeometryAttributes):
        raise TypeError("geometry must be a GeometryAttributes.")
    settings = config or BoxVolumeConfig()
    if not isinstance(settings, BoxVolumeConfig):
        raise TypeError("config must be a BoxVolumeConfig.")
    timestamp = _timestamp(timestamp_seconds)
    dimensions = (
        ("width_m", geometry.width_m),
        ("depth_m", geometry.depth_m),
        ("height_m", geometry.height_m),
    )
    for name, estimate in dimensions:
        if not estimate.is_available:
            return ScalarEstimate.unavailable(
                unit="m^3",
                source=VOLUME_SOURCE,
                timestamp_seconds=timestamp,
                evidence=(f"{name} is unavailable",),
            )
        if estimate.source not in settings.accepted_geometry_sources:
            return ScalarEstimate.unavailable(
                unit="m^3",
                source=VOLUME_SOURCE,
                timestamp_seconds=timestamp,
                evidence=(
                    f"{name} source {estimate.source!r} is not complete geometry",
                    (
                        "visible LiDAR envelopes are rejected as full object "
                        "volume by default"
                    ),
                ),
            )
        if estimate.confidence < settings.minimum_dimension_confidence:
            return ScalarEstimate.unavailable(
                unit="m^3",
                source=VOLUME_SOURCE,
                timestamp_seconds=timestamp,
                evidence=(
                    f"{name} confidence is below the configured threshold",
                ),
            )
        if estimate.timestamp_seconds > timestamp + 1e-12:
            raise ValueError("Geometry evidence cannot be newer than the result.")

    values = tuple(float(estimate.value) for _, estimate in dimensions)
    lowers = tuple(float(estimate.lower_bound) for _, estimate in dimensions)
    uppers = tuple(float(estimate.upper_bound) for _, estimate in dimensions)
    nominal_volume = math.prod(values)
    lower_volume = math.prod(lowers)
    upper_volume = math.prod(uppers)
    evidence_timestamp = min(
        estimate.timestamp_seconds for _, estimate in dimensions
    )
    if upper_volume > settings.maximum_volume_m3:
        return ScalarEstimate.unavailable(
            unit="m^3",
            source=VOLUME_SOURCE,
            timestamp_seconds=timestamp,
            evidence=(
                f"upper volume {upper_volume:.6g} m^3 exceeds corruption guard",
            ),
        )
    return ScalarEstimate(
        value=nominal_volume,
        lower_bound=lower_volume,
        upper_bound=upper_volume,
        confidence=min(estimate.confidence for _, estimate in dimensions),
        unit="m^3",
        source=VOLUME_SOURCE,
        timestamp_seconds=evidence_timestamp,
        status=combine_estimate_statuses(
            (
                EstimateStatus.INFERRED,
                *(estimate.status for _, estimate in dimensions),
            )
        ),
        evidence=(
            "axis-aligned box volume = width * depth * height",
            "dimensions were accepted as complete metric geometry",
            "box volume may exceed occupied material volume",
        ),
    )


def estimate_mass_from_volume_density(
    volume_m3: ScalarEstimate,
    density_kg_per_m3: ScalarEstimate,
    *,
    timestamp_seconds: float,
) -> MassAttributes:
    """Propagate nonnegative volume and density intervals into mass."""

    timestamp = _timestamp(timestamp_seconds)
    _validate_nonnegative_estimate(
        volume_m3,
        name="volume_m3",
        unit="m^3",
    )
    _validate_nonnegative_estimate(
        density_kg_per_m3,
        name="density_kg_per_m3",
        unit="kg/m^3",
    )
    if not volume_m3.is_available or not density_kg_per_m3.is_available:
        return _unavailable_mass(
            timestamp,
            source=VOLUME_DENSITY_MASS_SOURCE,
            evidence=("volume and density must both be available",),
        )
    if (
        volume_m3.timestamp_seconds > timestamp + 1e-12
        or density_kg_per_m3.timestamp_seconds > timestamp + 1e-12
    ):
        raise ValueError("Mass evidence cannot be newer than the result.")

    nominal = float(volume_m3.value) * float(density_kg_per_m3.value)
    lower = float(volume_m3.lower_bound) * float(
        density_kg_per_m3.lower_bound
    )
    upper = float(volume_m3.upper_bound) * float(
        density_kg_per_m3.upper_bound
    )
    evidence_timestamp = min(
        volume_m3.timestamp_seconds, density_kg_per_m3.timestamp_seconds
    )
    return MassAttributes(
        mass_kg=ScalarEstimate(
            value=nominal,
            lower_bound=lower,
            upper_bound=upper,
            confidence=min(
                volume_m3.confidence,
                density_kg_per_m3.confidence,
            ),
            unit="kg",
            source=VOLUME_DENSITY_MASS_SOURCE,
            timestamp_seconds=evidence_timestamp,
            status=combine_estimate_statuses(
                (
                    EstimateStatus.INFERRED,
                    volume_m3.status,
                    density_kg_per_m3.status,
                )
            ),
            evidence=(
                f"volume source={volume_m3.source}",
                f"density source={density_kg_per_m3.source}",
                "mass interval propagated by nonnegative interval multiplication",
            ),
        )
    )


def fuse_mass_estimates(
    candidates: Iterable[ScalarEstimate],
    *,
    timestamp_seconds: float,
) -> MassAttributes:
    """Fuse available mass candidates with an interval hull.

    The nominal value is confidence-weighted.  The result keeps the full hull
    of candidate intervals and reduces confidence when candidate nominal
    values disagree.  Fusion therefore cannot hide source disagreement by
    returning an artificially narrow interval.
    """

    timestamp = _timestamp(timestamp_seconds)
    if isinstance(candidates, ScalarEstimate):
        raise TypeError("candidates must be an iterable of ScalarEstimate.")
    try:
        candidate_items = tuple(candidates)
    except TypeError as error:
        raise TypeError("candidates must be iterable.") from error
    available: list[ScalarEstimate] = []
    for candidate in candidate_items:
        _validate_nonnegative_estimate(candidate, name="candidate", unit="kg")
        if candidate.timestamp_seconds > timestamp + 1e-12:
            raise ValueError("Mass candidate cannot be newer than the result.")
        if candidate.is_available:
            available.append(candidate)
    if not available:
        return _unavailable_mass(
            timestamp,
            source=MASS_FUSION_SOURCE,
            evidence=("no available mass evidence",),
        )
    if len(available) == 1:
        return MassAttributes(mass_kg=available[0])

    weights = [max(candidate.confidence, 1e-6) for candidate in available]
    weight_sum = sum(weights)
    nominal = sum(
        weight * float(candidate.value)
        for weight, candidate in zip(weights, available, strict=True)
    ) / weight_sum
    lower = min(float(candidate.lower_bound) for candidate in available)
    upper = max(float(candidate.upper_bound) for candidate in available)
    nominal_values = [float(candidate.value) for candidate in available]
    spread = max(nominal_values) - min(nominal_values)
    scale = max(max(nominal_values), 1e-9)
    agreement = max(0.0, 1.0 - spread / scale)
    weighted_confidence = sum(
        weight * candidate.confidence
        for weight, candidate in zip(weights, available, strict=True)
    ) / weight_sum
    confidence = weighted_confidence * agreement
    status = combine_estimate_statuses(candidate.status for candidate in available)
    evidence_timestamp = min(candidate.timestamp_seconds for candidate in available)
    return MassAttributes(
        mass_kg=ScalarEstimate(
            value=nominal,
            lower_bound=min(lower, nominal),
            upper_bound=max(upper, nominal),
            confidence=confidence,
            unit="kg",
            source=MASS_FUSION_SOURCE,
            timestamp_seconds=evidence_timestamp,
            status=status,
            evidence=(
                f"fused {len(available)} available mass candidates",
                f"candidate sources={','.join(item.source for item in available)}",
                f"nominal agreement factor={agreement:.6f}",
                "bounds are the conservative hull of all candidate intervals",
            ),
        )
    )


def estimate_track_mass(
    semantic_class: str,
    *,
    timestamp_seconds: float,
    prior_table: ClassMassPriorTable | None = None,
    learned_mass: ScalarEstimate | None = None,
    geometry: GeometryAttributes | None = None,
    density_kg_per_m3: ScalarEstimate | None = None,
    volume_config: BoxVolumeConfig | None = None,
) -> MassAttributes:
    """Build and conservatively fuse all supplied mass evidence."""

    timestamp = _timestamp(timestamp_seconds)
    semantic_key = _normalized_identifier(
        semantic_class,
        name="semantic_class",
    )
    candidates: list[ScalarEstimate] = []
    if learned_mass is not None:
        _validate_nonnegative_estimate(
            learned_mass,
            name="learned_mass",
            unit="kg",
        )
        candidates.append(learned_mass)
    if (geometry is None) != (density_kg_per_m3 is None):
        raise ValueError(
            "geometry and density_kg_per_m3 must be supplied together."
        )
    if geometry is not None and density_kg_per_m3 is not None:
        volume = estimate_box_volume(
            geometry,
            timestamp_seconds=timestamp,
            config=volume_config,
        )
        geometric_mass = estimate_mass_from_volume_density(
            volume,
            density_kg_per_m3,
            timestamp_seconds=timestamp,
        )
        candidates.append(geometric_mass.mass_kg)
    if prior_table is not None:
        if not isinstance(prior_table, ClassMassPriorTable):
            raise TypeError("prior_table must be a ClassMassPriorTable.")
        candidates.append(
            prior_table.estimate(
                semantic_key,
                timestamp_seconds=timestamp,
            ).mass_kg
        )
    return fuse_mass_estimates(candidates, timestamp_seconds=timestamp)


def _timestamp(value: object) -> float:
    """Return one nonnegative finite timestamp."""

    timestamp = _finite_real(value, name="timestamp_seconds")
    if timestamp < 0.0:
        raise ValueError("timestamp_seconds cannot be negative.")
    return timestamp


def _validate_nonnegative_estimate(
    estimate: object,
    *,
    name: str,
    unit: str,
) -> None:
    """Validate one optional nonnegative physical estimate."""

    if not isinstance(estimate, ScalarEstimate):
        raise TypeError(f"{name} must be a ScalarEstimate.")
    if estimate.unit != unit:
        raise ValueError(f"{name} must use unit {unit!r}.")
    if estimate.is_available and float(estimate.lower_bound) < 0.0:
        raise ValueError(f"{name} cannot contain negative values.")


def _unavailable_mass(
    timestamp: float,
    *,
    source: str,
    evidence: tuple[str, ...],
) -> MassAttributes:
    """Return explicit missing mass evidence."""

    return MassAttributes(
        mass_kg=ScalarEstimate.unavailable(
            unit="kg",
            source=source,
            timestamp_seconds=timestamp,
            evidence=evidence,
        )
    )

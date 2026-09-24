"""Dependency-free physical-attribute data contracts for CRA perception.

Vision, LiDAR, tracking, learned models, and controlled priors produce very
different kinds of evidence.  This module provides one immutable boundary
between those estimators and the CRA mathematical layer.  It intentionally
contains no Isaac Sim, model-framework, risk-fusion, or Safe-SAGE logic.

Every estimate records a bounded value (when available), confidence,
provenance, timestamp, and human-readable evidence.  Missing evidence is
represented explicitly and must not be confused with a measured zero.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
import math
from numbers import Integral, Real
from typing import Iterable, Iterator


PHYSICAL_ATTRIBUTE_SCHEMA_VERSION = 1
DIMENSIONLESS_UNIT = "1"


def _finite_real(value: object, *, name: str) -> float:
    """Return one finite real value while rejecting booleans."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _unit_interval(value: object, *, name: str) -> float:
    """Return one finite value in ``[0, 1]``."""

    result = _finite_real(value, name=name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1].")
    return result


def _nonnegative_integer(value: object, *, name: str) -> int:
    """Return one nonnegative integer while rejecting booleans."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} cannot be negative.")
    return result


def _nonempty_text(value: object, *, name: str) -> str:
    """Return stripped nonempty text."""

    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    result = value.strip()
    if not result:
        raise ValueError(f"{name} cannot be empty.")
    return result


def _normalized_identifier(value: object, *, name: str) -> str:
    """Return a stable lower-case identifier for a class or category."""

    text = _nonempty_text(value, name=name)
    normalized = "_".join(text.casefold().replace("-", " ").split())
    if not normalized:
        raise ValueError(f"{name} cannot be empty.")
    return normalized


def _evidence_items(value: object) -> tuple[str, ...]:
    """Return immutable, validated evidence statements."""

    if isinstance(value, str):
        raise TypeError("evidence must be an iterable of strings, not a string.")
    try:
        items = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError("evidence must be an iterable of strings.") from error
    return tuple(_nonempty_text(item, name="evidence item") for item in items)


class EstimateStatus(str, Enum):
    """How a physical-attribute estimate was obtained."""

    MEASURED = "measured"
    INFERRED = "inferred"
    PRIOR = "prior"
    ORACLE = "oracle"
    FALLBACK = "fallback"
    UNAVAILABLE = "unavailable"


def combine_estimate_statuses(
    statuses: Iterable[EstimateStatus],
) -> EstimateStatus:
    """Return a conservative provenance status for a derived estimate.

    A derived value must retain any fallback, oracle, or prior dependency so it
    cannot later be mislabeled as sensor-only evidence.
    """

    items = frozenset(EstimateStatus(item) for item in statuses)
    if not items:
        raise ValueError("at least one estimate status is required")
    for status in (
        EstimateStatus.UNAVAILABLE,
        EstimateStatus.FALLBACK,
        EstimateStatus.ORACLE,
        EstimateStatus.PRIOR,
        EstimateStatus.INFERRED,
        EstimateStatus.MEASURED,
    ):
        if status in items:
            return status
    raise AssertionError("unreachable estimate status combination")


@dataclass(frozen=True)
class ScalarEstimate:
    """One scalar physical estimate with interval and provenance.

    ``confidence`` describes confidence in the evidence, not hazard severity.
    An unavailable estimate has no numeric values and exactly zero confidence.
    """

    value: float | None
    lower_bound: float | None
    upper_bound: float | None
    confidence: float
    unit: str
    source: str
    timestamp_seconds: float
    status: EstimateStatus = EstimateStatus.INFERRED
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        try:
            status = EstimateStatus(self.status)
        except (TypeError, ValueError) as error:
            raise ValueError("status must be a supported EstimateStatus.") from error
        confidence = _unit_interval(self.confidence, name="confidence")
        unit = _nonempty_text(self.unit, name="unit")
        source = _nonempty_text(self.source, name="source")
        timestamp = _finite_real(
            self.timestamp_seconds,
            name="timestamp_seconds",
        )
        if timestamp < 0.0:
            raise ValueError("timestamp_seconds cannot be negative.")
        evidence = _evidence_items(self.evidence)

        numeric_values = (self.value, self.lower_bound, self.upper_bound)
        if status is EstimateStatus.UNAVAILABLE:
            if any(value is not None for value in numeric_values):
                raise ValueError(
                    "Unavailable estimates cannot contain numeric values."
                )
            if confidence != 0.0:
                raise ValueError(
                    "Unavailable estimates must have zero confidence."
                )
        else:
            if any(value is None for value in numeric_values):
                raise ValueError(
                    "Available estimates require value and both bounds."
                )
            nominal = _finite_real(self.value, name="value")
            lower = _finite_real(self.lower_bound, name="lower_bound")
            upper = _finite_real(self.upper_bound, name="upper_bound")
            if lower > upper:
                raise ValueError("lower_bound cannot exceed upper_bound.")
            if not lower <= nominal <= upper:
                raise ValueError("value must lie inside its bounds.")
            object.__setattr__(self, "value", nominal)
            object.__setattr__(self, "lower_bound", lower)
            object.__setattr__(self, "upper_bound", upper)

        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "timestamp_seconds", timestamp)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "evidence", evidence)

    @property
    def is_available(self) -> bool:
        """Return whether this estimate contains a numeric interval."""

        return self.status is not EstimateStatus.UNAVAILABLE

    @property
    def is_fallback(self) -> bool:
        """Return whether this is an explicit conservative fallback."""

        return self.status is EstimateStatus.FALLBACK

    @property
    def interval_width(self) -> float | None:
        """Return uncertainty-interval width, or ``None`` when unavailable."""

        if self.lower_bound is None or self.upper_bound is None:
            return None
        return self.upper_bound - self.lower_bound

    @classmethod
    def unavailable(
        cls,
        *,
        unit: str,
        source: str,
        timestamp_seconds: float,
        evidence: Iterable[str] = (),
    ) -> ScalarEstimate:
        """Return an explicit missing-evidence estimate."""

        return cls(
            value=None,
            lower_bound=None,
            upper_bound=None,
            confidence=0.0,
            unit=unit,
            source=source,
            timestamp_seconds=timestamp_seconds,
            status=EstimateStatus.UNAVAILABLE,
            evidence=tuple(evidence),
        )

    @classmethod
    def fallback(
        cls,
        *,
        value: float,
        lower_bound: float,
        upper_bound: float,
        unit: str,
        source: str,
        timestamp_seconds: float,
        confidence: float = 0.0,
        evidence: Iterable[str] = (),
    ) -> ScalarEstimate:
        """Return an explicitly marked conservative fallback estimate."""

        return cls(
            value=value,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            confidence=confidence,
            unit=unit,
            source=source,
            timestamp_seconds=timestamp_seconds,
            status=EstimateStatus.FALLBACK,
            evidence=tuple(evidence),
        )


@dataclass(frozen=True)
class CategoricalEstimate:
    """One normalized categorical estimate with provenance."""

    value: str | None
    confidence: float
    source: str
    timestamp_seconds: float
    status: EstimateStatus = EstimateStatus.INFERRED
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        try:
            status = EstimateStatus(self.status)
        except (TypeError, ValueError) as error:
            raise ValueError("status must be a supported EstimateStatus.") from error
        confidence = _unit_interval(self.confidence, name="confidence")
        source = _nonempty_text(self.source, name="source")
        timestamp = _finite_real(
            self.timestamp_seconds,
            name="timestamp_seconds",
        )
        if timestamp < 0.0:
            raise ValueError("timestamp_seconds cannot be negative.")
        evidence = _evidence_items(self.evidence)

        if status is EstimateStatus.UNAVAILABLE:
            if self.value is not None:
                raise ValueError("Unavailable estimates cannot contain a category.")
            if confidence != 0.0:
                raise ValueError(
                    "Unavailable estimates must have zero confidence."
                )
        else:
            value = _normalized_identifier(self.value, name="value")
            object.__setattr__(self, "value", value)

        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "timestamp_seconds", timestamp)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "evidence", evidence)

    @property
    def is_available(self) -> bool:
        """Return whether this estimate contains a category."""

        return self.status is not EstimateStatus.UNAVAILABLE

    @classmethod
    def unavailable(
        cls,
        *,
        source: str,
        timestamp_seconds: float,
        evidence: Iterable[str] = (),
    ) -> CategoricalEstimate:
        """Return an explicit missing-category estimate."""

        return cls(
            value=None,
            confidence=0.0,
            source=source,
            timestamp_seconds=timestamp_seconds,
            status=EstimateStatus.UNAVAILABLE,
            evidence=tuple(evidence),
        )


def _validate_scalar(
    estimate: object,
    *,
    name: str,
    unit: str,
    nonnegative: bool = False,
    unit_interval: bool = False,
) -> None:
    """Validate one typed scalar field without rejecting unavailability."""

    if not isinstance(estimate, ScalarEstimate):
        raise TypeError(f"{name} must be a ScalarEstimate.")
    if estimate.unit != unit:
        raise ValueError(f"{name} must use unit {unit!r}.")
    if not estimate.is_available:
        return
    assert estimate.lower_bound is not None
    assert estimate.upper_bound is not None
    if nonnegative and estimate.lower_bound < 0.0:
        raise ValueError(f"{name} cannot contain negative values.")
    if unit_interval and not (
        0.0 <= estimate.lower_bound <= estimate.upper_bound <= 1.0
    ):
        raise ValueError(f"{name} bounds must lie in [0, 1].")


def _validate_category(estimate: object, *, name: str) -> None:
    """Validate one categorical field."""

    if not isinstance(estimate, CategoricalEstimate):
        raise TypeError(f"{name} must be a CategoricalEstimate.")


@dataclass(frozen=True)
class MassAttributes:
    """Mass evidence for one tracked obstacle."""

    mass_kg: ScalarEstimate

    def __post_init__(self) -> None:
        _validate_scalar(
            self.mass_kg,
            name="mass_kg",
            unit="kg",
            nonnegative=True,
        )


@dataclass(frozen=True)
class GeometryAttributes:
    """Metric geometry needed by potential-energy estimation."""

    width_m: ScalarEstimate
    depth_m: ScalarEstimate
    height_m: ScalarEstimate
    center_height_m: ScalarEstimate
    fall_height_m: ScalarEstimate

    def __post_init__(self) -> None:
        for item in fields(self):
            _validate_scalar(
                getattr(self, item.name),
                name=item.name,
                unit="m",
                nonnegative=True,
            )


@dataclass(frozen=True)
class MotionAttributes:
    """Tracked motion evidence needed by kinetic-risk estimation.

    Velocity components use the world frame so they remain stable while the
    robot turns.  Estimator provenance should repeat the frame explicitly.
    """

    velocity_x_mps: ScalarEstimate
    velocity_y_mps: ScalarEstimate
    relative_speed_mps: ScalarEstimate
    route_crossing_likelihood: ScalarEstimate

    def __post_init__(self) -> None:
        _validate_scalar(
            self.velocity_x_mps,
            name="velocity_x_mps",
            unit="m/s",
        )
        _validate_scalar(
            self.velocity_y_mps,
            name="velocity_y_mps",
            unit="m/s",
        )
        _validate_scalar(
            self.relative_speed_mps,
            name="relative_speed_mps",
            unit="m/s",
            nonnegative=True,
        )
        _validate_scalar(
            self.route_crossing_likelihood,
            name="route_crossing_likelihood",
            unit=DIMENSIONLESS_UNIT,
            unit_interval=True,
        )


@dataclass(frozen=True)
class StabilityAttributes:
    """Support and instability evidence for potential-energy risk."""

    instability_likelihood: ScalarEstimate
    support_margin_m: ScalarEstimate
    support_state: CategoricalEstimate

    def __post_init__(self) -> None:
        _validate_scalar(
            self.instability_likelihood,
            name="instability_likelihood",
            unit=DIMENSIONLESS_UNIT,
            unit_interval=True,
        )
        _validate_scalar(
            self.support_margin_m,
            name="support_margin_m",
            unit="m",
        )
        _validate_category(self.support_state, name="support_state")


@dataclass(frozen=True)
class ContactAttributes:
    """Visual evidence about harmful physical contact."""

    sharpness: ScalarEstimate
    hardness: ScalarEstimate
    dangerous_part_exposure: ScalarEstimate
    orientation_toward_route: ScalarEstimate
    dangerous_part: CategoricalEstimate

    def __post_init__(self) -> None:
        for name in (
            "sharpness",
            "hardness",
            "dangerous_part_exposure",
            "orientation_toward_route",
        ):
            _validate_scalar(
                getattr(self, name),
                name=name,
                unit=DIMENSIONLESS_UNIT,
                unit_interval=True,
            )
        _validate_category(self.dangerous_part, name="dangerous_part")


@dataclass(frozen=True)
class ThermalAttributes:
    """Visual thermal cues; these are not direct temperature measurements."""

    heat_source_likelihood: ScalarEstimate
    steam_likelihood: ScalarEstimate
    open_container_likelihood: ScalarEstimate
    carried_likelihood: ScalarEstimate
    spill_likelihood: ScalarEstimate
    source_type: CategoricalEstimate

    def __post_init__(self) -> None:
        for name in (
            "heat_source_likelihood",
            "steam_likelihood",
            "open_container_likelihood",
            "carried_likelihood",
            "spill_likelihood",
        ):
            _validate_scalar(
                getattr(self, name),
                name=name,
                unit=DIMENSIONLESS_UNIT,
                unit_interval=True,
            )
        _validate_category(self.source_type, name="source_type")


PhysicalAttributeGroup = (
    MassAttributes
    | GeometryAttributes
    | MotionAttributes
    | StabilityAttributes
    | ContactAttributes
    | ThermalAttributes
)


def _group_estimates(
    group: PhysicalAttributeGroup,
) -> Iterator[ScalarEstimate | CategoricalEstimate]:
    """Yield every estimate stored in one physical-attribute group."""

    for item in fields(group):
        estimate = getattr(group, item.name)
        if not isinstance(estimate, (ScalarEstimate, CategoricalEstimate)):
            raise TypeError(f"{item.name} must be a physical estimate.")
        yield estimate


@dataclass(frozen=True)
class PhysicalAttributes:
    """Complete physical evidence associated with one semantic track."""

    track_id: int
    semantic_class: str
    semantic_confidence: float
    semantic_source: str
    timestamp_seconds: float
    mass: MassAttributes
    geometry: GeometryAttributes
    motion: MotionAttributes
    stability: StabilityAttributes
    contact: ContactAttributes
    thermal: ThermalAttributes

    def __post_init__(self) -> None:
        track_id = _nonnegative_integer(self.track_id, name="track_id")
        semantic_class = _normalized_identifier(
            self.semantic_class,
            name="semantic_class",
        )
        semantic_confidence = _unit_interval(
            self.semantic_confidence,
            name="semantic_confidence",
        )
        semantic_source = _nonempty_text(
            self.semantic_source,
            name="semantic_source",
        )
        timestamp = _finite_real(
            self.timestamp_seconds,
            name="timestamp_seconds",
        )
        if timestamp < 0.0:
            raise ValueError("timestamp_seconds cannot be negative.")

        expected_types = {
            "mass": MassAttributes,
            "geometry": GeometryAttributes,
            "motion": MotionAttributes,
            "stability": StabilityAttributes,
            "contact": ContactAttributes,
            "thermal": ThermalAttributes,
        }
        for name, expected_type in expected_types.items():
            group = getattr(self, name)
            if not isinstance(group, expected_type):
                raise TypeError(f"{name} must be a {expected_type.__name__}.")
            for estimate in _group_estimates(group):
                if estimate.timestamp_seconds > timestamp + 1e-12:
                    raise ValueError(
                        f"{name} evidence cannot be newer than the record."
                    )

        object.__setattr__(self, "track_id", track_id)
        object.__setattr__(self, "semantic_class", semantic_class)
        object.__setattr__(self, "semantic_confidence", semantic_confidence)
        object.__setattr__(self, "semantic_source", semantic_source)
        object.__setattr__(self, "timestamp_seconds", timestamp)

    @property
    def available_estimate_count(self) -> int:
        """Return the number of physical estimates containing evidence."""

        return sum(estimate.is_available for estimate in self.estimates())

    def estimates(self) -> tuple[ScalarEstimate | CategoricalEstimate, ...]:
        """Return all estimates in stable attribute-family order."""

        groups: tuple[PhysicalAttributeGroup, ...] = (
            self.mass,
            self.geometry,
            self.motion,
            self.stability,
            self.contact,
            self.thermal,
        )
        return tuple(
            estimate
            for group in groups
            for estimate in _group_estimates(group)
        )

    @classmethod
    def unknown(
        cls,
        *,
        track_id: int,
        semantic_class: str = "unknown",
        semantic_confidence: float = 0.0,
        semantic_source: str = "unavailable",
        timestamp_seconds: float,
        evidence: Iterable[str] = (),
    ) -> PhysicalAttributes:
        """Return a complete record whose physical evidence is unavailable."""

        evidence_items = tuple(evidence)

        def scalar(unit: str) -> ScalarEstimate:
            return ScalarEstimate.unavailable(
                unit=unit,
                source="unavailable",
                timestamp_seconds=timestamp_seconds,
                evidence=evidence_items,
            )

        def category() -> CategoricalEstimate:
            return CategoricalEstimate.unavailable(
                source="unavailable",
                timestamp_seconds=timestamp_seconds,
                evidence=evidence_items,
            )

        return cls(
            track_id=track_id,
            semantic_class=semantic_class,
            semantic_confidence=semantic_confidence,
            semantic_source=semantic_source,
            timestamp_seconds=timestamp_seconds,
            mass=MassAttributes(mass_kg=scalar("kg")),
            geometry=GeometryAttributes(
                width_m=scalar("m"),
                depth_m=scalar("m"),
                height_m=scalar("m"),
                center_height_m=scalar("m"),
                fall_height_m=scalar("m"),
            ),
            motion=MotionAttributes(
                velocity_x_mps=scalar("m/s"),
                velocity_y_mps=scalar("m/s"),
                relative_speed_mps=scalar("m/s"),
                route_crossing_likelihood=scalar(DIMENSIONLESS_UNIT),
            ),
            stability=StabilityAttributes(
                instability_likelihood=scalar(DIMENSIONLESS_UNIT),
                support_margin_m=scalar("m"),
                support_state=category(),
            ),
            contact=ContactAttributes(
                sharpness=scalar(DIMENSIONLESS_UNIT),
                hardness=scalar(DIMENSIONLESS_UNIT),
                dangerous_part_exposure=scalar(DIMENSIONLESS_UNIT),
                orientation_toward_route=scalar(DIMENSIONLESS_UNIT),
                dangerous_part=category(),
            ),
            thermal=ThermalAttributes(
                heat_source_likelihood=scalar(DIMENSIONLESS_UNIT),
                steam_likelihood=scalar(DIMENSIONLESS_UNIT),
                open_container_likelihood=scalar(DIMENSIONLESS_UNIT),
                carried_likelihood=scalar(DIMENSIONLESS_UNIT),
                spill_likelihood=scalar(DIMENSIONLESS_UNIT),
                source_type=category(),
            ),
        )

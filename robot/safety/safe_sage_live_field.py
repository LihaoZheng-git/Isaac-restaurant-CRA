"""Transactional freshness management for live Safe-SAGE fields.

The numerical field builder is intentionally separate from this module.  A
caller first constructs a complete candidate field and commits it only after
all perception, tracking, configuration-space, guidance, and Poisson stages
have succeeded.  Consequently, a partially built field can never replace the
last accepted snapshot.

This module has no Isaac Sim dependency and can therefore be unit tested with
small synthetic payloads.
"""

import math
from dataclasses import dataclass
from numbers import Real
from typing import Generic, TypeVar


FieldPayload = TypeVar("FieldPayload")


def _finite_real(value: object, *, name: str) -> float:
    """Return one finite real number while rejecting booleans."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _nonnegative_integer(value: object, *, name: str) -> int:
    """Return one nonnegative integer while rejecting booleans."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} cannot be negative.")
    return value


@dataclass(frozen=True)
class SafeSAGELiveFieldConfig:
    """Freshness thresholds for one live-field control loop."""

    maximum_field_age_seconds: float = 0.35
    maximum_reused_control_cycles: int = 2
    clock_tolerance_seconds: float = 1e-9

    def __post_init__(self) -> None:
        maximum_age = _finite_real(
            self.maximum_field_age_seconds,
            name="maximum_field_age_seconds",
        )
        maximum_reuses = _nonnegative_integer(
            self.maximum_reused_control_cycles,
            name="maximum_reused_control_cycles",
        )
        clock_tolerance = _finite_real(
            self.clock_tolerance_seconds,
            name="clock_tolerance_seconds",
        )
        if maximum_age <= 0.0:
            raise ValueError(
                "maximum_field_age_seconds must be greater than zero."
            )
        if clock_tolerance < 0.0:
            raise ValueError("clock_tolerance_seconds cannot be negative.")
        object.__setattr__(self, "maximum_field_age_seconds", maximum_age)
        object.__setattr__(
            self,
            "maximum_reused_control_cycles",
            maximum_reuses,
        )
        object.__setattr__(self, "clock_tolerance_seconds", clock_tolerance)


@dataclass(frozen=True)
class SafeSAGELiveFieldSnapshot(Generic[FieldPayload]):
    """One complete field and the LiDAR metadata from which it was built."""

    sensor_frame_id: int
    sensor_timestamp_seconds: float
    simulation_time_seconds: float
    build_wall_seconds: float
    payload: FieldPayload

    def __post_init__(self) -> None:
        frame_id = _nonnegative_integer(
            self.sensor_frame_id,
            name="sensor_frame_id",
        )
        sensor_time = _finite_real(
            self.sensor_timestamp_seconds,
            name="sensor_timestamp_seconds",
        )
        simulation_time = _finite_real(
            self.simulation_time_seconds,
            name="simulation_time_seconds",
        )
        build_time = _finite_real(
            self.build_wall_seconds,
            name="build_wall_seconds",
        )
        if sensor_time < 0.0:
            raise ValueError("sensor_timestamp_seconds cannot be negative.")
        if simulation_time < 0.0:
            raise ValueError("simulation_time_seconds cannot be negative.")
        if build_time < 0.0:
            raise ValueError("build_wall_seconds cannot be negative.")
        if self.payload is None:
            raise ValueError("payload cannot be None.")
        object.__setattr__(self, "sensor_frame_id", frame_id)
        object.__setattr__(self, "sensor_timestamp_seconds", sensor_time)
        object.__setattr__(self, "simulation_time_seconds", simulation_time)
        object.__setattr__(self, "build_wall_seconds", build_time)


@dataclass(frozen=True)
class SafeSAGELiveFieldStatus(Generic[FieldPayload]):
    """Current accepted field and its fail-closed freshness decision."""

    snapshot: SafeSAGELiveFieldSnapshot[FieldPayload]
    generation: int
    reused_control_cycles: int
    field_age_seconds: float
    field_updated: bool
    fail_closed: bool
    failure_reason: str | None


class SafeSAGELiveFieldBuffer(Generic[FieldPayload]):
    """Commit complete fields atomically and reject stale reuse."""

    def __init__(
        self,
        config: SafeSAGELiveFieldConfig | None = None,
    ) -> None:
        self.config = config or SafeSAGELiveFieldConfig()
        if not isinstance(self.config, SafeSAGELiveFieldConfig):
            raise TypeError("config must be a SafeSAGELiveFieldConfig.")
        self._snapshot: SafeSAGELiveFieldSnapshot[FieldPayload] | None = None
        self._generation = 0
        self._reused_control_cycles = 0

    @property
    def initialized(self) -> bool:
        """Whether an initial complete field has been committed."""

        return self._snapshot is not None

    @property
    def generation(self) -> int:
        """Return the number of accepted field snapshots."""

        return self._generation

    @property
    def snapshot(self) -> SafeSAGELiveFieldSnapshot[FieldPayload]:
        """Return the latest complete field or raise before initialization."""

        if self._snapshot is None:
            raise RuntimeError("No live Safe-SAGE field has been committed.")
        return self._snapshot

    def initialize(
        self,
        snapshot: SafeSAGELiveFieldSnapshot[FieldPayload],
    ) -> SafeSAGELiveFieldStatus[FieldPayload]:
        """Commit the first complete field exactly once."""

        if self.initialized:
            raise RuntimeError("The live field buffer is already initialized.")
        self._validate_snapshot_type(snapshot)
        self._snapshot = snapshot
        self._generation = 1
        self._reused_control_cycles = 0
        return self._status(
            current_simulation_time_seconds=snapshot.simulation_time_seconds,
            field_updated=True,
        )

    def commit(
        self,
        snapshot: SafeSAGELiveFieldSnapshot[FieldPayload],
    ) -> SafeSAGELiveFieldStatus[FieldPayload]:
        """Atomically replace the field with one strictly newer snapshot."""

        self._validate_snapshot_type(snapshot)
        previous = self.snapshot
        tolerance = self.config.clock_tolerance_seconds
        if snapshot.sensor_frame_id <= previous.sensor_frame_id:
            raise ValueError(
                "sensor_frame_id must increase when committing a live field."
            )
        if (
            snapshot.sensor_timestamp_seconds
            <= previous.sensor_timestamp_seconds + tolerance
        ):
            raise ValueError(
                "sensor_timestamp_seconds must increase when committing "
                "a live field."
            )
        if (
            snapshot.simulation_time_seconds
            < previous.simulation_time_seconds - tolerance
        ):
            raise ValueError(
                "simulation_time_seconds cannot move backward when "
                "committing a live field."
            )
        self._snapshot = snapshot
        self._generation += 1
        self._reused_control_cycles = 0
        return self._status(
            current_simulation_time_seconds=snapshot.simulation_time_seconds,
            field_updated=True,
        )

    def reuse(
        self,
        *,
        current_simulation_time_seconds: float,
    ) -> SafeSAGELiveFieldStatus[FieldPayload]:
        """Reuse the last field once and evaluate both freshness guards."""

        current_time = self._validated_current_time(
            current_simulation_time_seconds
        )
        self._reused_control_cycles += 1
        return self._status(
            current_simulation_time_seconds=current_time,
            field_updated=False,
        )

    def inspect(
        self,
        *,
        current_simulation_time_seconds: float,
    ) -> SafeSAGELiveFieldStatus[FieldPayload]:
        """Check field age without counting another control-cycle reuse."""

        current_time = self._validated_current_time(
            current_simulation_time_seconds
        )
        return self._status(
            current_simulation_time_seconds=current_time,
            field_updated=False,
        )

    @staticmethod
    def _validate_snapshot_type(snapshot: object) -> None:
        if not isinstance(snapshot, SafeSAGELiveFieldSnapshot):
            raise TypeError(
                "snapshot must be a SafeSAGELiveFieldSnapshot."
            )

    def _validated_current_time(self, value: object) -> float:
        current_time = _finite_real(
            value,
            name="current_simulation_time_seconds",
        )
        if (
            current_time
            < self.snapshot.simulation_time_seconds
            - self.config.clock_tolerance_seconds
        ):
            raise ValueError(
                "current_simulation_time_seconds cannot precede the "
                "accepted field snapshot."
            )
        return current_time

    def _status(
        self,
        *,
        current_simulation_time_seconds: float,
        field_updated: bool,
    ) -> SafeSAGELiveFieldStatus[FieldPayload]:
        snapshot = self.snapshot
        age = max(
            0.0,
            current_simulation_time_seconds
            - snapshot.simulation_time_seconds,
        )
        too_old = (
            age
            > self.config.maximum_field_age_seconds
            + self.config.clock_tolerance_seconds
        )
        reused_too_often = (
            self._reused_control_cycles
            > self.config.maximum_reused_control_cycles
        )
        fail_closed = too_old or reused_too_often
        if too_old:
            reason = (
                "Live Safe-SAGE field age exceeded "
                f"{self.config.maximum_field_age_seconds:.3f} s."
            )
        elif reused_too_often:
            reason = (
                "Live Safe-SAGE field was reused for more than "
                f"{self.config.maximum_reused_control_cycles} control cycles."
            )
        else:
            reason = None
        return SafeSAGELiveFieldStatus(
            snapshot=snapshot,
            generation=self._generation,
            reused_control_cycles=self._reused_control_cycles,
            field_age_seconds=age,
            field_updated=field_updated,
            fail_closed=fail_closed,
            failure_reason=reason,
        )

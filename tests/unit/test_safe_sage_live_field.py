"""Unit tests for transactional live Safe-SAGE field freshness."""

import math

import pytest

from robot.safety.safe_sage_live_field import (
    SafeSAGELiveFieldBuffer,
    SafeSAGELiveFieldConfig,
    SafeSAGELiveFieldSnapshot,
)


def make_snapshot(
    frame_id: int = 10,
    *,
    sensor_time: float = 1.0,
    simulation_time: float = 1.0,
    payload: object = "field-10",
) -> SafeSAGELiveFieldSnapshot[object]:
    """Return one deterministic complete-field placeholder."""

    return SafeSAGELiveFieldSnapshot(
        sensor_frame_id=frame_id,
        sensor_timestamp_seconds=sensor_time,
        simulation_time_seconds=simulation_time,
        build_wall_seconds=0.25,
        payload=payload,
    )


def test_initial_snapshot_is_fresh_and_generation_one() -> None:
    buffer: SafeSAGELiveFieldBuffer[object] = SafeSAGELiveFieldBuffer()

    status = buffer.initialize(make_snapshot())

    assert buffer.initialized
    assert buffer.snapshot.payload == "field-10"
    assert status.generation == 1
    assert status.field_updated
    assert status.field_age_seconds == pytest.approx(0.0)
    assert status.reused_control_cycles == 0
    assert not status.fail_closed
    assert status.failure_reason is None


def test_new_complete_snapshot_commits_atomically() -> None:
    buffer: SafeSAGELiveFieldBuffer[object] = SafeSAGELiveFieldBuffer()
    buffer.initialize(make_snapshot())

    status = buffer.commit(
        make_snapshot(
            11,
            sensor_time=1.1,
            simulation_time=1.2,
            payload="field-11",
        )
    )

    assert buffer.snapshot.payload == "field-11"
    assert status.generation == 2
    assert status.field_updated
    assert status.reused_control_cycles == 0
    assert not status.fail_closed


@pytest.mark.parametrize(
    "candidate",
    [
        make_snapshot(10, sensor_time=1.1, simulation_time=1.1),
        make_snapshot(11, sensor_time=1.0, simulation_time=1.1),
        make_snapshot(11, sensor_time=1.1, simulation_time=0.9),
    ],
)
def test_invalid_commit_does_not_replace_last_complete_field(
    candidate: SafeSAGELiveFieldSnapshot[object],
) -> None:
    initial_payload = object()
    buffer: SafeSAGELiveFieldBuffer[object] = SafeSAGELiveFieldBuffer()
    buffer.initialize(make_snapshot(payload=initial_payload))

    with pytest.raises(ValueError):
        buffer.commit(candidate)

    assert buffer.generation == 1
    assert buffer.snapshot.payload is initial_payload


def test_reuse_is_allowed_inside_age_and_cycle_limits() -> None:
    buffer: SafeSAGELiveFieldBuffer[object] = SafeSAGELiveFieldBuffer(
        SafeSAGELiveFieldConfig(
            maximum_field_age_seconds=0.30,
            maximum_reused_control_cycles=2,
        )
    )
    buffer.initialize(make_snapshot())

    first = buffer.reuse(current_simulation_time_seconds=1.10)
    second = buffer.reuse(current_simulation_time_seconds=1.30)

    assert not first.field_updated
    assert first.reused_control_cycles == 1
    assert not first.fail_closed
    assert second.field_age_seconds == pytest.approx(0.30)
    assert second.reused_control_cycles == 2
    assert not second.fail_closed


def test_field_age_guard_fails_closed() -> None:
    buffer: SafeSAGELiveFieldBuffer[object] = SafeSAGELiveFieldBuffer(
        SafeSAGELiveFieldConfig(
            maximum_field_age_seconds=0.30,
            maximum_reused_control_cycles=10,
        )
    )
    buffer.initialize(make_snapshot())

    status = buffer.reuse(current_simulation_time_seconds=1.31)

    assert status.fail_closed
    assert "age exceeded" in (status.failure_reason or "")


def test_reuse_count_guard_fails_closed_even_without_clock_progress() -> None:
    buffer: SafeSAGELiveFieldBuffer[object] = SafeSAGELiveFieldBuffer(
        SafeSAGELiveFieldConfig(
            maximum_field_age_seconds=1.0,
            maximum_reused_control_cycles=1,
        )
    )
    buffer.initialize(make_snapshot())
    first = buffer.reuse(current_simulation_time_seconds=1.0)

    second = buffer.reuse(current_simulation_time_seconds=1.0)

    assert not first.fail_closed
    assert second.fail_closed
    assert "reused" in (second.failure_reason or "")


def test_new_commit_resets_stale_reuse_count() -> None:
    buffer: SafeSAGELiveFieldBuffer[object] = SafeSAGELiveFieldBuffer()
    buffer.initialize(make_snapshot())
    buffer.reuse(current_simulation_time_seconds=1.1)

    status = buffer.commit(
        make_snapshot(
            11,
            sensor_time=1.1,
            simulation_time=1.2,
        )
    )

    assert status.reused_control_cycles == 0
    assert not status.fail_closed


def test_inspect_checks_age_without_counting_another_reuse() -> None:
    buffer: SafeSAGELiveFieldBuffer[object] = SafeSAGELiveFieldBuffer(
        SafeSAGELiveFieldConfig(
            maximum_field_age_seconds=0.30,
            maximum_reused_control_cycles=2,
        )
    )
    buffer.initialize(make_snapshot())
    buffer.reuse(current_simulation_time_seconds=1.10)

    fresh = buffer.inspect(current_simulation_time_seconds=1.20)
    stale = buffer.inspect(current_simulation_time_seconds=1.31)

    assert fresh.reused_control_cycles == 1
    assert not fresh.fail_closed
    assert stale.reused_control_cycles == 1
    assert stale.fail_closed
    assert "age exceeded" in (stale.failure_reason or "")


def test_reuse_rejects_backward_simulation_clock() -> None:
    buffer: SafeSAGELiveFieldBuffer[object] = SafeSAGELiveFieldBuffer()
    buffer.initialize(make_snapshot())

    with pytest.raises(ValueError, match="cannot precede"):
        buffer.reuse(current_simulation_time_seconds=0.9)


def test_access_before_initialization_fails() -> None:
    buffer: SafeSAGELiveFieldBuffer[object] = SafeSAGELiveFieldBuffer()

    with pytest.raises(RuntimeError, match="No live Safe-SAGE field"):
        _ = buffer.snapshot


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("maximum_field_age_seconds", 0.0),
        ("maximum_field_age_seconds", math.inf),
        ("maximum_reused_control_cycles", -1),
        ("maximum_reused_control_cycles", True),
        ("clock_tolerance_seconds", -1.0),
    ],
)
def test_config_rejects_invalid_values(
    field_name: str,
    invalid_value: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        SafeSAGELiveFieldConfig(**{field_name: invalid_value})


def test_snapshot_rejects_none_payload() -> None:
    with pytest.raises(ValueError, match="payload"):
        make_snapshot(payload=None)


def test_initialize_only_accepts_snapshot_type() -> None:
    buffer: SafeSAGELiveFieldBuffer[object] = SafeSAGELiveFieldBuffer()

    with pytest.raises(TypeError, match="SafeSAGELiveFieldSnapshot"):
        buffer.initialize("field")  # type: ignore[arg-type]

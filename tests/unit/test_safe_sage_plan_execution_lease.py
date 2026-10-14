import pytest

from robot.safety.safe_sage_runtime import SafeSAGEPlanExecutionLease


def test_older_optimizer_candidate_cannot_execute_before_current_validation():
    lease = SafeSAGEPlanExecutionLease(1.0, maximum_candidate_age_seconds=0.5)
    assert lease.candidate_available(1.4)
    assert not lease.can_execute(1.4, 3, field_fresh=True)
    lease.record_validation(1.4, 3, accepted=True)
    assert lease.can_execute(1.4, 3, field_fresh=True)
    assert lease.source_timestamp_seconds == 1.0


def test_changed_or_stale_field_immediately_blocks_execution():
    lease = SafeSAGEPlanExecutionLease(1.0, maximum_candidate_age_seconds=0.5)
    lease.record_validation(1.1, 3, accepted=True)
    assert not lease.can_execute(1.12, 4, field_fresh=True)
    assert not lease.can_execute(1.12, 3, field_fresh=False)
    assert lease.needs_validation(1.12, 4)
    lease.record_validation(1.12, 4, accepted=False)
    assert not lease.can_execute(1.12, 4, field_fresh=True)


def test_periodic_validation_and_original_candidate_expiry_are_separate():
    lease = SafeSAGEPlanExecutionLease(1.0, maximum_candidate_age_seconds=0.5)
    lease.record_validation(1.1, 3, accepted=True)
    assert lease.needs_validation(1.26, 3)
    assert not lease.can_execute(1.31, 3, field_fresh=True)
    lease.record_validation(1.4, 3, accepted=True)
    assert lease.can_execute(1.49, 3, field_fresh=True)
    assert not lease.can_execute(1.51, 3, field_fresh=True)
    lease.record_validation(1.51, 3, accepted=True)
    assert lease.validated_timestamp_seconds is None
    assert not lease.can_execute(1.51, 3, field_fresh=True)


def test_future_candidate_and_clock_reversal_never_execute():
    lease = SafeSAGEPlanExecutionLease(1)
    assert not lease.candidate_available(0.9)
    lease.record_validation(1.1, 1, accepted=True)
    assert not lease.can_execute(1.09, 1, field_fresh=True)


@pytest.mark.parametrize(
    "changes",
    [
        {"maximum_candidate_age_seconds": 0},
        {"source_timestamp_seconds": -1},
        {"execution_interval_seconds": 0.1},
        {"revalidation_interval_seconds": 0.2},
        {"maximum_candidate_age_seconds": True},
        {"validated_timestamp_seconds": 1.1},
    ],
)
def test_invalid_or_prevalidated_leases_are_rejected(changes):
    values = {"source_timestamp_seconds": 1, **changes}
    with pytest.raises((ValueError, TypeError)):
        SafeSAGEPlanExecutionLease(**values)

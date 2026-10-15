from threading import Event

import pytest

from robot.perception.live_inference import LiveInferenceRunner


def completed(runner):
    try:
        runner.pending[3].result(timeout=2)
    except RuntimeError:
        pass


def test_control_thread_can_continue_while_single_worker_is_busy():
    entered, release = Event(), Event()

    def work(packet):
        entered.set()
        assert release.wait(timeout=2)
        return packet * 2

    runner = LiveInferenceRunner(work)
    try:
        assert runner.submit(frame_id=1, sensor_timestamp_seconds=1.0, packet=2)
        assert runner.busy
        assert entered.wait(timeout=1)
        assert runner.poll(current_sensor_time_seconds=1.1) is None
        assert not runner.submit(frame_id=2, sensor_timestamp_seconds=1.1, packet=3)
        release.set()
        completed(runner)
        result = runner.poll(current_sensor_time_seconds=1.2)
        assert not runner.busy
        assert result.accepted and result.value == 4
        assert result.sensor_timestamp_seconds == 1.0
        assert runner.submit(frame_id=2, sensor_timestamp_seconds=1.1, packet=3)
    finally:
        release.set()
        runner.close()


@pytest.mark.parametrize("now,reason", [(1.5, "stale"), (0.9, "future")])
def test_results_are_checked_at_acquisition_time_and_hidden_when_invalid(now, reason):
    runner = LiveInferenceRunner(lambda packet: packet)
    try:
        runner.submit(frame_id=1, sensor_timestamp_seconds=1.0, packet="field")
        completed(runner)
        result = runner.poll(current_sensor_time_seconds=now)
        assert not result.accepted and result.value is None
        assert reason in result.rejection_reason
        assert result.sensor_timestamp_seconds == 1.0
    finally:
        runner.close()


def test_failed_inference_is_not_an_empty_safe_result():
    def fail(_):
        raise RuntimeError("mask rejected")

    runner = LiveInferenceRunner(fail)
    try:
        runner.submit(frame_id=1, sensor_timestamp_seconds=1.0, packet=None)
        completed(runner)
        result = runner.poll(current_sensor_time_seconds=1.1)
        assert not result.accepted and result.value is None
        assert "mask rejected" in result.rejection_reason
    finally:
        runner.close()


@pytest.mark.parametrize(
    "now, accepted", [(1.35, True), (1.3500000001, True), (1.350001, False)]
)
def test_deadline_roundoff_does_not_retime_or_relax_meaningfully_late_results(
    now, accepted
):
    runner = LiveInferenceRunner(lambda packet: packet)
    try:
        runner.submit(frame_id=1, sensor_timestamp_seconds=1.0, packet="field")
        completed(runner)
        result = runner.poll(current_sensor_time_seconds=now)
        assert result.accepted is accepted
        assert result.sensor_timestamp_seconds == 1.0
        assert (result.value == "field") is accepted
    finally:
        runner.close()


@pytest.mark.parametrize("frame,time", [(0, -0.1), (True, 0.0), (1, float("nan"))])
def test_invalid_epochs_are_rejected(frame, time):
    runner = LiveInferenceRunner(lambda packet: packet)
    try:
        with pytest.raises(ValueError):
            runner.submit(frame_id=frame, sensor_timestamp_seconds=time, packet=None)
    finally:
        runner.close()

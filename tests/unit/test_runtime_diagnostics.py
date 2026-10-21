import cProfile
import json
import pstats
import threading

import pytest

from robot.evaluation.runtime_diagnostics import profile_call, numeric_runtime_contract
from robot.evaluation import runtime_diagnostics


def test_profiling_preserves_result_and_uses_readable_stats(tmp_path):
    token = object()
    path = tmp_path / "job.pstats"

    def work(*, value):
        return value

    assert profile_call(work, value=token) is token
    assert (
        profile_call(work, value=token, profile_path=path, minimum_seconds=0) is token
    )
    stats = pstats.Stats(str(path))
    assert any(key[2] == "work" for key in stats.stats)


def test_profiling_preserves_original_exception(tmp_path):
    def fail():
        raise RuntimeError("original worker rejection")

    path = tmp_path / "failure.pstats"
    with pytest.raises(RuntimeError, match="original worker rejection"):
        profile_call(fail, profile_path=path, minimum_seconds=0)
    assert path.is_file()
    # A failed worker must not retain the profiler slot.
    next_path = tmp_path / "after-failure.pstats"
    assert profile_call(lambda: 42, profile_path=next_path, minimum_seconds=0) == 42
    assert next_path.is_file()


def test_concurrent_profile_calls_run_once_without_waiting(tmp_path):
    entered = threading.Event()
    release = threading.Event()
    second_done = threading.Event()
    results = {}
    errors = []
    calls = []

    def first_work():
        calls.append("first")
        entered.set()
        if not release.wait(timeout=10):
            raise RuntimeError("test did not release first worker")
        return "first result"

    def second_work():
        calls.append("second")
        return "second result"

    def run(name, work, done=None):
        try:
            results[name] = profile_call(
                work, profile_path=tmp_path / f"{name}.pstats", minimum_seconds=0
            )
        except BaseException as error:
            errors.append(error)
        finally:
            if done is not None:
                done.set()

    first = threading.Thread(target=run, args=("first", first_work))
    second = threading.Thread(target=run, args=("second", second_work, second_done))
    first.start()
    try:
        assert entered.wait(timeout=5), errors
        second.start()
        assert second_done.wait(timeout=5), "overlapping work waited for profiling"
        assert first.is_alive(), "first worker must still hold the profiling slot"
        assert results["second"] == "second result"
    finally:
        release.set()
        first.join(timeout=5)
        if second.ident is not None:
            second.join(timeout=5)
    assert not first.is_alive() and not second.is_alive()
    assert not errors
    assert results["first"] == "first result"
    assert calls == ["first", "second"]
    pstats.Stats(str(tmp_path / "first.pstats"))
    assert not (tmp_path / "second.pstats").exists()


def test_nested_profiling_runs_inner_work_once_without_profile(tmp_path):
    calls = []
    token = object()

    def inner(value):
        calls.append(value)
        return value

    def outer():
        return profile_call(
            inner, token, profile_path=tmp_path / "inner.pstats", minimum_seconds=0
        )

    assert (
        profile_call(outer, profile_path=tmp_path / "outer.pstats", minimum_seconds=0)
        is token
    )
    assert calls == [token]
    assert not (tmp_path / "inner.pstats").exists()
    pstats.Stats(str(tmp_path / "outer.pstats"))


def test_external_profiler_collision_skips_only_instrumentation(tmp_path, monkeypatch):
    calls = []

    class ConflictingProfile:
        def enable(self):
            raise ValueError("Another profiling tool is already active")

    with monkeypatch.context() as patch:
        patch.setattr(runtime_diagnostics.cProfile, "Profile", ConflictingProfile)
        assert (
            profile_call(
                lambda value: calls.append(value) or value,
                42,
                profile_path=tmp_path / "conflict.pstats",
                minimum_seconds=0,
            )
            == 42
        )
    assert calls == [42]
    assert not (tmp_path / "conflict.pstats").exists()
    path = tmp_path / "recovered.pstats"
    assert profile_call(lambda: 43, profile_path=path, minimum_seconds=0) == 43
    pstats.Stats(str(path))


def test_existing_external_profile_keeps_collecting_around_unprofiled_call(tmp_path):
    external = cProfile.Profile()
    path = tmp_path / "nested-external.pstats"
    calls = []

    def before():
        calls.append("before")

    def work(*, source_epoch):
        calls.append("work")
        return source_epoch

    def after():
        calls.append("after")

    external.enable()
    try:
        before()
        result = profile_call(
            work, source_epoch=78.9, profile_path=path, minimum_seconds=0
        )
        after()
    finally:
        external.disable()
    assert result == 78.9
    assert calls == ["before", "work", "after"]
    names = {key[2] for key in pstats.Stats(external).stats}
    assert {"before", "work", "after"} <= names
    assert not path.exists()


def test_worker_value_error_is_never_retried_as_profiler_collision(tmp_path):
    calls = []

    def fail():
        calls.append("called")
        raise ValueError("Another profiling tool is already active")

    with pytest.raises(ValueError, match="Another profiling tool is already active"):
        profile_call(fail, profile_path=tmp_path / "worker.pstats", minimum_seconds=0)
    assert calls == ["called"]
    pstats.Stats(str(tmp_path / "worker.pstats"))


def test_unrelated_profiler_enable_failure_is_not_hidden(tmp_path, monkeypatch):
    calls = []

    class BrokenProfile:
        def enable(self):
            raise ValueError("unrelated instrumentation failure")

    with monkeypatch.context() as patch:
        patch.setattr(runtime_diagnostics.cProfile, "Profile", BrokenProfile)
        with pytest.raises(ValueError, match="unrelated instrumentation failure"):
            profile_call(
                lambda: calls.append("must not run"),
                profile_path=tmp_path / "broken.pstats",
            )
    assert calls == []
    path = tmp_path / "after-enable-failure.pstats"
    assert profile_call(lambda: 44, profile_path=path, minimum_seconds=0) == 44
    pstats.Stats(str(path))


def test_profile_write_failure_cannot_replace_control_result(tmp_path):
    parent = tmp_path / "file"
    parent.write_text("already a file")
    with pytest.warns(RuntimeWarning, match="not saved"):
        assert (
            profile_call(
                lambda: 42, profile_path=parent / "job.pstats", minimum_seconds=0
            )
            == 42
        )


def test_numeric_contract_reports_loaded_modules_without_changing_limits():
    data = numeric_runtime_contract()
    assert data["modules"]["numpy"]["path"]
    assert "available" in data["thread_pools"]
    assert "no thread limits changed" in data["scope"]
    json.dumps(data, allow_nan=False)

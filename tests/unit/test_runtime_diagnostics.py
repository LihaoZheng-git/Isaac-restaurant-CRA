import pstats
import json

import pytest

from robot.evaluation.runtime_diagnostics import profile_call, numeric_runtime_contract


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

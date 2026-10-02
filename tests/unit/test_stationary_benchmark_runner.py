import json
import subprocess

import pytest

from scripts.evaluation.run_stationary_perception_benchmark import _read_payload, _run_script


def test_nonfinite_component_report_is_rejected(tmp_path):
    path = tmp_path / "report.json"
    path.write_text('{"passed": false, "error": Infinity}', encoding="utf-8")
    with pytest.raises(ValueError, match="nonfinite"):
        _read_payload(path)


def test_unknown_evidence_is_not_changed_to_numeric_zero(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"passed": False, "fall_height_m": None}), encoding="utf-8")
    assert _read_payload(path)["fall_height_m"] is None
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="object"):
        _read_payload(path)


def test_timeout_cannot_be_reported_as_process_success(tmp_path, monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", timeout)
    assert _run_script(tmp_path / "python.exe", "unstarted.py", tmp_path / "capture", {}, 1) == 124
    assert (tmp_path / "capture" / "process.log").is_file()

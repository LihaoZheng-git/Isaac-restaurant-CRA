import hashlib
import sys

import pytest

from experiment import run
from experiment.models.__main__ import download
from experiment.models import MODEL_FILES
from experiment.models import __main__ as setup


def test_verified_download_can_be_reused_offline(tmp_path):
    source = tmp_path / "upstream.pt"
    target = tmp_path / "model.pt"
    source.write_bytes(b"model weights")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    download(source.as_uri(), target, expected)
    source.unlink()
    download(source.as_uri(), target, expected)
    assert target.read_bytes() == b"model weights"


def test_bad_download_leaves_no_checkpoint_or_partial_file(tmp_path):
    source = tmp_path / "upstream.pt"
    source.write_bytes(b"unexpected weights")
    target = tmp_path / "model.pt"
    with pytest.raises(ValueError, match="SHA-256"):
        download(source.as_uri(), target, "0" * 64)
    assert not target.exists()
    assert not list(tmp_path.glob("*.part"))


def test_existing_different_checkpoint_is_preserved(tmp_path):
    target = tmp_path / "model.pt"
    target.write_bytes(b"custom weights")
    with pytest.raises(ValueError, match="Move it aside"):
        download(target.as_uri(), target, "0" * 64)
    assert target.read_bytes() == b"custom weights"


def test_check_reports_missing_models_without_creating_directory(tmp_path, monkeypatch, capsys):
    target = tmp_path / "missing"
    for variable, _ in MODEL_FILES.values():
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(sys, "argv", ["models", "--model-root", str(target), "--check"])
    with pytest.raises(SystemExit) as result:
        setup.main()
    assert result.value.code == 1
    assert not target.exists()
    error = capsys.readouterr().err
    assert all(filename in error for _, filename in MODEL_FILES.values())


def test_check_honors_existing_checkpoint_paths_and_detects_hash_mismatch(tmp_path, monkeypatch):
    expected = {}
    for key, (variable, _) in MODEL_FILES.items():
        path = tmp_path / key
        path.write_bytes(key.encode())
        monkeypatch.setenv(variable, str(path))
        expected[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(setup, "recorded_model_hashes", lambda: expected)
    monkeypatch.setattr(sys, "argv", ["models", "--check"])
    setup.main()
    (tmp_path / "mass").write_bytes(b"wrong revision")
    with pytest.raises(SystemExit) as result:
        setup.main()
    assert result.value.code == 1


def test_mission_preflight_rejects_substituted_weights(tmp_path, monkeypatch):
    scene = tmp_path / "mission.usda"
    survey = tmp_path / "survey.json"
    for path in (scene, survey, tmp_path / "authored-height-column-mask.png"):
        path.touch()
    scene.with_suffix(".json").write_text('{"additions":"mission.usda","source_sha256":{}}')
    command = ["--restaurant-scene", str(scene), "--restaurant-layout-survey", str(survey)]
    monkeypatch.setattr(run, "read_recipe", lambda *args: ({"left": command}, {}))
    expected = {}
    for key, (variable, filename) in MODEL_FILES.items():
        monkeypatch.delenv(variable, raising=False)
        (tmp_path / filename).write_bytes(key.encode())
        expected[key] = hashlib.sha256(key.encode()).hexdigest()
    monkeypatch.setattr(run, "recorded_model_hashes", lambda: expected)
    run.prepare_inputs(tmp_path, tmp_path)
    (tmp_path / "visual-mass.pt").write_bytes(b"different model")
    with pytest.raises(ValueError, match="Checkpoint differs from the recorded demo: .*visual-mass.pt"):
        run.prepare_inputs(tmp_path, tmp_path)

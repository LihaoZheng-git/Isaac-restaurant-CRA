import sys
from pathlib import Path

import pytest

from experiment import run


@pytest.mark.parametrize(("platform", "script"), [("linux", "python.sh"), ("win32", "python.bat")])
def test_installation_folder_selects_platform_launcher(tmp_path, monkeypatch, platform, script):
    installation = tmp_path / "Isaac Sim"
    installation.mkdir()
    launcher = installation / script
    launcher.touch()
    launcher.chmod(0o755)
    interpreter = str(tmp_path / "vision env" / "python")
    output = tmp_path / "recorded mission"
    config = tmp_path / "experiment/configs/mission.json"
    config.parent.mkdir(parents=True)
    config.write_text("{}")
    monkeypatch.setattr(run, "ROOT", tmp_path)
    monkeypatch.setattr(run, "CONFIG", config)
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(sys, "executable", interpreter)
    monkeypatch.setenv("ISAAC_SIM_USE_D3D12", "1")
    monkeypatch.setattr(sys, "argv", ["run", "--isaac-root", str(installation), "--output", str(output)])
    monkeypatch.setattr(run, "prepare_inputs", lambda *args: ({}, {}, {}))

    class LauncherReached(Exception):
        pass

    def capture(command, *, env, **kwargs):
        assert Path(command[0]) == launcher.resolve()
        assert Path(command[command.index("--scene-root") + 1]) == (tmp_path / "experiment/scene/prepared").resolve()
        assert Path(command[command.index("--output") + 1]) == output.resolve()
        assert env["ISAAC_SIM_USE_D3D12"] == ("1" if platform == "win32" else "0")
        assert env["PYTHONEXE"] == (f'"{interpreter}"' if platform == "win32" else interpreter)
        raise LauncherReached

    monkeypatch.setattr(run.subprocess, "Popen", capture)
    with pytest.raises(LauncherReached):
        run.main()

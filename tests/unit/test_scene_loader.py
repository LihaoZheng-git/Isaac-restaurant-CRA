"""Unit tests for composed-scene contract checks."""

from robot.scene.contracts import find_missing_prim_paths


class _FakePrim:
    def __init__(self, valid: bool) -> None:
        self._valid = valid

    def IsValid(self) -> bool:
        return self._valid


class _FakeStage:
    def __init__(self, available_paths: set[str]) -> None:
        self._available_paths = available_paths

    def GetPrimAtPath(self, path: str) -> _FakePrim:
        return _FakePrim(path in self._available_paths)


def test_find_missing_prim_paths_preserves_required_order() -> None:
    stage = _FakeStage({"/World/Robot/chassis"})

    missing = find_missing_prim_paths(
        stage,
        (
            "/World/Robot/chassis",
            "/World/Robot/camera",
            "/World/Robot/lidar",
        ),
    )

    assert missing == (
        "/World/Robot/camera",
        "/World/Robot/lidar",
    )


def test_find_missing_prim_paths_accepts_complete_stage() -> None:
    required = ("/World/Robot/chassis", "/World/Robot/camera")
    stage = _FakeStage(set(required))

    assert find_missing_prim_paths(stage, required) == ()

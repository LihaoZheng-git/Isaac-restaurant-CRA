"""Guard the assets that must be usable straight from a Git checkout."""

from experiment.scene import prepare


def test_bundled_assets_match_the_recorded_demo():
    expected = prepare.load("inputs.json")["live"]
    for name, digest in expected.items():
        if name.startswith(("restaurant_live_v1/", "restaurant_live_v5/")):
            assert prepare.sha256(prepare.PROJECT_LIVE / name) == digest, name
    assert all((prepare.PROJECT_LIVE / name).is_file() for name in prepare.NOTICES)


def test_live_setup_restores_missing_file_and_keeps_valid_files(tmp_path, monkeypatch):
    bundled = tmp_path / "bundled"
    prop = bundled / "restaurant_live_v1/dish.usdc"
    prop.parent.mkdir(parents=True)
    prop.write_bytes(b"dish")
    (bundled / "LICENSE.md").write_text("credits", encoding="utf-8")
    expected = {"restaurant_live_v1/dish.usdc": prepare.sha256(prop)}
    monkeypatch.setattr(prepare, "PROJECT_LIVE", bundled)
    monkeypatch.setattr(prepare, "NOTICES", ("LICENSE.md",))
    monkeypatch.setattr(prepare.vendor, "adult_characters", lambda root, manifest: None)
    cache = tmp_path / "cache"
    live = prepare.install_live_assets(cache, expected)
    copied = live / "restaurant_live_v1/dish.usdc"
    timestamp = copied.stat().st_mtime_ns

    (live / "LICENSE.md").unlink()
    assert prepare.install_live_assets(cache, expected) == live
    assert copied.stat().st_mtime_ns == timestamp
    assert (live / "LICENSE.md").read_text(encoding="utf-8") == "credits"

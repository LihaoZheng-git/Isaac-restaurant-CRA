"""Gate reuse is tied to runtime bytes, not a launcher/report-format revision."""
import json

from scripts.evaluation import run_restaurant_demo as runner


def test_launcher_only_change_retains_exact_runtime_scene_config_binding():
    hashes = {"robot/controller.py": "control", runner.LAUNCHER: "new"}
    scenes = {"scene.usda": "scene"}
    meta = dict(case="parallel", source_revision="old", config_sha256="config",
                source_sha256={**hashes, runner.LAUNCHER: "old"}, scene_sha256=scenes)
    execution = dict(passed=True, source_unchanged=True)
    def match(m=meta, e=execution, h=hashes, s=scenes, c="config"):
        return runner.prerequisite_matches(m, e, {"passed": True}, "parallel", c, h, s)
    assert match()
    assert not match(h={**hashes, "robot/controller.py": "changed"})
    assert not match(h={**hashes, "robot/new.py": "new"})
    assert not match(h={runner.LAUNCHER: "new"})
    assert not match(s={"scene.usda": "changed"})
    assert not match(c="changed")
    assert not match(e={"passed": False, "source_unchanged": True})
    assert not match(e={"passed": True, "source_unchanged": False})


def test_component_reads_original_numeric_clearance_contract(tmp_path):
    native=tmp_path/"native"
    native.mkdir()
    (native/"human-sensor-validation.json").write_text(json.dumps({"passed": True}))
    setup=native/"robot-setup-validation.json"
    setup.write_text(json.dumps({"configured_clearance_m": .8, "settled_clearance_m": .9}))
    assert runner.component_reports_pass(tmp_path)
    for value in (.02, -.1, float('nan'), None, True):
        setup.write_text(json.dumps({"configured_clearance_m": .8, "settled_clearance_m": value}))
        assert not runner.component_reports_pass(tmp_path)


def test_failed_or_unrecognized_component_report_cannot_be_ignored(tmp_path):
    native=tmp_path/"native"
    native.mkdir()
    assert not runner.component_reports_pass(tmp_path)
    (native/"human-sensor-validation.json").write_text(json.dumps({"passed": True}))
    (native/"robot-setup-validation.json").write_text(json.dumps(
        {"configured_clearance_m": .8, "settled_clearance_m": .9}))
    extra=native/"new-validation.json"
    extra.write_text(json.dumps({"passed": False}))
    assert not runner.component_reports_pass(tmp_path)
    extra.write_text(json.dumps({"unknown": True}))
    assert not runner.component_reports_pass(tmp_path)

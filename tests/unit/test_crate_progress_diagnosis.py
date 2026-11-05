"""Diagnostic recording must preserve the actual V3 controller/scene recipe."""
from test_crate_v3_controller import command, parse


def test_diagnostic_changes_only_recording_and_duration(monkeypatch):
    reference = vars(parse(monkeypatch, command('restaurant_crate_v3_controller'))['args'])
    captured = parse(monkeypatch, command('restaurant_crate_progress_diagnosis') + ['--motion'])
    actual = vars(captured['args'])
    assert captured['crate_v3_profile']['passed']
    assert actual['maximum_seconds'] == 8.
    assert actual['save_planner_replay'] and not actual['save_failed_planner_replay']
    assert actual['planner_replay_source_window'] == [11.5, 15.5]
    allowed = {'motion', 'maximum_seconds', 'save_planner_replay',
               'save_failed_planner_replay', 'planner_replay_source_window'}
    assert {k: v for k, v in actual.items() if k not in allowed} == {
        k: v for k, v in reference.items() if k not in allowed}

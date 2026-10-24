import hashlib
import json
import pickle
from types import SimpleNamespace
from dataclasses import replace

import numpy as np
import pytest

from test_laplace_guidance_tracking_mpc import config
from test_safe_sage_cbf_filter import make_filter
from test_safe_sage_runtime import make_runtime, make_observation, make_provider
from robot.evaluation.planner_replay import save_planner_replay
from robot.safety.laplace_guidance_tracking_mpc import LaplaceGuidanceTrackingMPC, LaplaceGuidanceTrackingConfig
from robot.safety.safe_sage_runtime import SafeSAGEControlRuntime
from robot.safety.safe_sage_unicycle_cbf_filter import SafeSAGEUnicycleCBFFilter


@pytest.mark.parametrize("position,accepted", [((0., 0., 0.), False), ((-2., 2., 0.), True)])
def test_native_numeric_snapshot_replays_same_plan_without_renewing_field(tmp_path, position, accepted):
    perception = make_runtime()
    perception.initialize(make_observation(1, 1.), make_provider())
    paper = make_filter()
    control = SafeSAGEControlRuntime(
        mpc=LaplaceGuidanceTrackingMPC(paper, config(),
            guidance_config=LaplaceGuidanceTrackingConfig(weight=10, reference_mode="planar_forecast_velocity")),
        backup_filter=SafeSAGEUnicycleCBFFilter(paper))
    state = np.array(position)
    request = (state, np.tile((.2, 0.), (config().horizon_steps, 1)),
        perception.field, None, state[:2] + (1., 0.))
    result = control.plan(request[0], request[1], request[2], nominal_goal_grid_xy=request[4])
    assert result.success is accepted
    metadata = save_planner_replay(tmp_path, control, request, result)
    blob = (tmp_path / metadata["file"]).read_bytes()
    assert hashlib.sha256(blob).hexdigest() == metadata["sha256"]
    restored = pickle.loads(blob)  # This test just generated the trusted file.
    copied = restored["request"]
    replay = restored["control"].plan(copied[0], copied[1], copied[2],
        initial_command_guess=copied[3], nominal_goal_grid_xy=copied[4])
    np.testing.assert_allclose(replay.command_sequence, result.command_sequence, atol=1e-10)
    np.testing.assert_allclose(replay.predicted_safety_values, result.predicted_safety_values, atol=1e-10)
    assert replay.success is result.success and replay.status == result.status
    assert copied[2].sensor_frame_id == perception.field.sensor_frame_id == 1
    assert copied[2].sensor_timestamp_seconds == perception.field.sensor_timestamp_seconds == 1.
    np.testing.assert_array_equal(copied[2].pipeline.h_full.values, perception.field.pipeline.h_full.values)
    stored = json.loads((tmp_path / "planner-000000.json").read_text())
    assert stored == metadata and stored["sensor_timestamp_seconds"] == 1.


def test_first_selected_snapshot_retains_original_request_and_is_not_replaced(tmp_path):
    perception = make_runtime()
    perception.initialize(make_observation(1, 1.), make_provider())
    paper = make_filter()
    control = SafeSAGEControlRuntime(
        mpc=LaplaceGuidanceTrackingMPC(paper, config(),
            guidance_config=LaplaceGuidanceTrackingConfig(weight=10)),
        backup_filter=SafeSAGEUnicycleCBFFilter(paper))
    f = replace(perception.field, motion_passing_reference_track_ids=(0,),
        motion_passing_reference_sides=("left",),
        motion_passing_reference_source_timestamp_seconds=.9)
    request = (np.zeros(3), (.2, 0.), f, None, (1., 0.))
    result = control.plan(request[0], request[1], f, nominal_goal_grid_xy=request[4])
    save_planner_replay(tmp_path, control, request, result)
    path = tmp_path / "first-selected.pickle"
    original = path.read_bytes()
    request = (np.ones(3), (.1, 0.), f, None, (2., 0.))
    save_planner_replay(tmp_path, control, request, result)
    assert path.read_bytes() == original
    metadata = json.loads((tmp_path / "first-selected.json").read_text())
    assert hashlib.sha256(original).hexdigest() == metadata["sha256"]
    assert metadata["selected_track_ids"] == [0]
    assert metadata["selected_sides"] == ["left"]
    assert metadata["original_selection_timestamp_seconds"] == .9
    restored = pickle.loads(original)  # This test just generated the trusted file.
    np.testing.assert_array_equal(restored["request"][0], np.zeros(3))
    assert restored["request"][2].sensor_timestamp_seconds == 1.


@pytest.mark.parametrize('selected,tier,name', [(False,None,'first-rear-unavailable'),
    (True,'checked_turn_first_fallback','first-rear-turn-fallback')])
def test_rear_debug_snapshot_preserves_first_failure_or_fallback_request(tmp_path, selected, tier, name):
    perception=make_runtime()
    perception.initialize(make_observation(1,1.),make_provider())
    paper=make_filter()
    control=SafeSAGEControlRuntime(mpc=LaplaceGuidanceTrackingMPC(paper,config()),
        backup_filter=SafeSAGEUnicycleCBFFilter(paper))
    result=SimpleNamespace(success=True,status='diagnostic',first_command=(0.,.2),
        rear_pass_reference=dict(context=dict(mode='rear'),selected=selected,selection_tier=tier,reason='diagnostic'))
    request=(np.zeros(3),(.2,0.),perception.field,None,(1.,0.))
    save_planner_replay(tmp_path,control,request,result)
    original=(tmp_path/f'{name}.pickle').read_bytes()
    save_planner_replay(tmp_path,control,(np.ones(3),*request[1:]),result)
    assert (tmp_path/f'{name}.pickle').read_bytes() == original
    meta=json.loads((tmp_path/f'{name}.json').read_text())
    assert hashlib.sha256(original).hexdigest() == meta['sha256']
    restored=pickle.loads(original)  # This test generated the trusted bytes.
    np.testing.assert_array_equal(restored['request'][0],np.zeros(3))

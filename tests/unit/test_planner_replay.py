import hashlib
import json
import pickle

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

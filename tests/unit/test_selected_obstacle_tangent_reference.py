from dataclasses import replace

import numpy as np
import pytest

from test_selected_tangent_reference import planner, field, runtime_field
from robot.safety.safe_sage_runtime import SafeSAGEControlRuntime
from robot.safety.safe_sage_unicycle_cbf_filter import SafeSAGEUnicycleCBFFilter


def selected_planner():
    m = planner(1.)
    m.guidance_config = replace(m.guidance_config, reference_tangent_basis="selected_observed_obstacle")
    m.config = replace(m.config, horizon_steps=6)
    return m


@pytest.mark.parametrize("side,sign", [("left", 1.), ("right", -1.)])
def test_selected_tangent_keeps_requested_side_when_combined_field_projection_opposes_it(side, sign):
    m = selected_planner()
    kwargs = field(-sign)
    nominal = np.tile((.2, 0.), (6, 1))
    original = nominal.copy()
    r = m.plan(np.zeros(3), nominal, nominal_goal_grid_xy=(.2, 0.),
        selected_passing_tangents=((7, (1., 0.), side),), **kwargs)
    assert r.success and r.guidance_tracking.active
    assert sign*r.angular_velocity > .02
    assert r.guidance_tracking.reference_tangent_basis == "selected_observed_obstacle"
    checked = m._evaluate_trajectory(np.zeros(3), r.command_sequence,
        temporal_safety_derivative=0., **kwargs)
    assert checked is not None and m._trajectory_is_safe(checked)
    np.testing.assert_array_equal(nominal, original)


@pytest.mark.parametrize("refs", [(), ((True, (1., 0.), "left"),),
    ((7, (np.nan, 0.), "left"),), ((7, (1., 0.), "unknown"),),
    ((7, (1., 0.), "left"), (8, (2., 0.), "right"))])
def test_missing_or_ambiguous_selected_reference_is_rejected(refs):
    with pytest.raises(ValueError, match="selected observed tangent"):
        selected_planner().plan(np.zeros(3), (.2, 0.), nominal_goal_grid_xy=(1., 0.),
            selected_passing_tangents=refs, **field())


@pytest.mark.parametrize("sides,epoch", [(("unknown",), 1.), (("left", "right"), 1.),
    (("left",), None), (("left",), True), (("left",), 1.1), (("left",), -8.), ((), 1.)])
def test_selected_side_and_original_epoch_contract_is_not_optional(sides, epoch):
    with pytest.raises(ValueError, match="selected tangent"):
        replace(runtime_field(), motion_passing_reference_track_ids=(0,),
            motion_passing_reference_sides=sides,
            motion_passing_reference_source_timestamp_seconds=epoch)


def test_selected_basis_without_side_metadata_falls_back_locally_and_keeps_source():
    f = replace(runtime_field(), motion_passing_reference_track_ids=(0,))
    m = selected_planner()
    runtime = SafeSAGEControlRuntime(mpc=m, backup_filter=SafeSAGEUnicycleCBFFilter(m.paper_filter))
    r = runtime.plan(np.zeros(3), (.2, 0.), f, nominal_goal_grid_xy=(1., 0.))
    assert r.guidance_tracking.tangent_velocity_fraction == 0.
    assert runtime.mpc.guidance_config.tangent_velocity_fraction == 1.
    assert f.sensor_timestamp_seconds == 1.
    assert f.motion_passing_reference_sides == ()


def test_selected_basis_cannot_silently_use_legacy_combined_field_search():
    with pytest.raises(ValueError, match="no legacy reference search"):
        replace(selected_planner().guidance_config, reference_search_width=2)

from dataclasses import fields, replace

import numpy as np
import pytest
from test_laplace_guidance_tracking_mpc import config, field
from test_safe_sage_cbf_filter import make_filter, make_fields, make_guidance_field

from robot.safety.checked_guidance_reference_search import search_checked_guidance_reference, reference_motion_rejection
from robot.safety.laplace_guidance_tracking_mpc import LaplaceGuidanceTrackingConfig, LaplaceGuidanceTrackingMPC
from robot.safety.safe_sage_unicycle_mpc import SafeSAGEUnicycleMPC


def planner(**kwargs):
    return LaplaceGuidanceTrackingMPC(make_filter(), config(),
        guidance_config=LaplaceGuidanceTrackingConfig(weight=10,
            reference_mode="planar_forecast_velocity", reference_search_width=4, **kwargs))


def search(mpc=None, **kwargs):
    return search_checked_guidance_reference(mpc or planner(), np.zeros(3), np.array([2.,0.]),
        .2, {**field(g=(1.,0.)), "temporal_safety_derivative":0., **kwargs}, width=4, candidate_limit=600)


@pytest.mark.parametrize("changes", [dict(reference_search_width=-1), dict(reference_search_width=9),
    dict(reference_search_width=True), dict(reference_search_width=1.5),
    dict(reference_search_maximum_candidates=0), dict(reference_search_maximum_candidates=True),
    dict(reference_search_maximum_candidates=1.5)])
def test_invalid_search_configuration_rejected(changes):
    with pytest.raises(ValueError):
        LaplaceGuidanceTrackingConfig(**changes)


def test_search_mode_cannot_silently_change_a_legacy_reference():
    with pytest.raises(ValueError, match="planar forecast"):
        LaplaceGuidanceTrackingConfig(reference_search_width=4)


def test_only_original_length_exact_safe_horizon_is_returned_and_fields_unchanged():
    mpc = planner()
    options = field(g=(1.,0.))
    before = options["h_full"].values.copy()
    result = search_checked_guidance_reference(mpc, np.zeros(3), np.array([2.,0.]), .2,
        {**options,"temporal_safety_derivative":0.}, width=4, candidate_limit=600)
    value = result.trajectory
    assert value is not None and len(value.commands) == mpc.config.horizon_steps
    assert mpc._trajectory_is_safe(value) and result.diagnostics["completed_depth"] == mpc.config.horizon_steps
    assert np.linalg.norm(value.chassis_states[-1,:2]) >= .01
    np.testing.assert_array_equal(options["h_full"].values, before)
    assert mpc.config.horizon_steps == config().horizon_steps


def test_budget_exhaustion_never_returns_partial_path():
    mpc = planner()
    result = search_checked_guidance_reference(mpc, np.zeros(3), np.array([2.,0.]), .2,
        {**field(g=(1.,0.)),"temporal_safety_derivative":0.}, width=4, candidate_limit=1)
    assert result.trajectory is None and result.diagnostics["truncated"]
    assert result.diagnostics["tested_candidates"] == 1
    assert result.diagnostics["completed_depth"] < mpc.config.horizon_steps


def test_mapped_guidance_rejects_commands_even_when_dtcbf_geometry_is_feasible():
    result = search(**field(g=(-1.,0.), h=.001))
    assert result.diagnostics["exact_feasible_prefixes"] > 0
    assert result.diagnostics["semantic_rejections"] > 0
    # A later turn/escape may be feasible; a directly inward FIRST command is
    # not. Do not make the test require rejection of a valid escape path.
    if result.trajectory is not None:
        assert result.trajectory.commands[0,0] == 0.


def test_forecast_retains_elapsed_time_from_original_initial_epoch():
    result = search(temporal_safety_derivative=-.1)
    assert result.trajectory is not None
    value = result.trajectory
    elapsed = np.arange(config().horizon_steps+1) * config().time_step
    np.testing.assert_allclose(value.safety_values,
        value.samples.safety_values + elapsed*value.samples.scaled_temporal_derivatives)


def test_fallback_search_warms_tracking_solver_but_never_rewrites_nominal(monkeypatch):
    mpc = planner()
    monkeypatch.setattr(mpc, "_guidance_rollout", lambda *a, **k: None)
    nominal = np.tile([.2,0.], (mpc.config.horizon_steps,1))
    before = nominal.copy()
    result = mpc.plan(np.zeros(3), nominal, nominal_goal_grid_xy=(2.,0.), **field(g=(1.,0.)))
    assert result.success and result.guidance_tracking.active
    assert result.guidance_tracking.reason == "exact_safe_bounded_guidance_search_reference"
    assert result.guidance_tracking.reference_search["full_horizon_candidates"] > 0
    np.testing.assert_array_equal(nominal, before)
    checked = mpc._evaluate_trajectory(np.zeros(3), np.array(result.guidance_tracking.reference_commands),
        **field(g=(1.,0.)), temporal_safety_derivative=0.)
    assert mpc._trajectory_is_safe(checked)


def test_failed_search_uses_unchanged_primary_fallback(monkeypatch):
    mpc = planner(reference_search_maximum_candidates=1)
    monkeypatch.setattr(mpc, "_guidance_rollout", lambda *a, **k: None)
    result = mpc.plan(np.zeros(3), (.2,0.), nominal_goal_grid_xy=(2.,0.), **field(g=(1.,0.)))
    primary = SafeSAGEUnicycleMPC(mpc.paper_filter, mpc.config).plan(np.zeros(3), (.2,0.), **field(g=(1.,0.)))
    assert not result.guidance_tracking.active and result.guidance_tracking.reference_search["truncated"]
    np.testing.assert_array_equal(result.command_sequence, primary.command_sequence)


def test_weight_zero_is_primary_even_with_search_enabled():
    mpc = planner()
    mpc.guidance_config = replace(mpc.guidance_config, weight=0.)
    result = mpc.plan(np.zeros(3), (.2,0.), **field(g=(1.,0.)))
    primary = SafeSAGEUnicycleMPC(mpc.paper_filter, mpc.config).plan(np.zeros(3), (.2,0.), **field(g=(1.,0.)))
    assert type(result) is type(primary)
    for f in fields(result):
        if isinstance(getattr(result,f.name), np.ndarray):
            np.testing.assert_array_equal(getattr(result,f.name),getattr(primary,f.name))
        else:
            assert getattr(result,f.name) == getattr(primary,f.name)


@pytest.mark.parametrize("kind", ["wait_then_move", "move_then_stop", "turn_then_move", "stop_at_goal"])
def test_geometrically_safe_reference_must_be_productive_or_have_arrived(kind):
    mpc = planner()
    commands = np.tile([.2,0.], (mpc.config.horizon_steps,1))
    goal = np.array([2.,0.])
    if kind == "wait_then_move":
        commands[:4] = 0.
    elif kind in {"move_then_stop", "stop_at_goal"}:
        commands[4:] = 0.
    else:
        commands[:2] = [0.,.8]
    value = mpc._evaluate_trajectory(np.zeros(3), commands, **field(g=(1.,0.)), temporal_safety_derivative=0.)
    assert mpc._trajectory_is_safe(value)  # This is quality, never a safety waiver.
    if kind == "stop_at_goal":
        goal = value.chassis_states[-1,:2].copy()
    rejections = []
    for n in range(1,len(commands)+1):
        partial = LaplaceGuidanceTrackingMPC(mpc.paper_filter, replace(mpc.config,horizon_steps=n),
            guidance_config=mpc.guidance_config)
        prefix = partial._evaluate_trajectory(np.zeros(3), commands[:n], **field(g=(1.,0.)), temporal_safety_derivative=0.)
        reason = reference_motion_rejection(prefix, partial, goal, terminal=n==len(commands))
        if reason:
            rejections.append(reason)
    assert bool(rejections) == (kind in {"wait_then_move", "move_then_stop"})


@pytest.mark.parametrize("side", [-1.,1.])
def test_symmetric_configuration_barrier_uses_the_guidance_preference(side):
    from robot.safety.local_occupancy_grid import LocalOccupancyGrid, LocalOccupancyGridConfig
    cfg = LocalOccupancyGridConfig(resolution=.1, minimum_x=-1, maximum_x=3, minimum_y=-2, maximum_y=2)
    xs = cfg.minimum_x + (np.arange(cfg.column_count)+.5)*cfg.resolution
    ys = cfg.minimum_y + (np.arange(cfg.row_count)+.5)*cfg.resolution
    x,y = np.meshgrid(xs,ys)
    occupied = (x>=.3-1e-9)&(x<=.6+1e-9)&(abs(y)<=.15+1e-9)
    grid = LocalOccupancyGrid(cfg,occupied,np.full(cfg.shape,-1),occupied.astype(float),
        -occupied.astype(float),int(occupied.sum()),int(occupied.sum()))
    cs, values, _ = make_fields(guidance=(1.,side))
    cs = replace(cs,occupancy_grids=(grid,)*4)
    shape = (4,)+cfg.shape
    boundaries = np.tile(occupied,(4,1,1))
    boundaries[:,0,:] = boundaries[:,-1,:] = True
    boundaries[:,:,0] = boundaries[:,:,-1] = True
    h = np.full(shape,.3)
    h[boundaries] = 0.
    values = replace(values,values=h,spatial_gradients_robot_xy=np.zeros(shape+(2,)),
        yaw_derivatives=np.zeros(shape),dirichlet_boundary_masks=boundaries,
        guidance_divergence=np.zeros(shape),forcing_fields=np.full(shape,-1.),resolution=.1)
    mpc = planner()
    mpc.config = replace(mpc.config,horizon_steps=20,safety_substeps_per_interval=3,
        integration_substeps_per_interval=3)
    result = search_checked_guidance_reference(mpc,np.zeros(3),np.array([2.,0.]),.2,
        dict(configuration_space=cs,h_full=values,guidance_fields=(make_guidance_field(grid,(1.,side)),)*4,
            temporal_safety_derivative=0.),width=8,candidate_limit=4800)
    assert result.trajectory is not None
    assert mpc._trajectory_is_safe(result.trajectory)
    assert side*result.trajectory.chassis_states[-1,1] > .1
    assert side*result.diagnostics["initial_guidance_lateral_preference"] > 0.

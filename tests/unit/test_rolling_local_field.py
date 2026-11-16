"""World-cell invariance, unknown edge rejection and transactional recentering."""

from dataclasses import replace

import numpy as np
import pytest

from robot.evaluation.live_checked_wait import StagedLivePerception
from robot.safety.rolling_local_field import RollingLocalField, rolling_temporal_derivative
from robot.safety.safe_sage_temporal_derivative import SafeSAGETemporalDerivativeEstimator
from test_safe_sage_temporal_derivative import make_grid_config, make_h_full
from test_live_object_perception import packet
from test_object_risk_map import mapped_worker


def difference(old, new, old_grid, new_grid, history=None, dt=.1):
    return rolling_temporal_derivative(SafeSAGETemporalDerivativeEstimator(), old, new,
        previous_grid=old_grid, current_grid=new_grid, time_delta_seconds=dt,
        previous_estimate=history)


@pytest.mark.parametrize("dx,dy", [(1, 0), (-1, 0), (0, 1), (1, -1)])
def test_translated_static_physical_field_has_zero_overlap_derivative(dx, dy):
    grid = make_grid_config()
    moved = replace(grid, minimum_x=grid.minimum_x+dx, maximum_x=grid.maximum_x+dx,
                    minimum_y=grid.minimum_y+dy, maximum_y=grid.maximum_y+dy)
    xy = np.indices((4, 5))
    old = make_h_full(np.tile(xy[1] + 2*xy[0], (4, 1, 1)))
    new = make_h_full(old.values + dx + 2*dy)
    result = difference(old, new, grid, moved)
    assert result.require_valid_samples and not result.valid_mask.all()
    np.testing.assert_allclose(result.raw_values[result.valid_mask], 0)
    row, col = np.argwhere(~result.valid_mask[0])[0]
    point = [moved.minimum_x+col+.5, moved.minimum_y+row+.5]
    with pytest.raises(ValueError, match="overlap"):
        result.sample(query_position_grid_xy=point, query_yaw_radians=0)


def test_second_actual_source_resolves_new_domain_without_fabricated_history():
    grid = make_grid_config()
    moved = replace(grid, minimum_x=grid.minimum_x+1, maximum_x=grid.maximum_x+1)
    old = make_h_full(np.zeros((4, 4, 5)))
    current = make_h_full(np.ones((4, 4, 5)))
    first = difference(old, current, grid, moved)
    later = difference(current, make_h_full(current.values+2), moved, moved, first)
    assert later.valid_mask.all()
    np.testing.assert_allclose(later.values[first.valid_mask], .35*20+.65*3.5)
    np.testing.assert_allclose(later.values[~first.valid_mask], .35*20)
    assert first.time_delta_seconds == later.time_delta_seconds == .1


@pytest.mark.parametrize("dt", [0, -.1, .351, float("nan")])
def test_source_gap_limits_are_unchanged(dt):
    grid = make_grid_config()
    h = make_h_full(np.zeros((4, 4, 5)))
    with pytest.raises(ValueError, match="interval"):
        difference(h, h, grid, grid, dt=dt)


def test_window_follows_long_route_without_rotating_or_expanding():
    grid = make_grid_config()
    rolling = RollingLocalField(grid)
    current = grid
    for point in [(x, 0) for x in np.arange(0, 20, .1)] + [(20, y) for y in np.arange(0, 12, .1)]:
        current = rolling.select(point, current)
        assert current.shape == grid.shape and current.resolution == grid.resolution
        assert current.minimum_x+1 <= point[0] <= current.maximum_x-1
        assert current.minimum_y+1 <= point[1] <= current.maximum_y-1


def test_rejected_async_proposal_does_not_move_accepted_grid_or_source():
    worker = mapped_worker()
    worker.rolling_field = RollingLocalField(worker.grid_config)
    worker(packet())
    original, grid = worker.previous, worker.grid_config
    staged = StagedLivePerception(worker)
    proposed = staged(packet(2, 1.1, 1.05))
    assert proposed.diagnostics["rolling_local_field"]["recentered"]
    assert worker.previous is original and worker.grid_config == grid
    staged.discard()
    proposed = staged(packet(3, 1.2, 1.05))
    staged.accept(proposed)
    assert worker.grid_config.minimum_x == grid.minimum_x+1
    assert worker.previous.sensor_timestamp_seconds == 1.2
    assert worker.previous.temporal_safety_derivative.require_valid_samples


def test_temporal_gap_recovery_carries_unpublished_grid_origin():
    worker = mapped_worker()
    worker.rolling_field = RollingLocalField(worker.grid_config)
    worker(packet())
    staged = StagedLivePerception(worker)
    with pytest.raises(ValueError, match="source gap"):
        staged(packet(2, 2., 1.05))
    assert worker.previous.sensor_timestamp_seconds == 1.
    assert worker.temporal_recovery_baseline[3].minimum_x == worker.grid_config.minimum_x+1
    result = staged(packet(3, 2.1, 1.06))
    assert result.diagnostics["temporal_gap_recovery"]["baseline_timestamp"] == 2.
    staged.accept(result)
    assert worker.previous.sensor_timestamp_seconds == 2.1


def test_batched_and_scalar_control_both_reject_unknown_temporal_cells():
    from test_safe_sage_cbf_filter import make_fields, make_filter
    from robot.safety.safe_sage_temporal_derivative import SafeSAGETemporalDerivativeResult
    space, h, guidance = make_fields()
    valid = np.ones_like(h.values, dtype=bool)
    valid[:, 2, 2] = False
    temporal = SafeSAGETemporalDerivativeResult(
        np.zeros_like(h.values), np.zeros_like(h.values), valid, h.yaw_angles_radians,
        space.occupancy_grids[0].config, .1, (0., 0.), .35, require_valid_samples=True)
    kwargs = dict(configuration_space=space, h_full=h, guidance_fields=guidance,
                  temporal_safety_derivative=temporal)
    with pytest.raises(ValueError, match="overlap"):
        make_filter().evaluate_constraint(query_position_grid_xy=np.array([0., 0.]),
                                         query_yaw_radians=0., **kwargs)
    with pytest.raises(ValueError, match="overlap"):
        make_filter().evaluate_constraints(np.array([[0., 0., 0.]]), **kwargs)
    assert len(make_filter().evaluate_constraints(np.array([[-2., -2., 0.]]), **kwargs)) == 1

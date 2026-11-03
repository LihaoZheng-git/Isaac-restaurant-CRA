"""Float32 sensor input must preserve exact field/continuity geometry."""

import ast
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from robot.evaluation.nearby_track_loss_guard import NearbyTrackLossConfig, NearbyTrackLossGuard
from robot.perception.unknown_obstacle_memory import UnknownObstacleConfig, UnknownObstacleMemory
from test_live_object_perception import packet
from test_object_risk_map import mapped_worker


def native_coverage_geometry(result, source):
    path = Path(__file__).resolve().parents[2]/'scripts/demos/cra_live_object_isaac_demo.py'
    node = next(n for n in ast.walk(ast.parse(path.read_text(encoding='utf-8')))
        if isinstance(n, ast.FunctionDef) and n.name == 'coverage_geometry')
    scope = {'human': object()}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    return scope['coverage_geometry'](result, source)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_world_conversion_preserves_sensor_and_has_no_float32_world_roundtrip(dtype):
    source = replace(packet(), obstacle_xyz=packet().obstacle_xyz.astype(dtype),
        robot_position_world_xy=(10.877, 4.052), robot_yaw=.001234567)
    before = source.obstacle_xyz.copy()
    world = source.obstacle_points_world_xy()
    assert world.dtype == np.float64 and not np.shares_memory(world, source.obstacle_xyz)
    np.testing.assert_array_equal(source.obstacle_xyz, before)
    assert not source.obstacle_xyz.flags.writeable
    assert np.any(world != world.astype(np.float32).astype(float))


@pytest.mark.parametrize('tamper', [False, True])
def test_actual_field_and_native_guard_share_exact_original_geometry(tamper):
    w = mapped_worker()
    w.anchor = np.array([10.877, 4.052])
    w.unknown_obstacle_memory = UnknownObstacleMemory(UnknownObstacleConfig(
        supported_background_prior=True, persistent_background_prior=True,
        unclassified_static_prior=True))
    first = packet().obstacle_xyz.astype(np.float32)
    second = first + np.array([0., 1.3, 0.], dtype=np.float32)
    source = replace(packet(), obstacle_xyz=np.vstack((first, second)),
        robot_position_world_xy=tuple(w.anchor), robot_yaw=.001234567)
    initial = w(source)
    source_geometry = native_coverage_geometry(initial, source)
    later = replace(source, sensor_frame_id=2, camera_frame_id=2,
        sensor_timestamp_seconds=1.2, camera_timestamp_seconds=1.3, obstacle_xyz=second)
    current = w(later)
    region, = current.unknown_obstacle_snapshot.regions
    old = region.original
    original = source_geometry['current_points_world_xy'][
        source_geometry['current_point_track_ids'] == old.track_id]
    np.testing.assert_array_equal(old.points, original)
    guard = NearbyTrackLossGuard(NearbyTrackLossConfig(allow_unknown_obstacle_planning=True))
    guard.observed[old.track_id] = (np.array(old.center), old.source_time)
    original = original.copy()
    if tamper:
        original[-1, 0] += 1e-5
    guard.positive_surfaces[old.track_id] = (original, old.source_time)
    guard.positive_risks[old.track_id] = old.guidance.risk
    result = guard.observe(current.field.tracking_result.tracked_obstacles,
        source_time=1.2, robot_world_xy=old.center,
        current_unknown_obstacle_snapshot=current.unknown_obstacle_snapshot,
        current_occupancy_grid=current.source_occupancy_grid,
        current_grid_anchor_xy=current.field.anchor_position_world_xy,
        current_grid_anchor_yaw=current.field.anchor_yaw)
    assert result['blocked'] is tamper
    if not tamper:
        proof, = result['unknown_obstacle_covered_missing_tracks']
        assert proof['original_source_time'] == 1.
        assert proof['original_point_count'] == len(first)
        assert proof['mode'] == 'unclassified_static_geometry_prior'

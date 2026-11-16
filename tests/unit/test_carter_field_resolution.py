"""Numerical refinement preserves physical geometry; no controller override."""

import argparse
import ast
from dataclasses import replace
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from robot.perception.human_body_envelope import envelope_grid_points
from robot.safety.local_occupancy_grid import LocalOccupancyGrid, LocalOccupancyGridConfig
from robot.safety.safe_sage_configuration_space import (
    RectangularRobotFootprint, SafeSAGEConfigurationSpaceConfig,
    build_safe_sage_configuration_space,
)


ROOT = Path(__file__).resolve().parents[2]
FOOTPRINT = RectangularRobotFootprint(.04, .687351, .2466, .2466, .05)


def body_grid(resolution, center):
    config = LocalOccupancyGridConfig(resolution=resolution, minimum_x=-2., maximum_x=5.)
    points, ids = envelope_grid_points([
        dict(center_world_xy=center, radius_m=.65, envelope_track_id=3_000_000_000),
    ], grid_config=config, anchor_xy=np.zeros(2), anchor_yaw=0.)
    indices = np.floor((points - [config.minimum_x, config.minimum_y]) / resolution).astype(int)
    assert np.all(indices >= 0)
    assert np.all(indices[:, 0] < config.column_count)
    assert np.all(indices[:, 1] < config.row_count)
    occupied = np.zeros(config.shape, bool)
    owners = np.full(config.shape, -1, np.int64)
    occupied[indices[:, 1], indices[:, 0]] = True
    owners[indices[:, 1], indices[:, 0]] = ids
    source = LocalOccupancyGrid(config, occupied, owners, occupied.astype(float),
        -1.7 * occupied, len(points), len(points))
    result = build_safe_sage_configuration_space(source,
        SafeSAGEConfigurationSpaceConfig(footprint=FOOTPRINT, yaw_slice_count=16,
            expansion_backend='vectorized'))
    assert result.footprint == FOOTPRINT and source.discarded_point_count == 0
    return result


def test_recorded_body_raster_rejection_changes_with_resolution_without_shrinking_body():
    # Rounded source-only body component from the first actual rejection at
    # source15.3. It excludes other occupancy and is never a navigation field.
    center = (1.31324893, -.92730207)
    query = (1.24980802, .23708477)
    masks = []
    for resolution in (.1, .05):
        result = body_grid(resolution, center)
        grid = result.occupancy_grids[9]  # Original active yaw slice pi/8.
        row, column = grid.robot_point_to_cell(query)
        masks.append(bool(grid.occupied_mask[row, column]))
    assert masks == [True, False]
    # Both cases retain radius .65, the .05 physical margin and full footprint.
    # The observed pose has positive continuous body-only clearance; interpolation
    # or this component calculation must never override an occupied runtime cell.
    state = np.array([1.15264121, .21344979, .23860751])
    cs, sn = math.cos(state[2]), math.sin(state[2])
    delta = np.array(center) - state[:2]
    local = np.array([cs * delta[0] + sn * delta[1], -sn * delta[0] + cs * delta[1]])
    front, rear = FOOTPRINT.front_extent + .1, FOOTPRINT.rear_extent - .1
    box_center = np.array([(front - rear) / 2, 0.])
    half = np.array([(front + rear) / 2, FOOTPRINT.left_extent]) + FOOTPRINT.safety_margin
    q = np.abs(local - box_center) - half
    margin = np.linalg.norm(np.maximum(q, 0)) + min(float(q.max()), 0) - .65
    assert margin == pytest.approx(.19979172, abs=1e-7)


@pytest.mark.parametrize('resolution', [.1, .05])
@pytest.mark.parametrize('center', [(.8, 0.), (-1.1, 0.), (0., .8), (0., -.8)])
def test_refined_raster_still_blocks_actual_body_envelope_overlap(resolution, center):
    grid = body_grid(resolution, center).occupancy_grids[8]  # Chassis yaw0.
    row, column = grid.robot_point_to_cell((.1, 0.))
    assert grid.occupied_mask[row, column]
    assert grid.cell_track_ids[row, column] == 3_000_000_000


@pytest.mark.parametrize('resolution', [.1, .05])
def test_actual_live_resolution_assignment_preserves_bounds_and_inflation(resolution):
    script = ROOT / 'scripts/demos/cra_live_object_isaac_demo.py'
    tree = ast.parse(script.read_text())
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'grid' for t in node.targets)
        and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
        and node.value.func.id == 'replace']
    assert len(assignments) == 1
    grid = LocalOccupancyGridConfig(minimum_x=-2.)
    result = eval(compile(ast.Expression(assignments[0].value), str(script), 'eval'),
        {'grid': grid, 'replace': replace, 'args': SimpleNamespace(grid_resolution=resolution)})
    assert result.resolution == resolution
    for name in ('minimum_x', 'maximum_x', 'minimum_y', 'maximum_y', 'obstacle_inflation_radius'):
        assert getattr(result, name) == getattr(grid, name)


@pytest.mark.parametrize('resolution', ['0.025', '0.2', 'nan', 'inf'])
def test_invalid_field_resolution_is_rejected_before_isaac(tmp_path, resolution):
    output = tmp_path / 'not_created'
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/demos/cra_live_object_isaac_demo.py'),
        '--output', str(output), '--grid-resolution', resolution],
        capture_output=True, text=True, timeout=10)
    assert result.returncode == 2 and '--grid-resolution' in result.stderr
    assert 'invalid choice' in result.stderr
    assert not output.exists()


@pytest.mark.parametrize('radius', [None, .4])
def test_mass_retention_cli_requires_body_identity_without_requiring_preparation(radius):
    script = ROOT / 'scripts/demos/cra_live_object_isaac_demo.py'
    tree = ast.parse(script.read_text())
    guards = [node for node in tree.body if isinstance(node, ast.If)
        and ast.unparse(node.test).startswith('args.human_mass_body_identity_retention')]
    assert len(guards) == 1
    code = compile(ast.Module(body=guards, type_ignores=[]), str(script), 'exec')
    # No preparation/redesign attribute is supplied. This policy only needs
    # its already approved current body identity; physical gates remain later.
    namespace = {'args': SimpleNamespace(human_mass_body_identity_retention=True,
        human_body_envelope_radius=radius), 'parser': argparse.ArgumentParser()}
    if radius is None:
        with pytest.raises(SystemExit) as error:
            exec(code, namespace)
        assert error.value.code == 2
    else:
        exec(code, namespace)

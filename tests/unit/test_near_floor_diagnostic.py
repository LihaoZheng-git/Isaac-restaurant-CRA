"""Thin-object discovery must not silently become a new geometry permission."""

import numpy as np
import pytest

from robot.evaluation.near_floor_diagnostic import diagnose_near_floor, NearFloorDiagnosticConfig


def floor(slope=0.):
    x, y = np.meshgrid(np.linspace(1, 3, 25), np.linspace(-1, 1, 25))
    return np.column_stack((x.ravel(), y.ravel(), slope*x.ravel()))


def run(points, **kwargs):
    options = dict(floor_reference_m=0., reference_scope="test declared flat-floor prior",
                   source_time=1., observation_time=1.1)
    return diagnose_near_floor(points, **(options | kwargs))


@pytest.mark.parametrize("slope", [0., .02, -.02])
def test_floor_alone_is_not_a_thin_object(slope):
    result = run(floor(slope))
    assert result["floor_fit_available"] and not result["candidates"]
    assert not result["runtime_admissible"]


def test_thin_returns_stay_sparse_and_unresolved_under_original_allowance():
    ground = floor()
    plate = np.column_stack((np.linspace(2, 2.08, 9), np.full(9, .02), np.full(9, .025)))
    points = np.vstack((ground, plate))
    original = points.copy()
    result = run(points)
    assert len(result["candidates"]) == 1
    item = result["candidates"][0]
    assert item["indices"] == list(range(len(ground), len(points)))
    assert item["point_count"] == 9
    assert not item["height_resolved_under_original_allowance"]
    assert item["residual_lower_bound_m"] < 0
    assert item["original_height_allowance_m"] >= .06
    assert not item["runtime_admissible"]
    np.testing.assert_array_equal(points, original)


def test_even_height_resolved_diagnostic_cannot_admit_runtime_geometry():
    raised = np.column_stack((np.linspace(2, 2.08, 9), np.full(9, .02), np.full(9, .09)))
    item = run(np.vstack((floor(), raised)))["candidates"][0]
    assert item["height_resolved_under_original_allowance"]
    assert not item["runtime_admissible"]


@pytest.mark.parametrize("time", [.9, 1.36])
def test_no_source_renewal_or_future_scan(time):
    result = run(floor(), observation_time=time)
    assert result["rejections"] == ["stale or future scan"]
    assert not result["candidates"]
    assert result["source_time"] == 1.


def test_collinear_or_absent_floor_rejects():
    line = np.column_stack((np.linspace(1, 3, 100), np.zeros(100), np.zeros(100)))
    assert not run(line)["floor_fit_available"]
    assert not run(np.empty((0, 3)))["floor_fit_available"]


@pytest.mark.parametrize("change", [dict(maximum_range_m=.1), dict(minimum_candidate_points=2),
    dict(minimum_candidate_points=3.5), dict(cluster_radius_m=float('nan')),
    dict(candidate_residual_min_m=.2)])
def test_invalid_search_config(change):
    with pytest.raises(ValueError):
        NearFloorDiagnosticConfig(**change)


def test_finite_points_and_reference_provenance_required():
    with pytest.raises(ValueError):
        run([[1., 0., np.nan]])
    with pytest.raises(ValueError):
        run(floor(), reference_scope="")

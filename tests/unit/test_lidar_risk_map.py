"""The new presentation must preserve applied risk, clocks and native timing."""

from copy import deepcopy

import numpy as np
import pytest

from robot.evaluation.lidar_risk_map import (
    map_pixels, points_in_field, render_lidar_risk_map, validate_geometry,
)
from robot.evaluation.risk_camera_overlay import risk_color_rgb
from scripts.evaluation.compose_restaurant_lidar_risk import (
    digest, geometry_path, load_verified_geometry, validate_frame_index,
)


def geometry():
    return dict(points_world_xy=np.array([[1., 1.], [1., 1.], [-1., 1.]]),
                point_phi=np.array([.9, .1, .4]), point_basis=np.array([2, 1, 0], np.uint8),
                point_track_ids=np.array([8, 7, 9], np.int64),
                retained_world_xy=np.array([[2., 1.]]), retained_phi=np.array([.8]),
                retained_track_ids=np.array([4], np.int64))


def frame():
    return dict(time=4., source_time=2.5, commit_time=3.0,
                robot_world_xy_yaw=[0., 0., 0.], geometry_file="field.npz", field_id="field-2")


def test_equal_metric_scale_north_up_and_bounds():
    pixel, visible = map_pixels([[0, 0], [1, 0], [0, 1], [0, 5]], [0, 0], (10, 10, 524, 380))
    np.testing.assert_array_equal(pixel[0], [272, 200])
    assert pixel[1, 0] - pixel[0, 0] == pixel[0, 1] - pixel[2, 1] == 47.5
    assert visible.tolist() == [True, True, True, False]


def test_overlapping_returns_show_max_applied_phi_independent_of_order():
    data = geometry()
    before = {key: value.copy() for key, value in data.items()}
    first, audit = render_lidar_risk_map(data, frame(), [], [])
    for key in ("points_world_xy", "point_phi", "point_basis", "point_track_ids"):
        data[key] = data[key][::-1]
    second, _ = render_lidar_risk_map(data, frame(), [], [])
    np.testing.assert_array_equal(first, second)
    x, y = np.round(map_pixels([[1, 1]], [0, 0], (10, 10, 524, 380))[0][0]).astype(int)
    assert first.getpixel((x, y)) == risk_color_rgb(.9)
    assert audit["point_basis_counts"] == dict(ordinary_policy=1, fresh=1, retained=1)
    assert audit["source_age_seconds"] == 1.5
    for key in ("retained_world_xy", "retained_phi", "retained_track_ids"):
        np.testing.assert_array_equal(data[key], before[key])


def test_missing_field_shows_no_fabricated_geometry_or_risk():
    metadata = frame()
    metadata.update(source_time=None, commit_time=None, geometry_file=None)
    _, audit = render_lidar_risk_map(None, metadata, [], [])
    assert audit["status"] == "no_committed_field"
    assert audit["current_points"] == audit["retained_points"] == 0


@pytest.mark.parametrize("fault", ["nan", "range", "shape", "basis", "ids"])
def test_invalid_geometry_rejected_not_replaced_by_low_risk(fault):
    data = geometry()
    if fault == "nan":
        data["point_phi"][0] = np.nan
    elif fault == "range":
        data["retained_phi"][0] = 1.01
    elif fault == "shape":
        data["points_world_xy"] = np.zeros((1, 3))
    elif fault == "basis":
        data["point_basis"][0] = 3
    else:
        data["point_track_ids"] = data["point_track_ids"].astype(float)
    with pytest.raises(ValueError):
        validate_geometry(data)


@pytest.mark.parametrize("changes", [dict(commit_time=4.1), dict(source_time=3.1), dict(time=float("nan"))])
def test_future_or_nonfinite_clock_is_rejected(changes):
    metadata = frame()
    metadata.update(changes)
    with pytest.raises(ValueError):
        render_lidar_risk_map(geometry(), metadata, [], [])


def test_frame_index_cannot_retime_drop_or_use_future_field():
    epochs = [dict(time=4.), dict(time=5.)]
    records = [dict(frame(), source_frame=0), dict(frame(), source_frame=1, time=5.)]
    manifest = dict(schema_version=1, frames=records)
    assert validate_frame_index(manifest, epochs) == records
    for fault in ("frame", "time", "future", "backwards", "drop"):
        bad = deepcopy(manifest)
        if fault == "frame":
            bad["frames"][0]["source_frame"] = 1
        elif fault == "time":
            bad["frames"][0]["time"] = 4.01
        elif fault == "future":
            bad["frames"][0]["commit_time"] = 4.01
        elif fault == "backwards":
            bad["frames"][1]["source_time"] = 1.
            bad["frames"][1]["commit_time"] = 2.
        else:
            bad["frames"].pop()
        with pytest.raises(ValueError):
            validate_frame_index(bad, epochs)


def test_npz_cannot_escape_export_or_request_pickle(tmp_path):
    assert geometry_path(tmp_path, "fields/one.npz").is_relative_to(tmp_path)
    for name in ("../other.npz", "one.pkl"):
        with pytest.raises(ValueError):
            geometry_path(tmp_path, name)


def test_recorded_field_identity_disambiguates_same_time_commit():
    epochs = [dict(time=4.)]
    record = dict(frame(), source_frame=0, field_id="12")
    camera = [dict(time=4., status="latest_assessed_frame", sensor_frame_id=12,
                   sensor_timestamp_seconds=2.5)]
    manifest = dict(schema_version=1, frames=[record])
    validate_frame_index(manifest, epochs, camera)
    record["field_id"] = "13"
    with pytest.raises(ValueError, match="receipt"):
        validate_frame_index(manifest, epochs, camera)
    record["geometry_file"] = None
    with pytest.raises(ValueError, match="receipt"):
        validate_frame_index(manifest, epochs, camera)
    camera[0]["status"] = "awaiting_assessment"
    validate_frame_index(manifest, epochs, camera)


def test_outside_current_and_retained_samples_are_grey_not_applied_risk():
    data = geometry()
    data["field_bounds_world"] = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]])
    data["point_in_field"] = np.zeros(3, dtype=bool)
    result, audit = render_lidar_risk_map(data, frame(), [], [])
    x, y = np.round(map_pixels([[1, 1]], [0, 0], (10, 10, 524, 380))[0][0]).astype(int)
    assert result.getpixel((x, y)) == (162, 174, 184)
    assert audit["point_in_field"] == 0 and audit["point_outside_field"] == 3
    assert audit["retained_in_field"] == 0 and audit["retained_outside_field"] == 1
    assert audit["point_basis_counts"] == dict(ordinary_policy=0, fresh=0, retained=0)


def test_field_bounds_retain_equal_membership_after_world_rotation():
    bounds = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
    points = np.array([[0, 0], [2, 0], [-1, -1], [1, 1]])
    angle = .45
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    transformed = points @ rotation.T + [7., 4.]
    # Exact interior/exterior membership should not depend on world axes.
    assert points_in_field(transformed[:2], bounds @ rotation.T + [7., 4.]).tolist() == [True, False]


def test_geometry_export_hash_verified_before_arrays_are_used(tmp_path):
    path = tmp_path / "one.npz"
    np.savez_compressed(path, **geometry())
    expected = digest(path)
    arrays, _, actual = load_verified_geometry(tmp_path, "one.npz", expected)
    assert actual == expected
    np.testing.assert_array_equal(arrays["point_phi"], geometry()["point_phi"])
    changed = geometry()
    changed["point_phi"][:] = 0
    np.savez_compressed(path, **changed)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_verified_geometry(tmp_path, "one.npz", expected)

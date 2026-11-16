"""View compensation may retain identity, never unsupported/competing geometry."""

from dataclasses import replace

import cv2
import numpy as np
import pytest

from test_image_object_association import proposal
from test_live_object_perception import packet, worker
from robot.perception.image_object_association import (
    ImageObjectConfig,
    StationaryImageObjects,
)
from robot.perception.moving_view_association import (
    MovingViewAssociationConfig,
    compare_moving_view,
    image_object_view,
)
from robot.perception.camera_lidar_fusion import project_fisheye_polynomial_points


def context(x=0.0, yaw=0.0):
    camera = np.eye(4)
    camera[:3, :3] = [[0, -1, 0], [0, 0, 1], [-1, 0, 0]]
    p = replace(
        packet(),
        camera_from_scan_robot=camera,
        robot_position_world_xy=(x, 0.0),
        robot_yaw=yaw,
    )
    return image_object_view(p, worker().calibration, MovingViewAssociationConfig())


def surface():
    return np.array(
        [
            [2.0, y, z]
            for y in np.linspace(-0.3, 0.3, 9)
            for z in np.linspace(0.1, 0.5, 7)
        ]
    )


def mask_for(points, view):
    projection = project_fisheye_polynomial_points(
        points, camera_from_source=view.camera_from_world, calibration=view.calibration
    )
    mask = np.zeros((100, 100), np.uint8)
    cv2.fillConvexPoly(
        mask, cv2.convexHull(projection.pixel_indices_xy.astype(np.int32)), 1
    )
    return cv2.dilate(mask, np.ones((3, 3), np.uint8)).astype(bool)


def candidate(mask, n):
    return {"proposal": proposal(mask), "selected": np.ones(n, bool), "diagnostic": {}}


def enrolled():
    points = surface()
    view = context()
    state = StationaryImageObjects(ImageObjectConfig(hold_overlapping_hypotheses=True))
    for t in (1.0, 1.1, 1.2):
        accepted = state.update(
            [candidate(mask_for(points, view), len(points))], points, t, view=view
        )
    assert len(accepted) == 1
    return state, accepted[0]["object_id"], points


def test_ego_approach_and_partial_lidar_keep_original_reference_and_clock():
    state, key, points = enrolled()
    original = state.records[key]["points"].copy()
    partial = points[points[:, 1] >= 0]
    view = context(0.3)
    current = candidate(mask_for(points, view), len(partial))
    accepted = state.update([current], partial, 1.3, view=view)
    assert [c["object_id"] for c in accepted] == [key]
    audit = current["diagnostic"]["reference_comparisons"][0]
    assert audit["original_maximum_distance_m"] > 0.06
    assert audit["warped_mask_iou"] >= 0.75
    assert audit["original_unobserved_samples"] > 0
    assert state.records[key]["first"] == 1.0
    np.testing.assert_array_equal(state.records[key]["points"], original)
    assert accepted[0]["selected"].sum() == len(partial)


@pytest.mark.parametrize("yaw", [-0.05, 0.05])
def test_small_camera_turn_keeps_static_identity_with_current_mask(yaw):
    state, key, points = enrolled()
    view = context(0.15, yaw)
    accepted = state.update(
        [candidate(mask_for(points, view), len(points))], points, 1.3, view=view
    )
    assert [c["object_id"] for c in accepted] == [key]


def test_occluded_rgb_outline_is_not_claimed_as_full_view():
    state, key, points = enrolled()
    view = context(0.2)
    mask = mask_for(points, view)
    yy, xx = np.where(mask)
    mask[:, int(np.median(xx)) :] = False
    assert not state.update([candidate(mask, len(points))], points, 1.3, view=view)
    assert state.records[key]["last"] == 1.2


@pytest.mark.parametrize(
    "case",
    ["outside_surface", "tiny_patch", "wrong_mask", "excess_view", "rank_deficient"],
)
def test_partial_matching_rejects_unsupported_or_ambiguous_inputs(case):
    state, key, points = enrolled()
    view = context(0.2)
    current = points.copy()
    mask = mask_for(points, view)
    if case == "outside_surface":
        current[0, 0] += 0.08
    if case == "tiny_patch":
        current = np.repeat(points[:2], 8, axis=0)
    if case == "wrong_mask":
        mask = np.roll(mask, 18, axis=1)
    if case == "excess_view":
        view = context(0.6)
        mask = mask_for(points, view)
    if case == "rank_deficient":
        state.records[key]["points"] = points[:7].copy()
        current = points[:7].copy()
    accepted = state.update([candidate(mask, len(current))], current, 1.3, view=view)
    assert not accepted
    assert state.records[key]["last"] == 1.2
    assert len(state.records) == 1  # overlap may reject, never create a duplicate


def test_two_compatible_references_are_ambiguous_not_selected_by_score():
    state, key, points = enrolled()
    import copy

    state.records[key + 1] = copy.deepcopy(state.records[key])
    view = context(0.2)
    row = candidate(mask_for(points, view), len(points))
    assert not state.update([row], points, 1.3, view=view)
    assert (
        row["diagnostic"]["temporal_rejection"] == "partial or ambiguous current object"
    )
    assert all(v["last"] == 1.2 for v in state.records.values())


def test_adjacent_separate_object_does_not_inherit_original_id_or_evidence():
    state, key, points = enrolled()
    other = points + [0.8, 0.3, 0]
    view = context(0.1)
    for t in (1.3, 1.4, 1.5):
        accepted = state.update(
            [candidate(mask_for(other, view), len(other))], other, t, view=view
        )
    assert len(accepted) == 1 and accepted[0]["object_id"] != key
    assert state.records[key]["last"] == 1.2


def test_gap_and_original_expiry_still_require_requalification():
    state, key, points = enrolled()
    view = context(0.2)
    for t in (1.7, 1.8):
        assert not state.update(
            [candidate(mask_for(points, view), len(points))], points, t, view=view
        )
    assert (
        state.update(
            [candidate(mask_for(points, view), len(points))], points, 1.9, view=view
        )[0]["object_id"]
        == key
    )
    assert not state.update(
        [candidate(mask_for(points, view), len(points))], points, 11.01, view=view
    )
    assert key not in state.records


def test_long_chain_cannot_walk_reference_to_new_surface():
    state, key, points = enrolled()
    view = context(0.2)
    for i in range(1, 6):
        current = points + [i * 0.02, 0, 0]
        accepted = state.update(
            [candidate(mask_for(current, view), len(current))],
            current,
            1.2 + i * 0.1,
            view=view,
        )
        if i > 3:
            assert all(c["object_id"] != key for c in accepted)
    np.testing.assert_array_equal(state.records[key]["points"], points)


def test_world_camera_conversion_preserves_canonical_ground_and_ego_transform():
    p = replace(
        packet(),
        robot_position_world_xy=(3.0, 4.0),
        robot_yaw=0.3,
        floor_height_robot_m=-0.2,
    )
    view = image_object_view(p, worker().calibration, MovingViewAssociationConfig())
    c, s = np.cos(0.3), np.sin(0.3)
    scan = np.array([[1.0, 0.1, 0.2], [2.0, 0.3, 0.4]])
    world = scan.copy()
    world[:, :2] = scan[:, :2] @ np.array([[c, s], [-s, c]]) + [3, 4]
    world[:, 2] += 0.2
    projected = world @ view.camera_from_world[:3, :3].T + view.camera_from_world[:3, 3]
    np.testing.assert_allclose(projected, scan, atol=1e-12)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"minimum_reference_coverage": 0},
        {"maximum_warp_residual_pixels": float("nan")},
        {"minimum_correspondences": 7},
        {"minimum_axis_span_fraction": 1.1},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        MovingViewAssociationConfig(**kwargs)


def test_no_calibrated_origin_cannot_be_treated_as_motion_proof():
    points = surface()
    view = context()
    old = {"points": points, "mask": mask_for(points, view), "first": 1.0}
    ok, _, audit = compare_moving_view(
        old, old["mask"], points, view, ImageObjectConfig()
    )
    assert not ok and "no original calibrated camera reference" in audit["rejections"]


def test_runtime_partial_group_preserves_sources_and_only_owns_current_points(tmp_path):
    from test_image_object_runtime import setup, observed
    from robot.perception.physical_attributes import EstimateStatus

    runtime = setup()
    runtime.image_object_property_persistence = True
    runtime.moving_image_object_config = MovingViewAssociationConfig()
    runtime.image_object_archive_directory = tmp_path
    points = surface()
    current_view = [context()]
    runtime.segmenter.image_proposals = lambda *a: (
        proposal(mask_for(points, current_view[0])),
    )
    for i, t in enumerate((1.0, 1.1, 1.2), 1):
        p = replace(observed(i, t), obstacle_xyz=points, scene_xyz=points)
        result = runtime(p)
    first = next(
        o for o in result.objects if o.diagnostics.get("image_object_association")
    )
    runtime.persistent_properties.freeze_properties()
    partial = points[points[:, 1] >= 0]
    current_view[0] = context(0.2)
    p = replace(
        observed(4, 1.3),
        obstacle_xyz=partial - [0.2, 0, 0],
        scene_xyz=partial - [0.2, 0, 0],
        robot_position_world_xy=(0.2, 0.0),
        measured_robot_velocity_world_xy=(0.2, 0.0),
    )
    result = runtime(p)
    second = next(
        o for o in result.objects if o.diagnostics.get("image_object_association")
    )
    assert second.attributes.track_id == first.attributes.track_id
    assert second.attributes.mass.mass_kg.status is EstimateStatus.PRIOR
    assert (
        second.attributes.mass.mass_kg.timestamp_seconds
        == first.attributes.mass.mass_kg.timestamp_seconds
    )
    assert second.diagnostics["policy_return_count"] == len(partial)
    assert second.diagnostics["image_object_association"][
        "reference_point_count"
    ] == len(points)
    assert result.field.all_obstacle_point_count >= len(partial)
    assert result.diagnostics["fixed_lambda"] == 1.25
    assert (tmp_path / "000004.npz").is_file()
    before = runtime.image_object_state
    runtime.segmenter.image_proposals = lambda *a: ()
    runtime(
        replace(
            p,
            sensor_frame_id=5,
            camera_frame_id=5,
            sensor_timestamp_seconds=1.4,
            camera_timestamp_seconds=1.5,
        )
    )
    assert not runtime.image_object_state.property_memory.properties
    assert before.property_memory.properties

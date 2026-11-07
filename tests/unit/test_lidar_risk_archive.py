"""Presentation replay must preserve ownership, source order and final risk."""

from copy import deepcopy

import numpy as np
import pytest

from robot.evaluation.lidar_risk_archive import (
    admitted_video_fields, applied_policy_table, assign_current_points, static_selection,
)


def record(objects, ids=(1, 1_000_000_001, 2_000_000_000, 8_000_000_000)):
    return dict(objects=objects, phi=.99, diagnostics=dict(applied_field_policies=[
        dict(track_or_group_id=i, applied_field_risk=.1 + n / 10,
             cra_phi=.9, field_used_for_control=True) for n, i in enumerate(ids)]))


def obj(key=1, accepted=False, count=2):
    return dict(track_id=key, diagnostics=dict(accepted_instance=accepted,
        policy_return_count=count, selected_return_count=count if accepted else 0,
        factor_scores={"kinetic": {"mode": "configured_prior"}}, property_evidence={}))


def test_auxiliary_policy_from_old_scan_does_not_relabel_current_raw_points():
    row = record([obj()])
    ids, phi, basis = assign_current_points(row, np.zeros((3, 3)), np.array([1, 1, -1]), None)
    assert ids.tolist() == [1, 1, 2_000_000_000]
    assert phi.tolist() == pytest.approx([.1, .1, .3])
    assert basis.tolist() == [0, 0, 0]


def test_original_owned_track_without_current_visual_ownership_gets_auxiliary():
    item = obj(count=0)
    item['diagnostics']['factor_scores']['kinetic']['mode'] = 'evidence_supported'
    ids, phi, _ = assign_current_points(record([item]), np.zeros((2, 3)),
        np.array([1, 1]), None)
    assert ids.tolist() == [1_000_000_001] * 2
    assert phi.tolist() == pytest.approx([.2, .2])


def test_late_human_mass_update_cannot_rewrite_earlier_current_ownership():
    item = obj(count=2)
    item['diagnostics']['factor_scores']['kinetic']['mode'] = 'evidence_supported'
    item['diagnostics']['human_mass_identity_retention'] = {'mode': 'persistence_prior'}
    item['diagnostics']['property_evidence']['mass'] = {'mode': 'persistence_prior'}
    ids, _, basis = assign_current_points(record([item]), np.zeros((2, 3)),
        np.array([1, 1]), None)
    assert ids.tolist() == [1, 1]
    assert basis.tolist() == [2, 2]


def test_fresh_exact_mask_and_unmatched_support_get_different_final_policies():
    selected = np.array([True, False, True])
    ids, phi, basis = assign_current_points(record([obj(accepted=True)]),
        np.zeros((3, 3)), np.array([1, 1, 1]), lambda _: selected)
    assert ids.tolist() == [1, 1_000_000_001, 1]
    assert phi.tolist() == pytest.approx([.1, .2, .1])
    assert basis.tolist() == [1, 0, 1]


def test_expanded_mask_rejected_even_with_correct_count():
    with pytest.raises(ValueError, match='expanded'):
        assign_current_points(record([obj(accepted=True)]), np.zeros((3, 3)),
            np.array([1, 1, 2]), lambda _: np.array([True, False, True]))


def test_static_proof_selects_only_original_region_and_fragment_members():
    diagnostic = dict(selected_return_count=2, static_prop_association=dict(
        rejections=[], position_allowance_m=.1, observed_bounds_minimum=[0, 0, 0],
        observed_bounds_maximum=[1, 1, 1], fragments=[dict(raw_track_id=1, hits=2)]))
    world = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0], [2, 2, 2]])
    assert static_selection(world, np.array([1, 1, 2, 1]), diagnostic).tolist() == [True, True, False, False]
    corrupted = deepcopy(diagnostic)
    corrupted['static_prop_association']['fragments'][0]['hits'] = 3
    with pytest.raises(ValueError, match='membership'):
        static_selection(world, np.array([1, 1, 2, 1]), corrupted)


def test_final_count_mismatch_and_missing_policy_fail_explicitly():
    with pytest.raises(ValueError, match='partial ownership'):
        assign_current_points(record([obj(count=1)]), np.zeros((2, 3)), np.array([1, 1]), None)
    with pytest.raises(ValueError, match='no original applied policy'):
        assign_current_points(record([], ids=()), np.zeros((2, 3)), np.array([1, 1]), None)


def test_final_applied_risk_not_raw_cra_is_used_and_duplicate_rejected():
    row = record([])
    assert applied_policy_table(row)[1]['applied_field_risk'] == .1
    row['diagnostics']['applied_field_policies'].append(row['diagnostics']['applied_field_policies'][0])
    with pytest.raises(ValueError, match='duplicate'):
        applied_policy_table(row)


def test_video_exact_tie_uses_recorded_admitted_id_not_newer_commit():
    fields = [dict(field_id=1, commit_time=2., source_time=1.),
              dict(field_id=2, commit_time=3., source_time=2.)]
    epochs = [dict(time=3.)]
    risk = [dict(time=3., status='latest_assessed_frame', sensor_frame_id=1, sensor_timestamp_seconds=1.)]
    assert admitted_video_fields(fields, epochs, risk) == [fields[0]]
    risk[0]['sensor_frame_id'] = 99
    with pytest.raises(ValueError, match='no committed field'):
        admitted_video_fields(fields, epochs, risk)


def test_future_field_and_mismatched_source_cannot_color_video():
    fields = [dict(field_id=1, commit_time=4., source_time=1.)]
    epochs = [dict(time=3.)]
    risk = [dict(time=3., status='latest_assessed_frame', sensor_frame_id=1, sensor_timestamp_seconds=1.)]
    with pytest.raises(ValueError, match='future'):
        admitted_video_fields(fields, epochs, risk)
    fields[0]['commit_time'] = 2.
    risk[0]['sensor_timestamp_seconds'] = 1.1
    with pytest.raises(ValueError, match='mismatched'):
        admitted_video_fields(fields, epochs, risk)


def test_duplicate_field_ids_are_ambiguous_even_with_same_times():
    field = dict(field_id=1, commit_time=2., source_time=1.)
    with pytest.raises(ValueError, match='duplicate committed'):
        admitted_video_fields([field, field], [], [])

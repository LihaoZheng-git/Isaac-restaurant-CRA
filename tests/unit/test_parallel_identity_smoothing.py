"""Original identity eligibility is separate from current motion and safe control."""

from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import pytest

from test_parallel_guidance_selection import selection_records
from robot.evaluation.parallel_smoothing import outward_identity_command_change
from robot.evaluation.parallel_walking_validation import validate_parallel_selection
from robot.perception.human_body_envelope import HUMAN_ENVELOPE_SOURCE, HumanIdentitySelectionPrior


def records():
    rows = selection_records()
    prior = HumanIdentitySelectionPrior(7, .1, 42, .35, .3, 100).diagnostic()
    for row in rows:
        audit = row['motion_aware_passing']
        audit['configuration']['maximum_identity_prior_selection_age_seconds'] = .5
        audit.update(original_selection_visual_person_verified=False,
            original_selection_identity_prior=deepcopy(prior) if audit['preferences'] else None,
            current_identity_prior_selection_evidence={7: deepcopy(prior)} if row['time'] == .3 else {},
            current_visual_person_track_ids=[])
        row['human_body_envelope_priors'] = [dict(track_id=7, source=HUMAN_ENVELOPE_SOURCE,
            status='prior', lidar_timestamp_seconds=row['time'], current_support_count=100)]
    return rows


def audit(rows):
    return validate_parallel_selection(rows, maximum_identity_prior_selection_age_seconds=.5)


def test_explicit_prior_audit_preserves_primary_fresh_only_policy_and_accepts_live_layout():
    rows = records()
    result = audit(rows)
    assert result['passed'] and result['identity_prior_selected_rows'] == 3
    assert result['identity_prior_selections'] == 1
    assert not validate_parallel_selection(rows)['passed']
    live = [dict(sensor_timestamp=r['time'], objects=r['objects'], diagnostics={
        k: r[k] for k in ('motion_aware_passing', 'human_body_envelope_priors', 'applied_field_policies')})
        for r in rows]
    assert audit(live)['passed']


@pytest.mark.parametrize('failure', ['renewed_image', 'renewed_selection', 'future', 'expired',
    'fresh_claim', 'wrong_track', 'missing_envelope', 'stale_envelope', 'no_support', 'not_prior',
    'missing_start_record', 'missing_start_proof', 'different_start_proof', 'different_configuration',
    'ambiguous_current_candidates', 'stale_motion', 'wrong_side', 'lambda', 'expired_policy'])
def test_parallel_prior_audit_rejects_invalid_identity_motion_field_and_epoch_provenance(failure):
    rows = records()
    row = rows[-1]
    a = row['motion_aware_passing']
    p = a['original_selection_identity_prior']
    e = row['human_body_envelope_priors'][0]
    if failure == 'renewed_image':
        p['original_person_camera_timestamp'] = .2
    elif failure == 'renewed_selection':
        a['encounter_selected_timestamp'] = .4
    elif failure in {'future', 'expired'}:
        p['original_person_camera_timestamp'] = .4 if failure == 'future' else -.3
    elif failure == 'fresh_claim':
        a['original_selection_visual_person_verified'] = True
    elif failure == 'wrong_track':
        p['track_id'] = 8
    elif failure == 'missing_envelope':
        row['human_body_envelope_priors'] = []
    elif failure == 'stale_envelope':
        e['lidar_timestamp_seconds'] -= .1
    elif failure == 'no_support':
        e['current_support_count'] = 0
    elif failure == 'not_prior':
        e['status'] = 'inferred'
    elif failure == 'missing_start_record':
        rows = rows[3:]
    elif failure == 'missing_start_proof':
        rows[2]['motion_aware_passing']['current_identity_prior_selection_evidence'] = {}
    elif failure == 'different_start_proof':
        rows[2]['motion_aware_passing']['current_identity_prior_selection_evidence'][7]['original_person_camera_frame_id'] = 43
    elif failure == 'different_configuration':
        a['configuration']['maximum_identity_prior_selection_age_seconds'] = .4
    elif failure == 'ambiguous_current_candidates':
        other = dict(a['qualified_candidates'][0], track_id=8)
        a['qualified_candidates'].append(other)
    elif failure == 'stale_motion':
        row['objects'][0]['attributes']['motion']['velocity_y_mps']['timestamp_seconds'] = 0.
    elif failure == 'wrong_side':
        a['preferences'][7] = 'right'
    elif failure == 'lambda':
        row['applied_field_policies'][0]['lambda'] = 0.
    else:
        row['time'] = 8.4
    assert not audit(rows)['passed']


def test_held_prior_policy_can_outlive_selection_cap_but_never_original_policy_clock():
    rows = records()
    row = deepcopy(rows[-1])
    row['time'] = 1.
    row['human_body_envelope_priors'][0]['lidar_timestamp_seconds'] = 1.
    for v in row['objects'][0]['attributes']['motion'].values():
        v['timestamp_seconds'] = 1.
    rows.append(row)
    assert audit(rows)['passed']
    assert row['motion_aware_passing']['original_selection_identity_prior']['original_person_camera_timestamp'] == .1


@pytest.mark.parametrize('cap', [-1., .500001, float('nan'), True])
def test_validation_cannot_expand_the_previously_approved_cap(cap):
    with pytest.raises(ValueError):
        validate_parallel_selection(records(), maximum_identity_prior_selection_age_seconds=cap)


def commands():
    old = ['python', 'demo', '--output', 'old', '--human-passing-identity-prior-max-age', '0.0',
        '--guidance-activation-h', '2.0', '--human-envelope-centroid-uncertainty', '.25', '--lambda', '1.25']
    new = old.copy()
    new[3] = 'new'
    new[5] = '.5'
    new += ['--guidance-outward-reference-fraction', '.35']
    return old, new


@pytest.mark.parametrize('change', [None, 'activation', 'body', 'lambda', 'zero_prior', 'no_outward', 'extra'])
def test_matched_variant_only_allows_the_two_explicit_reference_identity_options(change):
    old, new = commands()
    flag = {'activation': '--guidance-activation-h', 'body': '--human-envelope-centroid-uncertainty',
        'lambda': '--lambda', 'zero_prior': '--human-passing-identity-prior-max-age',
        'no_outward': '--guidance-outward-reference-fraction'}.get(change)
    if flag:
        new[new.index(flag)+1] = '0.0'
    elif change == 'extra':
        new += ['--skip-safety']
    assert outward_identity_command_change(old, new)['matched'] == (change is None)


@pytest.mark.parametrize('value', ['.6', 'nan', '-.1'])
def test_invalid_comparison_cannot_authorize_a_larger_prior(value):
    old, new = commands()
    new[5] = value
    with pytest.raises(ValueError):
        outward_identity_command_change(old, new)


@pytest.mark.parametrize('options', [[], ['--parallel-guidance-selection'],
    ['--parallel-guidance-selection', '--identity-selection-body-envelope', '--human-passing-identity-prior-max-age', '.6']])
def test_zero_wheel_cli_rejects_unlabeled_or_expanded_identity_scope_before_isaac(tmp_path, options):
    script = Path(__file__).resolve().parents[2] / 'scripts/demos/restaurant_human_locomotion_isaac.py'
    args = ['--scene', 'unused.usda', '--layout-survey', 'unused.json', '--output', str(tmp_path/'unused'),
        '--human-passing-identity-prior-max-age', '.5', *options]
    result = subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
    assert not (tmp_path/'unused').exists()

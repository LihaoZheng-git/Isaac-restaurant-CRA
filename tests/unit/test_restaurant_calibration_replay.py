"""Archived replay must not color invented ownership or erase old risk floors."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from scripts.evaluation.export_restaurant_calibration_replay import (
    digest,
    export,
    replay_policies,
    validate_field,
)


def test_complete_export_covers_stationary_and_live_manifest(tmp_path):
    native = tmp_path / "native"
    native.mkdir()
    rows = [dict(sensor_frame_id=i, sensor_timestamp=float(i),
                 commit_simulation_time=i+.1, objects=[],
                 diagnostics=dict(human_body_envelope_priors=[],
                                  applied_field_policies=[], carried_object_groups=None))
            for i in (1, 2)]
    stationary = native / "stationary-fields.json"
    live = native / "live-fields.jsonl"
    stationary.write_text(json.dumps(rows[:1]))
    live.write_text(json.dumps(rows[1])+"\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(
        source_directory=str(native),
        source_hashes={str(p): digest(p) for p in (stationary, live)},
        fields=[dict(field_id=r["sensor_frame_id"], source_time=r["sensor_timestamp"],
                     commit_time=r["commit_simulation_time"]) for r in rows])))
    output = tmp_path / "replay"
    summary = export(manifest, output)
    assert summary["counts"]["fields"] == 2
    assert summary["fields_sha256"] == digest(output / summary["fields_file"])
    assert summary["source_hashes_verified_after_export"]
    assert [json.loads(line)["sensor_frame_id"] for line in
            (output / summary["fields_file"]).read_text().splitlines()] == [1, 2]
    live.write_text(live.read_text()+"\n")
    with pytest.raises(ValueError, match="source hash changed"):
        export(manifest, output)


def policy(key, risk, applied=None):
    return dict(
        track_or_group_id=key,
        cra_phi=risk,
        applied_field_risk=risk if applied is None else applied,
        boundary_flux=-(0.5 + 1.2 * risk),
        **{"lambda": 1.25},
    )


def record(policies, envelopes=()):
    return dict(
        diagnostics=dict(
            applied_field_policies=policies, human_body_envelope_priors=envelopes
        )
    )


def test_current_scores_change_without_relabeling_old_geometry():
    result = replay_policies(record([policy(2, 0.62)]), {2: SimpleNamespace(phi=0.35)})[
        0
    ]
    assert result["candidate_current_phi"] == 0.35
    assert result["candidate_prior_geometry_phi"] == 0.62
    assert result["lambda_value"] == 1.25


def test_unassessed_and_external_floor_stay_unchanged():
    rows = replay_policies(
        record([policy(1, 0.68359375), policy(2, 0.4, 0.75)]),
        {2: SimpleNamespace(phi=0.3)},
    )
    assert rows[0]["candidate_current_phi"] == 0.68359375
    assert rows[1]["candidate_current_phi"] == 0.75
    assert rows[1]["preserved_external_floor"] == 0.75


def test_body_envelope_uses_exact_parent_with_original_floor():
    envelope = dict(track_id=2, envelope_track_id=1002, applied_to_occupancy=True)
    r = record([policy(2, 0.68), policy(1002, 0.7)], [envelope])
    rows = replay_policies(r, {2: SimpleNamespace(phi=0.84)})
    assert rows[1]["candidate_current_phi"] == 0.84
    assert rows[1]["candidate_prior_geometry_phi"] == 0.7
    assert rows[1]["envelope_parent_track_id"] == 2
    rows = replay_policies(r, {2: SimpleNamespace(phi=0.6)})
    assert rows[1]["candidate_current_phi"] == 0.7


def test_unused_envelope_does_not_fabricate_a_policy():
    r = record(
        [policy(1, 0.68)],
        [dict(track_id=1, envelope_track_id=1001, applied_to_occupancy=False)],
    )
    assert len(replay_policies(r, {1: SimpleNamespace(phi=0.4)})) == 1


@pytest.mark.parametrize(
    "change", ("missing_parent", "missing_envelope", "duplicate_policy")
)
def test_ambiguous_envelope_or_policy_mapping_is_rejected(change):
    r = record(
        [policy(1, 0.68), policy(1001, 0.7)],
        [dict(track_id=1, envelope_track_id=1001, applied_to_occupancy=True)],
    )
    individual = {1: SimpleNamespace(phi=0.84)}
    if change == "missing_parent":
        individual.clear()
    elif change == "missing_envelope":
        r["diagnostics"]["applied_field_policies"].pop()
    else:
        r["diagnostics"]["applied_field_policies"].append(policy(1, 0.68))
    with pytest.raises(ValueError):
        replay_policies(r, individual)


@pytest.mark.parametrize(
    "key", ("sensor_frame_id", "sensor_timestamp", "commit_simulation_time")
)
def test_field_replay_requires_exact_original_identity_and_clocks(key):
    original = dict(
        sensor_frame_id=12, sensor_timestamp=2.0, commit_simulation_time=2.1
    )
    expected = dict(field_id=12, source_time=2.0, commit_time=2.1)
    validate_field(original, expected)
    bad = deepcopy(original)
    bad[key] += 0.001
    with pytest.raises(ValueError, match="clocks"):
        validate_field(bad, expected)

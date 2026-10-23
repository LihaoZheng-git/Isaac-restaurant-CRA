"""Per-current-human tangential-bias ablation, separate from physical CRA."""

import copy
from dataclasses import replace


def apply_encounter_guidance(guidance, audit, classification, passing, *, source_time):
    """Suppress only a currently selected, stably transverse human and proven parts.

    Body-prior cells and positively proven current visual parts share the policy.
    Unmatched support/auxiliary occupancy and class fallbacks retain lambda1.25.
    No risk, flux, side, occupancy, identity or evidence value is changed.
    """
    if (
        classification["source_timestamp_seconds"] != source_time
        or passing["source_timestamp"] != source_time
    ):
        raise ValueError("encounter policies require the current source pair")
    states = classification["tracks"]
    selected = {int(k) for k in passing["preferences"]}
    selected_time = passing["encounter_selected_timestamp"]
    if selected and (
        selected_time is None or not 0 <= source_time - selected_time <= 8.0 + 1e-9
    ):
        raise ValueError(
            "current passing selection must retain its original bounded clock"
        )
    suppress = {
        key
        for key in selected
        if key in states and states[key]["classification"] == "crossing"
    }
    for key, row in states.items():
        if row["source_timestamp_seconds"] != source_time or row["track_id"] != key:
            raise ValueError("classification must match original current track epoch")
    if len({r["track_or_group_id"] for r in audit}) != len(audit) or set(guidance) != {
        r["track_or_group_id"] for r in audit
    }:
        raise ValueError("unique matching policy audit required")
    out, rows = dict(guidance), copy.deepcopy(audit)
    for row in rows:
        key = row["track_or_group_id"]
        value = guidance[key]
        if value.tangential_strength != 1.25 or row.get("lambda", 1.25) != 1.25:
            raise ValueError(
                "encounter variant starts from the fixed-lambda physical policy"
            )
        owner = key if key in states else row.get("geometry_prior_parent_track_id")
        part = row.get("human_part_control_policy_prior")
        if part is not None:
            proof = part["association"]
            if (
                proof["source_timestamp_seconds"] != source_time
                or proof["child_track_id"] != key
            ):
                raise ValueError(
                    "only current already-validated positive part proofs may share lambda"
                )
            owner = part["parent_track_id"]
        enabled = owner in suppress
        if enabled:
            out[key] = replace(value, tangential_strength=0.0)
        row.update(
            physical_CRA_lambda=1.25,
            **{"lambda": out[key].tangential_strength},
            encounter_guidance_ablation=dict(
                parent_track_id=owner,
                classification=states[owner]["classification"]
                if owner in states
                else "unassigned",
                source_timestamp_seconds=source_time,
                lambda_suppressed=enabled,
                original_selection_timestamp_seconds=selected_time if enabled else None,
                basis="current_selected_crossing_only"
                if enabled
                else "normal_guidance_retained",
                scope="control policy only; physical CRA and normal flux unchanged",
            ),
        )
    return out, rows, tuple(sorted(suppress))

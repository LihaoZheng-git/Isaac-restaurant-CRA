import copy

import pytest

from robot.evaluation.encounter_guidance import apply_encounter_guidance
from robot.safety.class_safety_policy import ClassSafetyParameters
from robot.safety.safe_sage_guidance_policy import SafeSAGEGuidanceParameters


def inputs(state="crossing"):
    g = {
        k: SafeSAGEGuidanceParameters(ClassSafetyParameters(0.6, 1.22), 1.25, "left")
        for k in (7, 8, 9, 70, 1000000007)
    }
    rows = [
        dict(
            track_or_group_id=k,
            cra_phi=0.6,
            applied_field_risk=0.6,
            boundary_flux=-1.22,
            **{"lambda": 1.25},
        )
        for k in g
    ]
    rows[1]["human_part_control_policy_prior"] = dict(
        parent_track_id=7,
        association=dict(source_timestamp_seconds=2.0, child_track_id=8),
    )
    rows[3]["geometry_prior_parent_track_id"] = 7
    c = dict(
        source_timestamp_seconds=2.0,
        tracks={
            7: dict(track_id=7, classification=state, source_timestamp_seconds=2.0)
        },
    )
    p = dict(
        source_timestamp=2.0, preferences={7: "left"}, encounter_selected_timestamp=1.0
    )
    return g, rows, c, p


def test_crossing_suppression_only_touches_current_parent_body_and_proven_parts():
    g, a, c, p = inputs()
    saved = copy.deepcopy((g, a, c, p))
    changed, rows, ids = apply_encounter_guidance(g, a, c, p, source_time=2.0)
    assert ids == (7,)
    assert {k for k, v in changed.items() if v.tangential_strength == 0} == {7, 8, 70}
    assert changed[9] is g[9] and changed[1000000007] is g[1000000007]
    for k in g:
        assert (
            changed[k].risk == g[k].risk
            and changed[k].boundary_flux == g[k].boundary_flux
        )
        assert changed[k].passing_side == g[k].passing_side
    assert (g, a, c, p) == saved
    assert all(r["physical_CRA_lambda"] == 1.25 for r in rows)


@pytest.mark.parametrize("state", ["parallel", "stationary", "uncertain"])
def test_non_crossing_keeps_normal_guidance_even_with_latched_old_passing_side(state):
    g, a, c, p = inputs(state)
    changed, _, ids = apply_encounter_guidance(g, a, c, p, source_time=2.0)
    assert ids == () and all(v.tangential_strength == 1.25 for v in changed.values())


def test_no_current_selection_cannot_suppress_or_create_a_yield():
    g, a, c, p = inputs()
    p.update(preferences={}, encounter_selected_timestamp=None)
    changed, _, ids = apply_encounter_guidance(g, a, c, p, source_time=2.0)
    assert ids == () and changed == g


@pytest.mark.parametrize(
    "failure",
    ["old_classification", "old_selection", "old_part", "duplicate", "already_zero"],
)
def test_policy_does_not_launder_stale_or_ambiguous_inputs(failure):
    g, a, c, p = inputs()
    if failure == "old_classification":
        c["source_timestamp_seconds"] = 1.9
    if failure == "old_selection":
        p["encounter_selected_timestamp"] = -10.0
    if failure == "old_part":
        a[1]["human_part_control_policy_prior"]["association"][
            "source_timestamp_seconds"
        ] = 1.9
    if failure == "duplicate":
        a.append(a[0])
    if failure == "already_zero":
        a[0]["lambda"] = 0.0
    with pytest.raises(ValueError):
        apply_encounter_guidance(g, a, c, p, source_time=2.0)

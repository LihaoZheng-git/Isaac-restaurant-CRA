"""An extra enrollment rejection must never become a relaxed support proof."""
from dataclasses import replace

import numpy as np
import pytest

from test_image_object_association import fixture, proposal
from robot.perception.image_object_association import ImageObjectConfig, StationaryImageObjects, associate_image_objects


def observe(state, xyz, time, mask=None):
    default, projection, _, ids = fixture()
    candidates, _ = associate_image_objects([proposal(default if mask is None else mask)],
        projection, ids, xyz, state.config)
    accepted = state.update(candidates, xyz, time)
    return accepted, candidates[0]["diagnostic"]


def test_one_bad_sample_cannot_create_competing_history_or_refresh_evidence():
    _, _, xyz, _ = fixture()
    states = [StationaryImageObjects(ImageObjectConfig(hold_overlapping_hypotheses=enabled)) for enabled in (False, True)]
    bad = xyz.copy()
    bad[-1, 2] += .08  # Whole-surface mismatch; every other point stays valid.
    for state in states:
        for t in (1., 1.1, 1.2):
            accepted, _ = observe(state, xyz, t)
        key = accepted[0]["object_id"]
        before = state.records[key].copy()
        assert not observe(state, bad, 1.3)[0]
        assert state.records[key]["first"] == before["first"]
        assert state.records[key]["last"] == before["last"]
        assert state.records[key]["count"] == before["count"]
    # The next intermediate sample fits both histories in legacy mode. The
    # opt-in retains only the original, independently fully satisfied proof.
    middle = xyz.copy()
    middle[-1, 2] += .04
    assert not observe(states[0], middle, 1.4)[0]
    accepted, _ = observe(states[1], middle, 1.4)
    assert len(accepted) == 1 and len(states[1].records) == 1
    assert accepted[0]["diagnostic"]["original_source_time"] == 1.
    np.testing.assert_array_equal(states[1].records[key]["points"], xyz)


def test_repeated_invalid_overlap_never_qualifies_or_extends_origin():
    _, _, xyz, _ = fixture()
    state = StationaryImageObjects(ImageObjectConfig(hold_overlapping_hypotheses=True))
    for t in (1., 1.1, 1.2):
        observe(state, xyz, t)
    bad = xyz.copy()
    bad[-1, 2] += .08
    for t in (1.3, 1.4, 1.8, 2., 4., 8.):
        accepted, audit = observe(state, bad, t)
        assert not accepted and "overlaps an original" in audit["temporal_rejection"]
        assert len(state.records) == 1
        old = next(iter(state.records.values()))
        assert old["first"] == 1. and old["last"] == 1.2
    # Expiry removes the original, but cannot turn one new source into a proof.
    assert not observe(state, bad, 11.1)[0]
    assert next(iter(state.records.values()))["first"] == 11.1


@pytest.mark.parametrize("mode", ["separate_depth", "distinct_image"])
def test_separate_object_still_requires_new_three_source_proof(mode):
    mask, _, xyz, _ = fixture()
    state = StationaryImageObjects(ImageObjectConfig(hold_overlapping_hypotheses=True))
    for t in (1., 1.1, 1.2):
        observe(state, xyz, t)
    if mode == "separate_depth":
        xyz = xyz + [1., 0., 0.]
    else:
        # Contains the observed points, but not an equivalent visual instance.
        mask[40:, :] = False
        mask[1:15, 15:65] = True
    assert not observe(state, xyz, 1.3, mask)[0]
    assert not observe(state, xyz, 1.4, mask)[0]
    accepted, _ = observe(state, xyz, 1.5, mask)
    assert len(accepted) == 1 and len(state.records) == 2
    assert accepted[0]["diagnostic"]["original_source_time"] == 1.3


def test_support_gap_and_multiple_existing_matches_still_reject():
    _, _, xyz, _ = fixture()
    state = StationaryImageObjects(ImageObjectConfig(hold_overlapping_hypotheses=True))
    for t in (1., 1.1, 1.2):
        observe(state, xyz, t)
    assert not observe(state, xyz, 2.)[0]
    assert not observe(state, xyz, 2.1)[0]
    assert len(observe(state, xyz, 2.2)[0]) == 1
    old = next(iter(state.records.values()))
    state.records[123] = {**old, "points": old["points"].copy(), "mask": old["mask"].copy()}
    accepted, audit = observe(state, xyz, 2.3)
    assert not accepted and audit["temporal_rejection"] == "partial or ambiguous current object"


def test_hold_requires_an_explicit_boolean():
    assert ImageObjectConfig().hold_overlapping_hypotheses is False
    with pytest.raises(TypeError, match="boolean"):
        replace(ImageObjectConfig(), hold_overlapping_hypotheses=1)

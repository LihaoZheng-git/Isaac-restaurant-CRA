import pytest

from scripts.evaluation.compare_bin_crate_scores import compare_bins, live_observations


def field(x, *scores):
    return dict(state_at_commit=[x, 0, 0], targets=[
        dict(cue=value, contact=value, phi=value, kinetic=.1, mass=2., relative_speed=None)
        for value in scores])


def test_matched_progress_excludes_unshared_intervals_and_retains_empty_rows():
    bin_targets, bin_bins = live_observations([field(.10, .1), field(.12), field(.26, .2), field(.55, .9)])
    _, crate_bins = live_observations([field(.11, .5), field(.14, .5), field(.26, .4), field(.80)])
    assert len(bin_targets) == 3
    result = compare_bins(bin_bins, crate_bins)
    assert result['common_intervals'] == 2
    assert result['crate_minus_bin_within_interval_medians']['phi']['median'] == pytest.approx(.3)
    first = result['intervals'][0]['bin']
    assert (first['total'], first['qualified'], first['available_speed']) == (2, 1, 0)
    assert result['intervals'][2]['crate']['scores']['phi'] is None
    assert result['intervals'][3]['crate']['total'] == 1


def test_disjoint_evidence_cannot_establish_a_ranking():
    _, left = live_observations([field(.1, .2)])
    _, right = live_observations([field(.5, .3)])
    result = compare_bins(left, right)
    assert result['common_intervals'] == 0
    assert result['crate_minus_bin_within_interval_medians']['contact'] is None


def test_two_retained_target_ids_in_same_field_are_ambiguous():
    with pytest.raises(ValueError, match='ambiguous'):
        live_observations([field(.1, .2, .3)])

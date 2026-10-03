import pytest

from robot.perception.relative_heaviness import HeavinessPrediction, evaluate_relative_heaviness


def row(name, label, target, prediction):
    return HeavinessPrediction(name, label, target, prediction)


def test_reversals_ties_and_class_scale_are_reported_without_selecting_successes():
    result = evaluate_relative_heaviness((
        row("a", "bowl", 1, 2), row("b", "bowl", 4, 1), row("c", "chair", 20, 2),
    ))
    assert result["all_pairs"] == {
        "pair_count": 3, "correct": 1, "reversed": 1, "ties": 1, "ordering_accuracy": 1 / 3,
    }
    assert result["within_class"]["ordering_accuracy"] == 0
    assert result["sensor_domain_acceptance"] is False


def test_close_pairs_are_excluded_and_no_pairs_has_no_accuracy():
    result = evaluate_relative_heaviness((row("a", "bowl", 1, 1), row("b", "bowl", 2, 3)))
    assert result["all_pairs"]["pair_count"] == 0
    assert result["all_pairs"]["ordering_accuracy"] is None


def test_duplicate_objects_and_invalid_masses_rejected():
    with pytest.raises(ValueError, match="independent"):
        evaluate_relative_heaviness((row("a", "bowl", 1, 1), row("a", "bowl", 4, 3)))
    with pytest.raises(ValueError, match="positive"):
        row("a", "bowl", float("nan"), 2)

"""Historical response reports remain readable without retaining misleading labels."""

from copy import deepcopy

import pytest

from scripts.evaluation.report_restaurant_calibration import archived_response_keys


def test_legacy_report_keys_are_read_without_modifying_original_values():
    original = {
        "calibration_bin_live": [0.3],
        "heldout_bin_retained": [{"phi": 0.4}],
        "heldout_verified_human": [{"phi": 0.8}],
    }
    before = deepcopy(original)
    renamed = archived_response_keys(original)
    assert set(renamed) == {
        "calibration_bin_live", "archived_v3_bin_retained", "archived_v3_verified_human"
    }
    assert renamed["archived_v3_bin_retained"] is original["heldout_bin_retained"]
    assert original == before


def test_current_report_keys_remain_unchanged():
    current = {"archived_v3_crate_retained": [0.5], "calibration_crate_live": [0.6]}
    assert archived_response_keys(current) == current


def test_mixed_duplicate_report_keys_are_not_silently_overwritten():
    with pytest.raises(ValueError, match="ambiguous legacy/current"):
        archived_response_keys({"heldout_bin_retained": [0.3], "archived_v3_bin_retained": [0.4]})

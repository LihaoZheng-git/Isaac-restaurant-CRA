import numpy as np
import pytest

from robot.perception.visual_mass_isaac_inventory import (
    ISAAC_MASS_TEST_INVENTORY,
    IsaacMassObject,
    mask_for_semantic_class,
)


def test_inventory_meets_minimum_independent_object_and_class_counts():
    assert len(ISAAC_MASS_TEST_INVENTORY) == 10
    assert len({item.object_id for item in ISAAC_MASS_TEST_INVENTORY}) == 10
    counts = {
        name: sum(item.semantic_class == name for item in ISAAC_MASS_TEST_INVENTORY)
        for name in {item.semantic_class for item in ISAAC_MASS_TEST_INVENTORY}
    }
    assert set(counts.values()) == {2}


def test_semantic_mask_accepts_replicator_label_mappings():
    segmentation = np.array([[0, 4], [4, 7]], dtype=np.uint32)

    mask = mask_for_semantic_class(
        segmentation,
        {"4": {"class": "cup"}, 7: "table"},
        "cup",
    )

    assert mask.dtype == np.bool_
    assert mask.tolist() == [[False, True], [True, False]]


def test_semantic_mask_rejects_missing_or_empty_class():
    with pytest.raises(ValueError, match="missing"):
        mask_for_semantic_class(np.zeros((2, 2)), {0: "background"}, "cup")
    with pytest.raises(ValueError, match="no visible pixels"):
        mask_for_semantic_class(np.zeros((2, 2)), {4: "cup"}, "cup")


def test_fixture_rejects_invalid_physical_values():
    with pytest.raises(ValueError, match="positive"):
        IsaacMassObject("cup-03", "cup", (0.1, 0.1, 0.0), 1.0, (1, 1, 1), "cylinder")

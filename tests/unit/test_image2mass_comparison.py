"""Offline compatibility checks that need neither downloaded weights nor Isaac."""

import pickle

import numpy as np
import pytest
import torch

from scripts.evaluation.compare_image2mass import (
    crop_object,
    independent_objects,
    summarize_rows,
    view_variation,
)
from scripts.evaluation.export_image2mass import density_constants
from scripts.evaluation.image2mass_compat import (
    Image2MassOffline,
    KG_PER_POUND,
    METRES_PER_INCH,
    NumericGraph,
    auxiliary_features,
    d4,
    legacy_rectangle,
    tf_same_pad,
)


@pytest.mark.parametrize("index", range(8))
def test_dihedral_transforms_restore_corresponding_pixels(index):
    image = np.arange(27).reshape(3, 3, 3)
    np.testing.assert_array_equal(d4(d4(image, index), index, inverse=True), image)


def test_legacy_axis_aligned_rectangle_keeps_geometry_and_restores_side_order():
    assert legacy_rectangle(((3, 4), (2, 8), -90.0)) == ((3, 4), (8, 2), 0.0)
    assert legacy_rectangle(((3, 4), (2, 8), -30.0)) == ((3, 4), (2, 8), -30.0)


def test_tensorflow_same_padding_places_odd_extra_pad_on_bottom_right():
    image = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    padded = tf_same_pad(image, (3, 3), (2, 2))
    pooled = torch.nn.functional.max_pool2d(padded, 3, 2)
    torch.testing.assert_close(pooled, torch.tensor([[[[10.0, 11.0], [14.0, 15.0]]]]))
    torch.testing.assert_close(padded[:, :, :4, :4], image)


def test_geometry_feature_units_and_positions():
    features = auxiliary_features(
        np.asarray([2, 3, 4]), np.ones((19, 19)), ((15, 16), (31, 30), -45)
    )
    assert features.shape == (35,)
    assert features[0] == 24
    np.testing.assert_array_equal(features[1:4], [2, 3, 4])
    np.testing.assert_array_equal(features[4:7], [31, 30, 32])
    np.testing.assert_array_equal(features[7:10], [15, 16, -45])
    assert features[10] == 361 * 32
    np.testing.assert_array_equal(features[22:25], [4, 3, 2])


@pytest.mark.parametrize("dims", ([0, 1, 1], [1, np.nan, 1], [1, 1]))
def test_invalid_dimensions_are_rejected(dims):
    with pytest.raises(ValueError, match="dimensions"):
        auxiliary_features(np.asarray(dims), np.ones((19, 19)), ((1, 1), (3, 3), 0))


def test_missing_thickness_is_not_replaced_by_zero_mass():
    with pytest.raises(ValueError, match="foreground"):
        auxiliary_features(np.ones(3), np.zeros((19, 19)), ((1, 1), (3, 3), 0))


def test_density_parser_accepts_only_numeric_list_without_unpickling():
    np.testing.assert_allclose(
        density_constants(pickle.dumps([0.1, 0.2], protocol=4)), [0.1, 0.2]
    )
    for value in (
        [0.2, 0.1],
        [float("nan"), 1.0],
        [0.1, "x"],
        {"a": 1.0},
        Exception("x"),
    ):
        with pytest.raises(ValueError):
            density_constants(pickle.dumps(value, protocol=4))


def test_unreviewed_lambda_and_unknown_layer_are_never_executed():
    graph = NumericGraph.__new__(NumericGraph)
    layer = {
        "class_name": "Lambda",
        "config": {
            "name": "lambda_1",
            "function": ["unreviewed byte string", None, None],
        },
    }
    with pytest.raises(ValueError, match="unreviewed"):
        graph._layer(layer, [torch.ones(1)], "")
    with pytest.raises(ValueError, match="unsupported layer"):
        graph._layer(
            {"class_name": "Custom", "config": {"name": "x"}}, [torch.ones(1)], ""
        )


def test_si_conversion_and_eight_transform_median():
    estimator = Image2MassOffline.__new__(Image2MassOffline)
    estimator.geometry = lambda rgb: (np.ones((19, 19)), ((1, 1), (3, 3), 0))
    captured = {}

    def mass(inputs):
        captured["features"] = inputs[2]
        return np.arange(1, 9, dtype=np.float32)[:, None]

    estimator.mass = mass
    result = estimator.predict(
        np.ones((20, 20, 3), np.uint8), np.asarray([2, 3, 4]) * METRES_PER_INCH
    )
    assert result["mass_kg"] == pytest.approx(4.5 * KG_PER_POUND)
    assert captured["features"].shape == (8, 35)
    np.testing.assert_allclose(captured["features"][:, 1:4], [[2, 3, 4]] * 8)


def test_common_mask_uses_native_background_without_changing_object_pixels():
    image = np.arange(75, dtype=np.uint8).reshape(5, 5, 3)
    mask = np.zeros((5, 5), bool)
    mask[1:4, 1:4] = True
    mask[2, 2] = False
    gray, white = crop_object(image, mask, 127), crop_object(image, mask, 255)
    np.testing.assert_array_equal(gray[0, 0], image[1, 1])
    np.testing.assert_array_equal(gray[1, 1], [127] * 3)
    np.testing.assert_array_equal(white[1, 1], [255] * 3)


def test_repeated_views_do_not_inflate_primary_independent_count():
    rows = [
        {"object_id": "a", "reference_mass_kg": 1, "mass": 2},
        {"object_id": "a", "reference_mass_kg": 1, "mass": 4},
        {"object_id": "b", "reference_mass_kg": 4, "mass": 1},
    ]
    collapsed = independent_objects(rows, "mass")
    assert len(collapsed) == 2
    metrics = summarize_rows(collapsed, "mass")
    assert metrics["twofold_pairs"] == 1
    assert metrics["twofold_pair_order_accuracy"] == 0
    assert view_variation(rows, "mass")["worst_max_min_ratio"] == 2

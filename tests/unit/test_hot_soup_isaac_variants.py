import pytest

from robot.perception.hot_soup_isaac_variants import (
    HOT_SOUP_ISAAC_VARIANTS,
    HotSoupIsaacVariant,
    select_hot_soup_isaac_variants,
)


def test_default_selection_returns_all_variants_in_registry_order():
    selected = select_hot_soup_isaac_variants(None)

    assert selected == HOT_SOUP_ISAAC_VARIANTS
    assert [variant.variant_id for variant in selected] == [
        "baseline",
        "wispy-right",
        "broad-left",
    ]


def test_explicit_selection_uses_registry_order_and_rejects_unknown_ids():
    selected = select_hot_soup_isaac_variants("broad-left,baseline")

    assert [variant.variant_id for variant in selected] == ["baseline", "broad-left"]
    with pytest.raises(ValueError, match="unknown"):
        select_hot_soup_isaac_variants("missing")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"variant_id": "bad id"}, "variant_id"),
        ({"plume_scale": 0.0}, "plume_scale"),
        ({"plume_opacity": 0.0}, "plume_opacity"),
        ({"plume_color": (1.1, 0.0, 0.0)}, "plume_color"),
        ({"light_intensity": 0.0}, "light_intensity"),
    ],
)
def test_variant_rejects_invalid_scene_parameters(overrides, message):
    values = {
        "variant_id": "valid",
        "plume_scale": 1.0,
        "plume_lateral_offset_m": 0.0,
        "plume_opacity": 0.5,
        "plume_color": (0.5, 0.5, 0.5),
        "table_color": (0.5, 0.5, 0.5),
        "light_intensity": 100.0,
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        HotSoupIsaacVariant(**values)

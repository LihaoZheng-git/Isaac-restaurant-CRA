"""Deterministic authored-scene variants for the stationary hot-soup fixture.

These settings control synthetic appearance only. They are not temperature
measurements and do not establish that a visible plume is physically steam.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HotSoupIsaacVariant:
    """Appearance parameters for one stationary Isaac capture."""

    variant_id: str
    plume_scale: float
    plume_lateral_offset_m: float
    plume_opacity: float
    plume_color: tuple[float, float, float]
    table_color: tuple[float, float, float]
    light_intensity: float

    def __post_init__(self) -> None:
        if not self.variant_id or not self.variant_id.replace("-", "").isalnum():
            raise ValueError("variant_id must contain letters, digits, or hyphens")
        if self.plume_scale <= 0.0:
            raise ValueError("plume_scale must be positive")
        if not 0.0 < self.plume_opacity <= 1.0:
            raise ValueError("plume_opacity must be in (0, 1]")
        if self.light_intensity <= 0.0:
            raise ValueError("light_intensity must be positive")
        for name, color in (
            ("plume_color", self.plume_color),
            ("table_color", self.table_color),
        ):
            if len(color) != 3 or any(channel < 0.0 or channel > 1.0 for channel in color):
                raise ValueError(f"{name} channels must be in [0, 1]")


HOT_SOUP_ISAAC_VARIANTS = (
    HotSoupIsaacVariant(
        variant_id="baseline",
        plume_scale=1.0,
        plume_lateral_offset_m=0.0,
        plume_opacity=0.28,
        plume_color=(0.92, 0.94, 0.97),
        table_color=(0.35, 0.16, 0.06),
        light_intensity=1000.0,
    ),
    HotSoupIsaacVariant(
        variant_id="wispy-right",
        plume_scale=0.68,
        plume_lateral_offset_m=0.09,
        plume_opacity=0.16,
        plume_color=(0.80, 0.86, 0.91),
        table_color=(0.18, 0.20, 0.22),
        light_intensity=850.0,
    ),
    HotSoupIsaacVariant(
        variant_id="broad-left",
        plume_scale=1.22,
        plume_lateral_offset_m=-0.08,
        plume_opacity=0.38,
        plume_color=(0.98, 0.91, 0.82),
        table_color=(0.48, 0.31, 0.16),
        light_intensity=1250.0,
    ),
)


def select_hot_soup_isaac_variants(selection: str | None) -> tuple[HotSoupIsaacVariant, ...]:
    """Resolve a comma-separated selection, preserving registry order."""

    by_id = {variant.variant_id: variant for variant in HOT_SOUP_ISAAC_VARIANTS}
    if selection is None or not selection.strip() or selection.strip().lower() == "all":
        return HOT_SOUP_ISAAC_VARIANTS
    requested = {item.strip() for item in selection.split(",") if item.strip()}
    unknown = requested - by_id.keys()
    if unknown:
        raise ValueError(f"unknown hot-soup Isaac variants: {sorted(unknown)}")
    if not requested:
        raise ValueError("at least one hot-soup Isaac variant is required")
    return tuple(variant for variant in HOT_SOUP_ISAAC_VARIANTS if variant.variant_id in requested)

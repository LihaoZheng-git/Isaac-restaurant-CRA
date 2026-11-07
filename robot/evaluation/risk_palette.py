"""Fixed score-to-colour mappings; presentation never changes a risk value."""

import math

import numpy as np


PALETTES = {
    "legacy": ((0., .5, .75, 1.),
               ((35, 195, 85), (245, 220, 45), (245, 135, 35), (225, 45, 45))),
    "restaurant_v1": ((0., .30, .45, .60, .75, .85, 1.),
                      ((26, 160, 72), (54, 196, 84), (152, 207, 50),
                       (245, 210, 45), (245, 132, 32), (231, 70, 40), (163, 24, 42))),
}


def risk_color_rgb(phi, *, palette="legacy"):
    """Interpolate a declared fixed scale; never normalize to visible objects."""
    if not math.isfinite(phi) or not 0 <= phi <= 1:
        raise ValueError("phi must be finite and within [0, 1]")
    if palette not in PALETTES:
        raise ValueError("unknown fixed risk palette")
    stops, colors = PALETTES[palette]
    colors = np.asarray(colors)
    return tuple(int(round(np.interp(phi, stops, colors[:, i]))) for i in range(3))


def palette_description(name):
    if name not in PALETTES:
        raise ValueError("unknown fixed risk palette")
    stops, colors = PALETTES[name]
    return dict(name=name, score_knots=list(stops), rgb_knots=[list(c) for c in colors],
                scope="fixed piecewise-linear RGB colour interpolation; phi values unchanged")

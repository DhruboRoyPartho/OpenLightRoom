import numpy as np

from core.processing.color_space import smoothstep


def circular_hue_distance(hue_deg: np.ndarray, center_deg: float) -> np.ndarray:
    """Shortest distance between `hue_deg` (any real values - wrapped onto
    [0, 360) internally) and `center_deg`, going around the hue circle in
    whichever direction is shorter. Always in [0, 180]. This is what makes
    a hue mask centered near 0 degrees (red) correctly include hues near
    359 degrees, instead of treating them as maximally far apart."""
    return np.abs((hue_deg - center_deg + 180.0) % 360.0 - 180.0)


def hue_range_mask(hue_deg: np.ndarray, center_deg: float, width_deg: float, feather_deg: float) -> np.ndarray:
    """1.0 for hues within +/- width_deg of center_deg, smoothly easing to
    0.0 over an additional feather_deg beyond that - a smooth color-distance
    mask (per-pixel distance from a target hue) rather than a hard bin
    edge, which is what lets adjacent HSL channels (e.g. Orange and Yellow)
    blend into each other instead of showing a visible seam.
    """
    distance = circular_hue_distance(hue_deg, center_deg)
    outer = width_deg + max(feather_deg, 1e-6)
    t = np.clip((outer - distance) / max(outer - width_deg, 1e-6), 0.0, 1.0)
    return smoothstep(t)
